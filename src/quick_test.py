#!/usr/bin/env python3
import os
import re
import json
import argparse
import statistics as stats
from collections import defaultdict
from typing import Dict, Any, Optional
import hashlib
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Stats helpers (same as yours)
# -------------------------
def summarize(nums):
    if not nums:
        return {"n": 0}
    nums_sorted = sorted(nums)

    def pct(p):
        i = int(round((p/100) * (len(nums_sorted)-1)))
        return nums_sorted[max(0, min(i, len(nums_sorted)-1))]

    return {
        "n": len(nums),
        "mean": sum(nums)/len(nums),
        "median": stats.median(nums_sorted),
        "p10": pct(10),
        "p90": pct(90),
        "min": nums_sorted[0],
        "max": nums_sorted[-1],
    }

def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

# -------------------------
# Local HF Judge
# -------------------------
JUDGE_SYSTEM = (
    "You are a strict classifier.\n"
    "You will be given a model's solution output to a math word problem.\n"
    "Decide whether the model REFUSES / says the answer cannot be determined due to missing information, "
    "or instead attempts to provide an answer.\n\n"
    "Return ONLY ONE TOKEN as your entire output: REFUSAL or NON_REFUSAL or UNCLEAR.\n\n"
    "Guidelines:\n"
    "- REFUSAL: says 'insufficient information', 'not enough info', 'cannot be determined', "
    "  'missing information', 'not specified', or explicitly refuses to answer.\n"
    "- NON_REFUSAL: provides or commits to a concrete answer (number/expression), or assumes missing info.\n"
    "- UNCLEAR: empty output or too cut off/garbled to tell.\n"
    "- If it includes both refusal language AND a final concrete answer, output NON_REFUSAL.\n"
)

def chat_encode(tokenizer, system: str, user: str):
    if getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    raw = f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"
    return tokenizer(raw, return_tensors="pt")["input_ids"]

@torch.inference_mode()
def judge_with_hf(
    model,
    tokenizer,
    output_text: str,
    max_new_tokens: int = 3,
) -> Dict[str, Any]:
    user = f"MODEL_OUTPUT:\n{output_text}"
    input_ids = chat_encode(tokenizer, JUDGE_SYSTEM, user).to(model.device)

    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    gen = out[0, input_ids.shape[-1]:]
    txt = tokenizer.decode(gen, skip_special_tokens=True).strip()

    # Normalize to one of three labels
    t = txt.upper()
    if "REFUSAL" in t:
        label = "REFUSAL"
    elif "NON_REFUSAL" in t or "NON-REFUSAL" in t or "NONREFUSAL" in t:
        label = "NON_REFUSAL"
    elif "UNCLEAR" in t:
        label = "UNCLEAR"
    else:
        # If model outputs something unexpected, treat as UNCLEAR
        label = "UNCLEAR"

    return {"label": label, "raw": txt[:200]}

# -------------------------
# Main (original logic preserved)
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=str, required=True, help="Path to records.jsonl")
    ap.add_argument("--perturbation", type=str, default="critical thinking")
    ap.add_argument("--baseline", type=str, default="baseline")
    ap.add_argument("--rlad", type=str, default="rlad_main")
    ap.add_argument("--topk", type=int, default=15)

    # Judge model (local)
    ap.add_argument("--judge_model_id", type=str, required=True,
                    help="HF model id for local judge (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--max_new_tokens_judge", type=int, default=3)

    # Cache
    ap.add_argument("--judge_cache", type=str, default="judge_cache.jsonl")
    ap.add_argument("--rejudge", action="store_true")

    # Optional tokenizer for cheatsheet token counts (your original feature)
    ap.add_argument("--tokenizer", type=str, default="", help="Optional HF model id for token counts (loads tokenizer only)")
    args = ap.parse_args()

    # Load optional tokenizer for token-count stats
    tok_len = None
    if args.tokenizer:
        tok_len = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    
    # Load judge cache
    cache: Dict[str, Dict[str, Any]] = {}
    if (not args.rejudge) and args.judge_cache and os.path.exists(args.judge_cache):
        with open(args.judge_cache, "r", encoding="utf-8") as cf:
            for line in cf:
                try:
                    o = json.loads(line)
                    cache[o["key"]] = o
                except Exception:
                    continue

    # Load local judge model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    judge_tok = AutoTokenizer.from_pretrained(args.judge_model_id, trust_remote_code=True)
    judge_mdl = AutoModelForCausalLM.from_pretrained(
        args.judge_model_id,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=True,
    )
    judge_mdl.eval()

    def cached_judge(text: str) -> Dict[str, Any]:
        key = sha1_text(text)
        if (not args.rejudge) and key in cache:
            return cache[key]["result"]
        res = judge_with_hf(judge_mdl, judge_tok, text, max_new_tokens=args.max_new_tokens_judge)
        rec = {"key": key, "ts": time.time(), "result": res}
        cache[key] = rec
        if args.judge_cache:
            with open(args.judge_cache, "a", encoding="utf-8") as cf:
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return res

    # row_idx -> pipeline -> record
    by_row = defaultdict(dict)

    with open(args.records, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("perturbation_type") or "").strip().lower() != args.perturbation.strip().lower():
                continue
            row = r.get("row_idx")
            pipe = r.get("pipeline")
            if row is None or pipe is None:
                continue
            by_row[row][pipe] = r

    # Buckets (same as your original)
    buckets = {
        "both_correct": [],
        "rlad_wins": [],
        "baseline_wins": [],
        "both_wrong": [],
        "missing_pair": [],
    }

    # Length stats (same structure as your original)
    lens = {b: {"base_chars": [], "rlad_chars": [], "delta_chars": [],
               "base_words": [], "rlad_words": [], "delta_words": [],
               "base_toks": [], "rlad_toks": [], "delta_toks": []}
            for b in buckets}

    def get_lengths(rec):
        cs = rec.get("cheatsheet") or ""
        ch = len(cs)
        wd = len(cs.split())
        tk = None
        if tok_len is not None:
            tk = len(tok_len.encode(cs, add_special_tokens=False))
        return ch, wd, tk

    # Store judge results for printing
    judged = defaultdict(dict)

    paired_n = 0
    missing_pairs = 0

    for row, d in by_row.items():
        if args.baseline not in d or args.rlad not in d:
            buckets["missing_pair"].append(row)
            missing_pairs += 1
            continue

        paired_n += 1
        rb = d[args.baseline]
        rr = d[args.rlad]

        b_out = rb.get("solver_output", "") or ""
        r_out = rr.get("solver_output", "") or ""

        jb = cached_judge(b_out)
        jr = cached_judge(r_out)

        judged[row][args.baseline] = jb
        judged[row][args.rlad] = jr

        # For critical thinking: "correct" == REFUSAL
        b_ok = (jb["label"] == "REFUSAL")
        r_ok = (jr["label"] == "REFUSAL")

        if b_ok and r_ok:
            bucket = "both_correct"
        elif (not b_ok) and r_ok:
            bucket = "rlad_wins"
        elif b_ok and (not r_ok):
            bucket = "baseline_wins"
        else:
            bucket = "both_wrong"

        buckets[bucket].append(row)

        # Same cheatsheet length stats as your original
        bch, bwd, btk = get_lengths(rb)
        rch, rwd, rtk = get_lengths(rr)

        lens[bucket]["base_chars"].append(bch)
        lens[bucket]["rlad_chars"].append(rch)
        lens[bucket]["delta_chars"].append(rch - bch)

        lens[bucket]["base_words"].append(bwd)
        lens[bucket]["rlad_words"].append(rwd)
        lens[bucket]["delta_words"].append(rwd - bwd)

        if tok_len is not None:
            lens[bucket]["base_toks"].append(btk)
            lens[bucket]["rlad_toks"].append(rtk)
            lens[bucket]["delta_toks"].append(rtk - btk)

    print(f"paired_n: {paired_n} | missing_pairs: {len(buckets['missing_pair'])}")
    for k in ["both_correct", "rlad_wins", "baseline_wins", "both_wrong"]:
        print(f"{k}: {len(buckets[k])}")

    def print_bucket_stats(name):
        print("\n" + "="*80)
        print(name)
        print("- chars (baseline / rlad / delta=rlad-baseline)")
        print("  base:", summarize(lens[name]["base_chars"]))
        print("  rlad:", summarize(lens[name]["rlad_chars"]))
        print("  delt:", summarize(lens[name]["delta_chars"]))
        print("- words")
        print("  base:", summarize(lens[name]["base_words"]))
        print("  rlad:", summarize(lens[name]["rlad_words"]))
        print("  delt:", summarize(lens[name]["delta_words"]))
        if tok_len is not None:
            print("- tokens")
            print("  base:", summarize(lens[name]["base_toks"]))
            print("  rlad:", summarize(lens[name]["rlad_toks"]))
            print("  delt:", summarize(lens[name]["delta_toks"]))

    # same reporting focus
    print_bucket_stats("baseline_wins")
    print_bucket_stats("rlad_wins")

    # Print examples from baseline_wins (like your original), but show judge labels and last 400 chars
    bw_rows = buckets["baseline_wins"]
    if bw_rows:
        def row_score(row):
            rb = by_row[row][args.baseline]
            rr = by_row[row][args.rlad]
            bch = len((rb.get("cheatsheet") or ""))
            rch = len((rr.get("cheatsheet") or ""))
            return (rch - bch, rch)

        bw_rows_sorted = sorted(bw_rows, key=row_score, reverse=True)

        def tail(x, n=400):
            t = (x or "").strip().replace("\r", "")
            return t if len(t) <= n else ("… " + t[-n:])

        print("\n" + "="*80)
        print(f"Top {min(args.topk, len(bw_rows_sorted))} baseline_wins with largest RLAD cheatsheet delta (chars):")
        for row in bw_rows_sorted[: args.topk]:
            rb = by_row[row][args.baseline]
            rr = by_row[row][args.rlad]
            bch = len((rb.get("cheatsheet") or ""))
            rch = len((rr.get("cheatsheet") or ""))

            jb = judged[row][args.baseline]
            jr = judged[row][args.rlad]

            print(f"\nrow={row}  delta={rch-bch:+5d}  base_cs={bch:5d}  rlad_cs={rch:5d}")
            print(f"  BASELINE_JUDGE: {jb['label']} (raw='{jb.get('raw','')}')")
            print(tail(rb.get("solver_output", "")))
            print("-"*80)
            print(f"  RLAD_JUDGE:     {jr['label']} (raw='{jr.get('raw','')}')")
            print(tail(rr.get("solver_output", "")))
            print("-"*80)

if __name__ == "__main__":
    main()
