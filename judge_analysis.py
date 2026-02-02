#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, Any, Optional, List

LABELS = [
    "caution_alert",
    "productive_launchpoint",
    "blind_follow_trajectory",
    "structural_shortcut",
]

# Short, strict rubric
RUBRIC = """
Classify the cheatsheet into exactly ONE label:

- caution_alert: warns against a pitfall / wrong approach.
- productive_launchpoint: suggests a useful framing/reformulation; not a full recipe.
- blind_follow_trajectory: step-by-step procedure/recipe to follow.
- structural_shortcut: key invariant/insight that collapses many steps into one leap.

Return ONLY valid JSON:
{"class":"<one of: caution_alert|productive_launchpoint|blind_follow_trajectory|structural_shortcut>","rationale":"1-2 sentences"}
""".strip()


def build_prompt(cheatsheet: str) -> str:
    return f"{RUBRIC}\n\nCheatsheet:\n\"\"\"{cheatsheet}\"\"\""


def robust_json_extract(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # extract first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0).strip()

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None


def normalize_label(label: str) -> Optional[str]:
    if not label:
        return None
    label = label.strip().lower().replace(" ", "_").replace("-", "_")
    return label if label in LABELS else None


def classify_with_transformers(
    model: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    max_input_tokens: int,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        device_map="auto",
    )
    mdl.eval()

    # Some models lack pad token; make padding safe
    if tok.pad_token_id is None:
        # common safe default: use eos token as pad
        tok.pad_token = tok.eos_token

    has_chat_template = hasattr(tok, "chat_template") and tok.chat_template is not None

    results: List[str] = []

    # Optional speed tweak on A100: allow TF32 matmul
    torch.backends.cuda.matmul.allow_tf32 = True

    # Inference mode reduces overhead
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]

            if has_chat_template:
                chats = [[{"role": "user", "content": p}] for p in chunk]
                input_ids = tok.apply_chat_template(
                    chats,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(mdl.device)
                attention_mask = (input_ids != tok.pad_token_id).long()
            else:
                enc = tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(mdl.device)
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

            gen = mdl.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                # helps prevent rambling
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )

            # decode only new tokens
            for j in range(gen.shape[0]):
                in_len = input_ids[j].shape[0]
                new_ids = gen[j][in_len:]
                results.append(tok.decode(new_ids, skip_special_tokens=True))

    return results


def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=5000)

    # Transformers-only
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=96)   # shorter is faster
    ap.add_argument("--max_input_tokens", type=int, default=2*4096)  # cap long cheatsheets

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--write_debug", action="store_true")
    args = ap.parse_args()

    # Load records
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "row_idx" not in obj or "cheatsheet" not in obj:
                continue
            records.append(obj)
            if len(records) >= args.limit:
                break

    prompts = [build_prompt(r["cheatsheet"]) for r in records]

    texts = classify_with_transformers(
        model=args.model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
    )

    assert len(texts) == len(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    bad = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for r, t in zip(records, texts):
            parsed = robust_json_extract(t)
            label = None
            rationale = None
            if parsed:
                label = normalize_label(parsed.get("class", ""))
                rationale = parsed.get("rationale", None)

            if label is None:
                bad += 1
                label = "productive_launchpoint"

            out_obj = {
                "row_idx": r["row_idx"],
                "cheatsheet": r["cheatsheet"],
                "class": label,
            }
            if args.write_debug:
                out_obj["rationale"] = rationale
                out_obj["raw_output"] = t

            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(records)} rows to {args.output}. Malformed outputs: {bad}.")


if __name__ == "__main__":
    main()

