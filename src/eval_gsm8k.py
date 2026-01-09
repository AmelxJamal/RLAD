import os
import re
import json
import gc
import time
import random
from dataclasses import dataclass
from collections import defaultdict
from fractions import Fraction
from typing import Dict, Any, Optional, List, Tuple
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback


# -------------------- Defaults --------------------
DEFAULT_PIPELINES = [
    {
        "name": "baseline_non_rlad",
        "hint_model_id": os.getenv("BASE_HINT_ID", "Qwen/Qwen3-4B"),
        "solver_model_id": os.getenv("BASE_SOL_ID", "Qwen/Qwen3-1.7B"),
    },
    {
        "name": "rlad",
        "hint_model_id": os.getenv("RLAD_HINT_ID", "CMU-AIRe/RLAD-Hint-Gen"),
        "solver_model_id": os.getenv("RLAD_SOL_ID", "CMU-AIRe/RLAD-Sol-Gen"),
    },
]


# -------------------- Utils --------------------
def torch_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown DTYPE={dtype_str}")


def chat_encode(tokenizer, system: str, user: str):
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    raw = f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"
    return tokenizer(raw, return_tensors="pt")["input_ids"]


def truncate_left(input_ids: torch.Tensor, keep_tokens: int):
    if input_ids.shape[-1] <= keep_tokens:
        return input_ids
    return input_ids[:, -keep_tokens:]


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    system: str,
    user: str,
    max_context_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    input_ids = chat_encode(tokenizer, system=system, user=user)
    keep = max(1, max_context_tokens - max_new_tokens)
    input_ids = truncate_left(input_ids, keep_tokens=keep).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    out = model.generate(input_ids, **gen_kwargs)
    gen = out[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def free_model(mdl, tok):
    del mdl
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_hint_prompt(problem: str) -> Tuple[str, str]:
    system = (
        "You are a hint/abstraction generator for competition math.\n"
        "Given a problem, write a CHEATSHEET of 3–6 short notes that help solve the problem.\n"
        "Rules:\n"
        "- Do NOT reveal or compute the final numeric answer.\n"
        "- Prefer general procedures, identities, pitfalls, and checks.\n"
        "- Output ONLY the cheatsheet notes, no extra commentary.\n"
        "- Use the XML-ish format:\n"
        "  <note1><description>...</description><example>...</example></note1>\n"
        "  <note2>...</note2>\n"
    )
    user = f"PROBLEM:\n{problem}"
    return system, user


def build_solver_prompt(problem: str, cheatsheet: str) -> Tuple[str, str]:
    system = (
        "You are an expert problem-solving assistant.\n"
        "Use the provided cheatsheet (if any) to solve the problem.\n"
        "Give the final answer in the exact form \\boxed{<answer>}.\n"
        "If the answer is a number, do not add units.\n"
    )
    user = f"CHEATSHEET:\n{cheatsheet}\n\nPROBLEM:\n{problem}\n"
    return system, user


# GSM8K answers are numeric; we keep your numeric extraction approach.
_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*(?:/\d[\d,]*\.?\d*)?)")


def _to_fraction(s: str) -> Optional[Fraction]:
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    t = t.replace(",", "").strip()

    # Some models output like "#### 150"
    if t.startswith("####"):
        t = t[4:].strip()

    # Trim to first numeric-looking token
    m = _NUM_RE.search(t)
    if not m:
        return None
    token = m.group(1).replace(",", "")
    try:
        return Fraction(token)  # ints, decimals, simple fractions
    except Exception:
        return None


def extract_pred_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = _BOX_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: last numeric-looking token
    ms = list(_NUM_RE.finditer(text))
    if ms:
        return ms[-1].group(1).strip()
    return None


def extract_gsm8k_gt(answer_text: str) -> str:
    """
    GSM8K 'answer' field typically contains a rationale ending with: '#### <number>'.
    We extract the part after the last '####'. If not present, fall back to raw text.
    """
    if not answer_text:
        return ""
    parts = answer_text.split("####")
    if len(parts) >= 2:
        return parts[-1].strip()
    return answer_text.strip()


def is_correct(pred_text: str, gt_text: str) -> Tuple[bool, Dict[str, Any]]:
    pred_raw = extract_pred_answer(pred_text)
    gt_raw = extract_gsm8k_gt(gt_text)

    pred_frac = _to_fraction(pred_raw) if pred_raw is not None else None
    gt_frac = _to_fraction(gt_raw)

    info = {
        "pred_raw": pred_raw,
        "gt_raw": gt_raw,
        "pred_frac": str(pred_frac) if pred_frac is not None else None,
        "gt_frac": str(gt_frac) if gt_frac is not None else None,
    }
    if pred_frac is None or gt_frac is None:
        return False, info
    return pred_frac == gt_frac, info


@dataclass
class GenConfig:
    dtype: str = os.getenv("DTYPE", "bfloat16")
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))
    max_new_tokens_hint: int = int(os.getenv("MAX_NEW_TOKENS_HINT", "256"))
    max_new_tokens_sol: int = int(os.getenv("MAX_NEW_TOKENS_SOL", "512"))
    do_sample: bool = bool(int(os.getenv("DO_SAMPLE", "0")))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))


def load_model_and_tokenizer(model_id: str, cfg: GenConfig):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype(cfg.dtype),
        device_map=cfg.device_map,
        trust_remote_code=True,
    )
    mdl.eval()
    return mdl, tok


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------- Main evaluation --------------------
def run_pipeline_on_indices(
    *,
    dataset,
    indices: List[int],
    pipeline: Dict[str, str],
    cfg: GenConfig,
    out_jsonl_path: str,
    resume: bool,
):
    # Resume support: skip (pipeline_name, row_idx) already written
    done = set()
    if resume and os.path.exists(out_jsonl_path):
        with open(out_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add((obj.get("pipeline"), obj.get("row_idx")))
                except Exception:
                    pass

    hint_model_id = pipeline["hint_model_id"]
    solver_model_id = pipeline["solver_model_id"]

    hint_model, hint_tok = load_model_and_tokenizer(hint_model_id, cfg)
    sol_model, sol_tok = load_model_and_tokenizer(solver_model_id, cfg)

    stats = {
        "n": 0,
        "correct": 0,
        "parse_fail": 0,
        "no_boxed": 0,
    }

    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for row_idx in tqdm(indices, desc=f"{pipeline['name']}"):
            if (pipeline["name"], row_idx) in done:
                continue

            ex = dataset[row_idx]
            question = (ex.get("question") or "").strip()
            gt_full = (ex.get("answer") or "").strip()  # GSM8K rationale + #### answer
            gt_extracted = extract_gsm8k_gt(gt_full)

            # hint
            hs, hu = build_hint_prompt(question)
            try:
                cheatsheet = generate_text(
                    hint_model, hint_tok, hs, hu,
                    max_context_tokens=cfg.max_context_tokens,
                    max_new_tokens=cfg.max_new_tokens_hint,
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cheatsheet = ""

            # solve
            ss, su = build_solver_prompt(question, cheatsheet)
            try:
                solver_out = generate_text(
                    sol_model, sol_tok, ss, su,
                    max_context_tokens=cfg.max_context_tokens,
                    max_new_tokens=cfg.max_new_tokens_sol,
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                solver_out = ""

            correct, info = is_correct(solver_out, gt_full)

            rec = {
                "ts": time.time(),
                "pipeline": pipeline["name"],
                "hint_model_id": hint_model_id,
                "solver_model_id": solver_model_id,
                "row_idx": row_idx,
                "question": question,
                "ground_truth_answer_full": gt_full,
                "ground_truth_answer": gt_extracted,
                "cheatsheet": cheatsheet,
                "solver_output": solver_out,
                "pred_extracted": info["pred_raw"],
                "correct": correct,
                "debug_numeric": info,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            stats["n"] += 1
            if info["pred_raw"] is None:
                stats["parse_fail"] += 1
            if "\\boxed" not in (solver_out or ""):
                stats["no_boxed"] += 1
            if correct:
                stats["correct"] += 1

    free_model(hint_model, hint_tok)
    free_model(sol_model, sol_tok)

    stats["acc"] = (stats["correct"] / stats["n"]) if stats["n"] else 0.0
    return stats


def main():
    parser = argparse.ArgumentParser()

    # GSM8K dataset defaults
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")  # gsm8k has config "main"
    parser.add_argument("--split", type=str, default="test")

    # optional subsampling
    parser.add_argument("--max_examples", type=int, default=0, help="If >0, evaluate only first N examples.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")

    # optional: override pipelines via JSON
    parser.add_argument(
        "--pipelines_json",
        type=str,
        default="",
        help="Optional path to JSON list of pipelines: [{name,hint_model_id,solver_model_id}, ...]",
    )

    # generation overrides
    parser.add_argument("--dtype", type=str, default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--device_map", type=str, default=os.getenv("DEVICE_MAP", "auto"))
    parser.add_argument("--max_context_tokens", type=int, default=int(os.getenv("MAX_CONTEXT_TOKENS", "8192")))
    parser.add_argument("--max_new_tokens_hint", type=int, default=int(os.getenv("MAX_NEW_TOKENS_HINT", "256")))
    parser.add_argument("--max_new_tokens_sol", type=int, default=int(os.getenv("MAX_NEW_TOKENS_SOL", "512")))
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.7")))
    parser.add_argument("--top_p", type=float, default=float(os.getenv("TOP_P", "0.9")))

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    records_path = os.path.join(args.out_dir, "records.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.json")

    cfg = GenConfig(
        dtype=args.dtype,
        device_map=args.device_map,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens_hint=args.max_new_tokens_hint,
        max_new_tokens_sol=args.max_new_tokens_sol,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Load dataset
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)

    # Select indices
    n = len(ds)
    indices = list(range(n))
    if args.max_examples and args.max_examples > 0:
        indices = indices[: args.max_examples]
    # optional shuffle
    if args.seed and args.max_examples and args.max_examples > 0:
        rnd = random.Random(args.seed)
        rnd.shuffle(indices)

    # Pipelines
    if args.pipelines_json:
        with open(args.pipelines_json, "r", encoding="utf-8") as f:
            pipelines = json.load(f)
    else:
        pipelines = DEFAULT_PIPELINES

    all_results = {
        "dataset": {"name": args.dataset_name, "config": args.dataset_config, "split": args.split},
        "selected": {"max_examples": args.max_examples, "n_total": len(indices), "seed": args.seed},
        "gen_cfg": cfg.__dict__,
        "pipelines": pipelines,
        "aggregate": {},
    }

    for pipe in pipelines:
        stats_ = run_pipeline_on_indices(
            dataset=ds,
            indices=indices,
            pipeline=pipe,
            cfg=cfg,
            out_jsonl_path=records_path,
            resume=args.resume,
        )
        all_results["aggregate"][pipe["name"]] = {"overall": stats_}

        # Save summary after each pipeline
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Wrote per-example logs: {records_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
