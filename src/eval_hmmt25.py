import os
import re
import json
import gc
import time
import random
import subprocess
from dataclasses import dataclass
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback


# -------------------- Defaults --------------------
DEFAULT_DATASET_NAME = os.getenv("DATASET_NAME", "MathArena/hmmt_feb_2025")
DEFAULT_SPLIT = os.getenv("SPLIT", "train")

DEFAULT_N_LIST = os.getenv("N_LIST", "1,4")  # RLAD-style: try 1 and 4
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "1.0"))

DEFAULT_PIPELINES = [
    {
        "name": "zeroshot_solver",
        "hint_model_id": "",
        "solver_model_id": os.getenv("ZEROSHOT_SOL_ID", "Qwen/Qwen3-1.7B"),
    },
    {
        "name": "rlad_solver_no_hint",
        "hint_model_id": "",
        "solver_model_id": os.getenv("RLAD_SOL_ID", "CMU-AIRe/RLAD-Sol-Gen"),
    },
    {
        "name": "rlad_hint_zeroshot_solver",
        "hint_model_id": os.getenv("RLAD_HINT_ID", "CMU-AIRe/RLAD-Hint-Gen"),
        "solver_model_id": os.getenv("ZEROSHOT_SOL_ID", "Qwen/Qwen3-1.7B"),
    },
    {
        "name": "rlad_hint_rlad_solver",
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
def generate_text_k(
    model,
    tokenizer,
    system: str,
    user: str,
    max_context_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
) -> List[Tuple[str, bool, bool]]:
    """
    Generate k outputs for a single prompt.
    Returns: List[(decoded, ended_with_eos, hit_max_new_tokens)]
    """
    input_ids = chat_encode(tokenizer, system=system, user=user)
    input_ids = truncate_left(input_ids, keep_tokens=max(1, max_context_tokens - max_new_tokens))
    input_ids = input_ids.to(model.device)
    
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_return_sequences=num_return_sequences,
    )
    
    if not do_sample and num_return_sequences > 1:
        gen_kwargs["num_beams"] = num_return_sequences
    
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
    
    outputs = model.generate(input_ids=input_ids, **gen_kwargs)
    
    # Extract generated tokens (skip prompt)
    start_idx = input_ids.shape[-1]
    results = []
    for out_i in range(num_return_sequences):
        gen_tokens = outputs[out_i, start_idx:]
        ended_with_eos = tokenizer.eos_token_id is not None and gen_tokens.numel() > 0 and gen_tokens[-1].item() == tokenizer.eos_token_id
        hit_max_new_tokens = gen_tokens.shape[-1] >= max_new_tokens and not ended_with_eos
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        results.append((decoded, ended_with_eos, hit_max_new_tokens))
    
    return results


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


_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*(?:/\d[\d,]*\.?\d*)?)")


def _to_fraction(s: str) -> Optional[Fraction]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    t = t.replace(",", "").strip()
    if t.startswith("####"):
        t = t[4:].strip()
    m = _NUM_RE.search(t)
    if not m:
        return None
    token = m.group(1).replace(",", "")
    try:
        return Fraction(token)
    except Exception:
        return None


def extract_pred_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = _BOX_RE.search(text)
    if m:
        return m.group(1).strip()
    ms = list(_NUM_RE.finditer(text))
    if ms:
        return ms[-1].group(1).strip()
    return None


def is_correct(pred_text: str, gt_value: Any) -> Tuple[bool, Dict[str, Any]]:
    pred_raw = extract_pred_answer(pred_text)
    gt_raw = "" if gt_value is None else str(gt_value).strip()

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


def majority_vote_answer(texts: List[str]) -> Tuple[Optional[str], Dict[str, int]]:
    preds = [extract_pred_answer(t) for t in texts]
    preds = [p for p in preds if p is not None and str(p).strip() != ""]
    if not preds:
        return None, {}
    c = Counter(preds)
    top = c.most_common(1)[0][0]
    return top, dict(c)


@dataclass
class GenConfig:
    dtype: str = os.getenv("DTYPE", "bfloat16")
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
    max_new_tokens_hint: int = int(os.getenv("MAX_NEW_TOKENS_HINT", "4096"))
    max_new_tokens_sol: int = int(os.getenv("MAX_NEW_TOKENS_SOL", "4096"))

    do_sample: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P


def load_model_and_tokenizer(model_id: str, cfg: GenConfig):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype(cfg.dtype),
        device_map=cfg.device_map,
        trust_remote_code=True,
    )
    mdl.eval()
    return mdl, tok


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_model_id_for_path(model_id: str) -> str:
    if not model_id:
        return "no_hint"
    return model_id.replace("/", "__").replace(":", "_")


def sample_gpu_stats() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    query = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(query, encoding="utf-8")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return {"available": True, "error": "nvidia-smi-empty-output"}
        parts = [p.strip() for p in lines[0].split(",")]
        keys = [
            "timestamp",
            "index",
            "clocks_sm_mhz",
            "clocks_mem_mhz",
            "util_gpu_pct",
            "util_mem_pct",
            "power_w",
            "temp_c",
        ]
        data = {k: parts[i] if i < len(parts) else None for i, k in enumerate(keys)}
        return {"available": True, **data}
    except FileNotFoundError:
        return {"available": True, "error": "nvidia-smi-not-found"}
    except Exception as e:
        return {"available": True, "error": str(e)}


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -------------------- Hint caching --------------------
def load_hint_cache(cache_path: str) -> Dict[int, Dict[str, Any]]:
    """
    row_idx -> {"abstractions":[...], "abstractions_meta":[[eos,hit],...]}
    """
    cache: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                row_idx = int(obj["row_idx"])
                cache[row_idx] = {
                    "abstractions": list(obj.get("abstractions", [])),
                    "abstractions_meta": list(obj.get("abstractions_meta", [])),
                }
            except Exception:
                continue
    return cache


def generate_and_cache_hints(
    *,
    ds,
    indices: List[int],
    hint_model_id: str,
    cfg: GenConfig,
    cache_path: str,
    resume: bool,
    log_gpu_stats: bool,
    max_n: int,
) -> Dict[int, Dict[str, Any]]:
    if not hint_model_id or str(hint_model_id).strip() == "":
        # No hint model: trivial cache
        return {
            idx: {
                "abstractions": ["" for _ in range(max_n)],
                "abstractions_meta": [[False, False] for _ in range(max_n)],
            }
            for idx in indices
        }

    if not resume and os.path.exists(cache_path):
        os.remove(cache_path)

    cache = load_hint_cache(cache_path) if resume else {}
    hint_model, hint_tok = load_model_and_tokenizer(hint_model_id, cfg)

    with open(cache_path, "a", encoding="utf-8") as f:
        for row_idx in tqdm(indices, desc=f"hints:{hint_model_id}"):
            if row_idx in cache:
                continue
            
            ex = ds[row_idx]
            problem = str(ex.get("question") or ex.get("problem") or "").strip()
            sys_prompt, usr_prompt = build_hint_prompt(problem)
            
            gpu_before = sample_gpu_stats() if log_gpu_stats else None
            try:
                abstractions = generate_text_k(
                    hint_model, hint_tok,
                    system=sys_prompt,
                    user=usr_prompt,
                    max_context_tokens=cfg.max_context_tokens,
                    max_new_tokens=cfg.max_new_tokens_hint,
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    num_return_sequences=max_n,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                abstractions = [("", False, False) for _ in range(max_n)]
            gpu_after = sample_gpu_stats() if log_gpu_stats else None
            
            abstractions_list = [t for (t, _, _) in abstractions]
            abstractions_meta = [[bool(eos), bool(hit)] for (_, eos, hit) in abstractions]
            
            rec = {
                "row_idx": row_idx,
                "abstractions": abstractions_list,
                "abstractions_meta": abstractions_meta,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
            }
            cache[row_idx] = {
                "abstractions": abstractions_list,
                "abstractions_meta": abstractions_meta,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

    free_model(hint_model, hint_tok)
    return cache


# -------------------- Pipeline eval using cached hints --------------------
def run_pipelines_with_cached_hints(
    *,
    ds,
    indices: List[int],
    pipelines: List[Dict[str, str]],
    hint_cache_by_model: Dict[str, Dict[int, Dict[str, Any]]],
    cfg: GenConfig,
    out_jsonl_path: str,
    resume: bool,
    log_gpu_stats: bool,
    n_list: List[int],
):
    max_n = max(n_list)

    # Track completed (pipeline, row_idx) pairs
    done = set()
    if resume and os.path.exists(out_jsonl_path):
        with open(out_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add((obj.get("pipeline"), int(obj.get("row_idx"))))
                except Exception:
                    pass

    # Load unique solvers once
    solvers: Dict[str, Tuple[Any, Any]] = {}
    for solver_id in sorted({p["solver_model_id"] for p in pipelines}):
        sol_model, sol_tok = load_model_and_tokenizer(solver_id, cfg)
        solvers[solver_id] = (sol_model, sol_tok)

    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for pipe in pipelines:
            solver_id = pipe["solver_model_id"]
            sol_model, sol_tok = solvers[solver_id]
            hint_id = pipe.get("hint_model_id", "") or ""

            for row_idx in tqdm(indices, desc=f"pipe:{pipe['name']}"):
                if (pipe["name"], row_idx) in done:
                    continue
                
                ex = ds[row_idx]
                problem = str(ex.get("question") or ex.get("problem") or "").strip()
                gt = ex.get("answer")
                problem_type = ex.get("problem_type")
                
                # Get cached hints for this problem
                if hint_id.strip():
                    hint_info = hint_cache_by_model[hint_id].get(row_idx)
                    if not hint_info:
                        hint_info = {
                            "abstractions": ["" for _ in range(max_n)],
                            "abstractions_meta": [[False, False] for _ in range(max_n)],
                        }
                    abstractions = hint_info["abstractions"]
                    abstractions_meta = hint_info["abstractions_meta"]
                else:
                    abstractions = ["" for _ in range(max_n)]
                    abstractions_meta = [[False, False] for _ in range(max_n)]
                
                # Generate solutions for each abstraction
                solutions = []  # [abs_j] -> list of (text, eos, hit) tuples
                solutions_meta = []  # [abs_j] -> list of (eos, hit) bools
                
                gpu_before_sol = sample_gpu_stats() if log_gpu_stats else None
                for abs_idx, abstraction in enumerate(abstractions):
                    sys_prompt, usr_prompt = build_solver_prompt(problem, abstraction)
                    try:
                        sols = generate_text_k(
                            sol_model, sol_tok,
                            system=sys_prompt,
                            user=usr_prompt,
                            max_context_tokens=cfg.max_context_tokens,
                            max_new_tokens=cfg.max_new_tokens_sol,
                            do_sample=cfg.do_sample,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            num_return_sequences=max_n,
                        )
                    except torch.cuda.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        sols = [("", False, False) for _ in range(max_n)]
                    
                    sol_texts = [t for (t, _, _) in sols]
                    sol_flags = [(bool(eos), bool(hit)) for (_, eos, hit) in sols]
                    solutions.append(sol_texts)
                    solutions_meta.append(sol_flags)
                
                gpu_after_sol = sample_gpu_stats() if log_gpu_stats else None
                
                # Compute metrics
                metrics_by_n = {}
                for n in n_list:
                    candidates = []
                    any_ok = False
                    for aj in range(n):
                        for k in range(n):
                            txt = solutions[aj][k]
                            ok, _ = is_correct(txt, gt)
                            any_ok = any_ok or bool(ok)
                            candidates.append(txt)
                    
                    vote_pred, vote_counts = majority_vote_answer(candidates)
                    vote_correct = False
                    if vote_pred is not None:
                        vp = _to_fraction(vote_pred)
                        gg = _to_fraction(str(gt))
                        vote_correct = (vp is not None and gg is not None and vp == gg)
                    
                    metrics_by_n[str(n)] = {
                        "n_abs": n,
                        "n_sol_per_abs": n,
                        "k_total": n * n,
                        "pass_at_k": bool(any_ok),
                        "vote_pred": vote_pred,
                        "vote_correct": bool(vote_correct),
                        "vote_counts": vote_counts,
                    }
                
                # Weighted metrics (best-of-4x1)
                w_abs_avg_4x1 = None
                w_abs_best_4x1 = None
                if max_n >= 4:
                    per_abs_ok = []
                    for aj in range(4):
                        ok, _ = is_correct(solutions[aj][0], gt)
                        per_abs_ok.append(bool(ok))
                    w_abs_avg_4x1 = float(sum(per_abs_ok) / 4.0)
                    w_abs_best_4x1 = bool(any(per_abs_ok))
                
                rec = {
                    "ts": time.time(),
                    "pipeline": pipe["name"],
                    "hint_model_id": hint_id,
                    "solver_model_id": solver_id,
                    "row_idx": row_idx,
                    "question": problem,
                    "problem_type": problem_type,
                    "ground_truth_answer": gt,
                    "gen_cfg": {
                        "do_sample": cfg.do_sample,
                        "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "max_n": max_n,
                    },
                    "abstractions": abstractions,
                    "abstractions_meta": abstractions_meta,
                    "solutions": solutions,
                    "solutions_meta": solutions_meta,
                    "metrics_by_n": metrics_by_n,
                    "w_abs_avg_4x1": w_abs_avg_4x1,
                    "w_abs_best_4x1": w_abs_best_4x1,
                    "gpu_before_solver": gpu_before_sol,
                    "gpu_after_solver": gpu_after_sol,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    for mdl, tok in solvers.values():
        free_model(mdl, tok)


def compute_summary_from_records(
    records_path: str,
    pipeline_names: List[str],
    indices_set: set,
    n_list: List[int],
):
    """
    Returns:
      {
        pipeline_name: {
          "all": { ...overall metrics... },
          "by_problem_type": {
             "<ptype>": { ...same metric structure... },
             ...
          }
        },
        ...
      }

    Notes:
    - `problem_type` may be a list like ["Number Theory"] or ["Algebra","Geometry"].
      A problem is counted in EACH listed type.
    """

    def _empty_bucket():
        return {
            "n_total": 0,
            "by_n": {str(n): {"k_total": n * n, "pass": 0, "vote": 0} for n in n_list},
            "w_abs_avg_4x1_sum": 0.0,  # summed; divided by n_total at finalize
            "w_abs_best_4x1": 0,
        }

    def _finalize_bucket(bucket):
        n_total = bucket["n_total"]
        out_bucket = {
            "n_total": n_total,
            "by_n": {},
            "w_abs_avg_4x1": (bucket["w_abs_avg_4x1_sum"] / n_total) if n_total else 0.0,
            "w_abs_best_4x1": (bucket["w_abs_best_4x1"] / n_total) if n_total else 0.0,
        }
        for n in n_list:
            key = str(n)
            out_bucket["by_n"][key] = {
                "k_total": n * n,
                "pass_at_k": (bucket["by_n"][key]["pass"] / n_total) if n_total else 0.0,
                "vote_acc": (bucket["by_n"][key]["vote"] / n_total) if n_total else 0.0,
            }
        return out_bucket

    def _normalize_problem_types(v) -> List[str]:
        """
        Accepts:
          - ["Number Theory"] or ["Alg","Geo"]
          - "Number Theory"
          - None / "" / [] -> ["unknown"]
        Returns a de-duplicated list of normalized type strings.
        """
        if v is None:
            return ["unknown"]

        # If it's already a list/tuple/set, flatten to strings
        if isinstance(v, (list, tuple, set)):
            types = []
            for x in v:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    types.append(s)
            if not types:
                return ["unknown"]
            # de-dupe while preserving order
            seen = set()
            out = []
            for t in types:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out

        # Otherwise treat as a single string
        s = str(v).strip()
        return [s] if s else ["unknown"]

    # acc[pipeline]["all"] = bucket; acc[pipeline]["by_problem_type"][ptype] = bucket
    acc = {
        p: {
            "all": _empty_bucket(),
            "by_problem_type": defaultdict(_empty_bucket),
        }
        for p in pipeline_names
    }

    if not os.path.exists(records_path):
        return {}

    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue

            p = r.get("pipeline")
            if p not in acc:
                continue

            row_idx = r.get("row_idx")
            if row_idx is None:
                continue
            try:
                row_idx = int(row_idx)
            except Exception:
                continue
            if indices_set and row_idx not in indices_set:
                continue

            ptypes = _normalize_problem_types(r.get("problem_type"))

            mbn = r.get("metrics_by_n", {}) or {}

            # always update "all"
            buckets = [acc[p]["all"]]

            # update each problem type bucket (count in each category)
            for pt in ptypes:
                buckets.append(acc[p]["by_problem_type"][pt])

            for b in buckets:
                b["n_total"] += 1

                for n in n_list:
                    key = str(n)
                    m = mbn.get(key)
                    if not m:
                        continue
                    if m.get("pass_at_k"):
                        b["by_n"][key]["pass"] += 1
                    if m.get("vote_correct"):
                        b["by_n"][key]["vote"] += 1

                if r.get("w_abs_avg_4x1") is not None:
                    b["w_abs_avg_4x1_sum"] += float(r["w_abs_avg_4x1"])
                if r.get("w_abs_best_4x1"):
                    b["w_abs_best_4x1"] += 1

    # Finalize to rates
    out = {}
    for p in pipeline_names:
        out[p] = {
            "all": _finalize_bucket(acc[p]["all"]),
            "by_problem_type": {},
        }
        for pt in sorted(acc[p]["by_problem_type"].keys()):
            out[p]["by_problem_type"][pt] = _finalize_bucket(acc[p]["by_problem_type"][pt])

    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--pipelines_json",
        type=str,
        default="",
        help="Optional path to JSON list of pipelines: [{name,hint_model_id,solver_model_id}, ...]",
    )

    # generation controls
    parser.add_argument("--dtype", type=str, default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--device_map", type=str, default=os.getenv("DEVICE_MAP", "auto"))
    parser.add_argument("--max_context_tokens", type=int, default=int(os.getenv("MAX_CONTEXT_TOKENS", "32768")))
    parser.add_argument("--max_new_tokens_hint", type=int, default=int(os.getenv("MAX_NEW_TOKENS_HINT", "16384")))
    parser.add_argument("--max_new_tokens_sol", type=int, default=int(os.getenv("MAX_NEW_TOKENS_SOL", "8192")))

    parser.add_argument("--do_sample", action="store_true", help="Force sampling on.")
    parser.add_argument("--no_sample", action="store_true", help="Force greedy/beam decoding (sampling off).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)

    parser.add_argument("--n_list", type=str, default=DEFAULT_N_LIST)  # e.g. 1,4
    parser.add_argument("--log_gpu_stats", action="store_true")

    args = parser.parse_args()

    set_global_seeds(args.seed)
    ensure_dir(args.out_dir)

    records_path = os.path.join(args.out_dir, "records.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.json")
    indices_path = os.path.join(args.out_dir, "selected_indices.json")

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    if not n_list:
        raise ValueError("Empty --n_list")
    max_n = max(n_list)

    do_sample = True
    if args.no_sample:
        do_sample = False
    if args.do_sample:
        do_sample = True

    cfg = GenConfig(
        dtype=args.dtype,
        device_map=args.device_map,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens_hint=args.max_new_tokens_hint,
        max_new_tokens_sol=args.max_new_tokens_sol,
        do_sample=do_sample,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    # Load dataset: HMMT25 train split by default (only split available)
    ds = load_dataset(args.dataset_name, split=args.split)

    indices = list(range(len(ds)))
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(indices)
    if args.max_examples and args.max_examples > 0:
        indices = indices[: args.max_examples]
    indices_set = set(indices)

    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": {"name": args.dataset_name, "split": args.split},
                "problem_types": list(set(ex.get("problem_type")[0] for ex in ds)),
                "indices": indices,
                "n_total": len(indices),
                "seed": args.seed,
                "shuffle": bool(args.shuffle),
                "max_examples": args.max_examples,
                "n_list": n_list,
                "max_n": max_n,
            },
            f,
            indent=2,
        )
    
    # Pipelines
    if args.pipelines_json:
        with open(args.pipelines_json, "r", encoding="utf-8") as f:
            pipelines = json.load(f)
    else:
        pipelines = DEFAULT_PIPELINES

    # Cache hints once per unique hint model id
    hint_cache_by_model: Dict[str, Dict[int, Dict[str, Any]]] = {}
    unique_hint_ids = {p.get("hint_model_id", "") for p in pipelines if p.get("hint_model_id", "").strip()}
    for hint_id in sorted(unique_hint_ids):
        cache_path = os.path.join(
            args.out_dir,
            f"hints_{sanitize_model_id_for_path(hint_id)}_K{max_n}.jsonl",
        )
        hint_cache_by_model[hint_id] = generate_and_cache_hints(
            ds=ds,
            indices=indices,
            hint_model_id=hint_id,
            cfg=cfg,
            cache_path=cache_path,
            resume=args.resume,
            log_gpu_stats=args.log_gpu_stats,
            max_n=max_n,
        )

    # Run pipelines using cached hints
    run_pipelines_with_cached_hints(
        ds=ds,
        indices=indices,
        pipelines=pipelines,
        hint_cache_by_model=hint_cache_by_model,
        cfg=cfg,
        out_jsonl_path=records_path,
        resume=args.resume,
        log_gpu_stats=args.log_gpu_stats,
        n_list=n_list,
    )

    # Summarize from records (so resume is consistent)
    pipeline_names = [p["name"] for p in pipelines]
    results = compute_summary_from_records(records_path, pipeline_names, indices_set, n_list)

    out = {
        "dataset": {"name": args.dataset_name, "split": args.split},
        "selected_indices_path": indices_path,
        "records_path": records_path,
        "hint_caches": {
            hid: os.path.join(args.out_dir, f"hints_{sanitize_model_id_for_path(hid)}_K{max_n}.jsonl")
            for hid in sorted(unique_hint_ids)
        },
        "gen_cfg": cfg.__dict__,
        "n_list": n_list,
        "pipelines": pipelines,
        "results": results,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote per-example logs: {records_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()