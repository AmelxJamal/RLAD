import os
import re
import json
import gc
import time
import math
import random
from dataclasses import dataclass
from collections import defaultdict
from fractions import Fraction
from typing import Dict, Any, Optional, List, Tuple
import argparse


import subprocess
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback


# -------------------- Defaults --------------------
GSMPLUS_PERTURBATIONS = [
    "problem understanding",
    "numerical substitution",
    "distraction insertion",
    "digit expansion",
    # "critical thinking",
    # "adding operation",
    "integer-decimal-fraction conversion",
    # "reversing operation",
]

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


def _prepare_batch_inputs(
    tokenizer,
    systems: List[str],
    users: List[str],
    max_context_tokens: int,
    max_new_tokens: int,
    device,
):
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    )
    input_id_seqs = []
    lengths = []
    keep = max(1, max_context_tokens - max_new_tokens)
    for sys_str, usr_str in zip(systems, users):
        ids = chat_encode(tokenizer, system=sys_str, user=usr_str)
        ids = truncate_left(ids, keep_tokens=keep)
        ids = ids.squeeze(0)
        lengths.append(ids.shape[-1])
        input_id_seqs.append(ids)
    padded = pad_sequence(input_id_seqs, batch_first=True, padding_value=pad_id)
    attention_mask = (padded != pad_id).long()
    return padded.to(device), attention_mask.to(device), lengths


@torch.inference_mode()
def generate_batch_texts(
    model,
    tokenizer,
    systems: List[str],
    users: List[str],
    max_context_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    assert len(systems) == len(users)
    if len(systems) == 0:
        return []

    input_ids, attention_mask, lengths = _prepare_batch_inputs(
        tokenizer, systems, users, max_context_tokens, max_new_tokens, model.device
    )

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    results = []
    for i in range(len(systems)):
        start = lengths[i]
        gen_tokens = outputs[i, start:]
        ended_with_eos = False
        if tokenizer.eos_token_id is not None and gen_tokens.numel() > 0:
            ended_with_eos = gen_tokens[-1].item() == tokenizer.eos_token_id
        hit_max_new_tokens = gen_tokens.shape[-1] >= max_new_tokens and not ended_with_eos
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        results.append((decoded, ended_with_eos, hit_max_new_tokens))
    return results


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
    res = generate_batch_texts(
        model=model,
        tokenizer=tokenizer,
        systems=[system],
        users=[user],
        max_context_tokens=max_context_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    return res[0]


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
    t = s.strip()
    if not t:
        return None
    # common cleanup
    t = t.replace(",", "")
    # keep only the first token if there are trailing words
    # (still lets "150." or "150\n" work)
    t = t.strip()

    # Some models output like "#### 150"
    if t.startswith("####"):
        t = t[4:].strip()

    # If it's something like "150)" or "150." trim non-numeric tail
    m = _NUM_RE.search(t)
    if not m:
        return None
    token = m.group(1).replace(",", "")
    try:
        return Fraction(token)  # handles ints, decimals, simple fractions
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


def is_correct(pred_text: str, gt_text: str) -> Tuple[bool, Dict[str, Any]]:
    pred_raw = extract_pred_answer(pred_text)
    gt_raw = gt_text.strip() if gt_text is not None else ""

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
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
    max_new_tokens_hint: int = int(os.getenv("MAX_NEW_TOKENS_HINT", "4096"))
    max_new_tokens_sol: int = int(os.getenv("MAX_NEW_TOKENS_SOL", "4096"))
    do_sample: bool = bool(int(os.getenv("DO_SAMPLE", "0")))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))
    

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


def group_indices_by_perturbation(dataset) -> Dict[str, List[int]]:
    groups = defaultdict(list)
    for i, ex in enumerate(dataset):
        pt = (ex.get("perturbation_type") or "").strip()
        groups[pt].append(i)
    return dict(groups)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sample_gpu_stats() -> Dict[str, Any]:
    """
    Lightweight GPU snapshot using nvidia-smi. Safe if nvidia-smi is missing.
    """
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
        # Only log first visible GPU (respecting CUDA_VISIBLE_DEVICES mapping)
        parts = [p.strip() for p in lines[0].split(",")]
        # Expected 8 fields
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


# -------------------- Main evaluation --------------------
def run_pipeline_on_indices(
    *,
    dataset,
    indices: List[int],
    pipeline: Dict[str, str],
    cfg: GenConfig,
    out_jsonl_path: str,
    resume: bool,
    seed: int,
):
    rnd = random.Random(seed)

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

    # Load models ONCE per pipeline
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
    by_pt = defaultdict(lambda: {"n": 0, "correct": 0, "parse_fail": 0, "no_boxed": 0})

    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for row_idx in tqdm(indices, desc=f"{pipeline['name']}"):
            if (pipeline["name"], row_idx) in done:
                continue

            ex = dataset[row_idx]
            question = (ex.get("question") or ex.get("problem") or "").strip()
            gt = (ex.get("answer") or "").strip()
            pt = (ex.get("perturbation_type") or "").strip()
            seed_q = (ex.get("seed_question") or "").strip()

            # hint
            hs, hu = build_hint_prompt(question)
            try:
                cheatsheet, hint_eos, hint_hit_max = generate_text(
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
                cheatsheet, hint_eos, hint_hit_max = "", False, False

            # solve
            ss, su = build_solver_prompt(question, cheatsheet)
            try:
                solver_out, solver_eos, solver_hit_max = generate_text(
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
                solver_out, solver_eos, solver_hit_max = "", False, False

            correct, info = is_correct(solver_out, gt)

            rec = {
                "ts": time.time(),
                "pipeline": pipeline["name"],
                "hint_model_id": hint_model_id,
                "solver_model_id": solver_model_id,
                "row_idx": row_idx,
                "perturbation_type": pt,
                "seed_question": seed_q,
                "question": question,
                "ground_truth_answer": gt,
                "cheatsheet": cheatsheet,
                "cheatsheet_ended_with_eos": hint_eos,
                "cheatsheet_hit_max_new_tokens": hint_hit_max,
                "solver_output": solver_out,
                "solver_ended_with_eos": solver_eos,
                "solver_hit_max_new_tokens": solver_hit_max,
                "pred_extracted": info["pred_raw"],
                "correct": correct,
                "debug_numeric": info,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            stats["n"] += 1
            by_pt[pt]["n"] += 1

            if info["pred_raw"] is None:
                stats["parse_fail"] += 1
                by_pt[pt]["parse_fail"] += 1
            if "\\boxed" not in (solver_out or ""):
                stats["no_boxed"] += 1
                by_pt[pt]["no_boxed"] += 1
            if correct:
                stats["correct"] += 1
                by_pt[pt]["correct"] += 1

    free_model(hint_model, hint_tok)
    free_model(sol_model, sol_tok)

    # compute accuracies
    def finalize(d):
        n = d["n"]
        d["acc"] = (d["correct"] / n) if n else 0.0
        return d

    stats = finalize(stats)
    by_pt = {k: finalize(v) for k, v in by_pt.items()}
    return stats, by_pt


def sanitize_model_id_for_path(model_id: str) -> str:
    if not model_id:
        return "no_hint"
    return model_id.replace("/", "__").replace(":", "_")


def load_hint_cache(cache_path: str) -> Dict[int, Dict[str, Any]]:
    cache: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                cache[int(obj["row_idx"])] = {
                    "cheatsheet": obj.get("cheatsheet", ""),
                    "ended_with_eos": bool(obj.get("ended_with_eos", False)),
                    "hit_max_new_tokens": bool(obj.get("hit_max_new_tokens", False)),
                }
            except Exception:
                continue
    return cache


def generate_and_cache_hints(
    *,
    dataset,
    indices: List[int],
    hint_model_id: str,
    cfg: GenConfig,
    cache_path: str,
    resume: bool,
    log_gpu_stats: bool,
    batch_size: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Generate cheatsheets for the given indices using the hint model and cache them to disk.
    Returns a dict: row_idx -> {"cheatsheet": str, "ended_with_eos": bool, "hit_max_new_tokens": bool}
    """
    if not hint_model_id or str(hint_model_id).strip() == "":
        return {idx: {"cheatsheet": "", "ended_with_eos": False, "hit_max_new_tokens": False} for idx in indices}

    if not resume and os.path.exists(cache_path):
        os.remove(cache_path)

    cache = load_hint_cache(cache_path) if resume else {}
    hint_model, hint_tok = load_model_and_tokenizer(hint_model_id, cfg)

    with open(cache_path, "a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(indices), batch_size), desc=f"hints:{hint_model_id}"):
            batch_ids = [idx for idx in indices[start:start + batch_size] if idx not in cache]
            if not batch_ids:
                continue

            systems = []
            users = []
            for row_idx in batch_ids:
                ex = dataset[row_idx]
                question = (ex.get("question") or (ex.get("problem") or "")).strip()
                hs, hu = build_hint_prompt(question)
                systems.append(hs)
                users.append(hu)

            gpu_before = sample_gpu_stats() if log_gpu_stats else None
            try:
                gens = generate_batch_texts(
                    hint_model,
                    hint_tok,
                    systems,
                    users,
                    max_context_tokens=cfg.max_context_tokens,
                    max_new_tokens=cfg.max_new_tokens_hint,
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gens = [("", False, False) for _ in batch_ids]
            gpu_after = sample_gpu_stats() if log_gpu_stats else None

            for row_idx, (cheatsheet, ended_eos, hit_max) in zip(batch_ids, gens):
                rec = {
                    "row_idx": row_idx,
                    "cheatsheet": cheatsheet,
                    "ended_with_eos": ended_eos,
                    "hit_max_new_tokens": hit_max,
                    "gpu_before": gpu_before,
                    "gpu_after": gpu_after,
                }
                cache[row_idx] = {
                    "cheatsheet": cheatsheet,
                    "ended_with_eos": ended_eos,
                    "hit_max_new_tokens": hit_max,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    free_model(hint_model, hint_tok)
    return cache


def run_pipelines_with_cached_hints(
    *,
    dataset,
    indices: List[int],
    pipelines: List[Dict[str, str]],
    hint_cache: Dict[str, Dict[int, Dict[str, Any]]],
    cfg: GenConfig,
    out_jsonl_path: str,
    resume: bool,
    log_gpu_stats: bool,
    batch_size: int,
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

    # Load unique solvers once
    solvers: Dict[str, Tuple[Any, Any]] = {}
    for solver_id in sorted({p["solver_model_id"] for p in pipelines}):
        sol_model, sol_tok = load_model_and_tokenizer(solver_id, cfg)
        solvers[solver_id] = (sol_model, sol_tok)

    stats: Dict[str, Dict[str, Any]] = {}
    by_pt: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for pipe in pipelines:
        stats[pipe["name"]] = {
            "n": 0,
            "correct": 0,
            "parse_fail": 0,
            "no_boxed": 0,
            "ended_with_eos": 0,
            "hit_max_new_tokens": 0,
        }
        by_pt[pipe["name"]] = defaultdict(lambda: {
            "n": 0,
            "correct": 0,
            "parse_fail": 0,
            "no_boxed": 0,
            "ended_with_eos": 0,
            "hit_max_new_tokens": 0,
        })

    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for pipe in pipelines:
            solver_id = pipe["solver_model_id"]
            sol_model, sol_tok = solvers[solver_id]

            pending_indices = [idx for idx in indices if (pipe["name"], idx) not in done]
            for start in tqdm(range(0, len(pending_indices), batch_size), desc=f"pipe:{pipe['name']}"):
                batch_ids = pending_indices[start:start + batch_size]
                if not batch_ids:
                    continue

                systems = []
                users = []
                questions = []
                gts = []
                pts = []
                seed_qs = []
                cheatsheets = []
                cheat_meta = []

                for row_idx in batch_ids:
                    ex = dataset[row_idx]
                    question = (ex.get("question") or (ex.get("problem") or "")).strip()
                    gt = (ex.get("answer") or "").strip()
                    pt = (ex.get("perturbation_type") or "").strip()
                    seed_q = (ex.get("seed_question") or "").strip()
                    cheatsheet_info = hint_cache.get(pipe["hint_model_id"], {}).get(
                        row_idx,
                        {"cheatsheet": "", "ended_with_eos": False, "hit_max_new_tokens": False},
                    )
                    cheatsheet = cheatsheet_info["cheatsheet"]

                    ss, su = build_solver_prompt(question, cheatsheet)
                    systems.append(ss)
                    users.append(su)
                    questions.append(question)
                    gts.append(gt)
                    pts.append(pt)
                    seed_qs.append(seed_q)
                    cheatsheets.append(cheatsheet)
                    cheat_meta.append(cheatsheet_info)

                gpu_before_solver = sample_gpu_stats() if log_gpu_stats else None
                try:
                    gen_batch = generate_batch_texts(
                        sol_model,
                        sol_tok,
                        systems,
                        users,
                        max_context_tokens=cfg.max_context_tokens,
                        max_new_tokens=cfg.max_new_tokens_sol,
                        do_sample=cfg.do_sample,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                    )
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gen_batch = [("", False, False) for _ in batch_ids]
                gpu_after_solver = sample_gpu_stats() if log_gpu_stats else None

                for row_idx, question, gt, pt, seed_q, cheatsheet, cheat_info, gen_out in zip(
                    batch_ids, questions, gts, pts, seed_qs, cheatsheets, cheat_meta, gen_batch
                ):
                    solver_out, solver_eos, solver_hit_max = gen_out
                    correct, info = is_correct(solver_out, gt)

                    rec = {
                        "ts": time.time(),
                        "pipeline": pipe["name"],
                        "hint_model_id": pipe["hint_model_id"],
                        "solver_model_id": solver_id,
                        "row_idx": row_idx,
                        "perturbation_type": pt,
                        "seed_question": seed_q,
                        "question": question,
                        "ground_truth_answer": gt,
                        "cheatsheet": cheatsheet,
                        "cheatsheet_ended_with_eos": cheat_info.get("ended_with_eos", False),
                        "cheatsheet_hit_max_new_tokens": cheat_info.get("hit_max_new_tokens", False),
                        "solver_output": solver_out,
                        "solver_ended_with_eos": solver_eos,
                        "solver_hit_max_new_tokens": solver_hit_max,
                        "gpu_before_solver": gpu_before_solver,
                        "gpu_after_solver": gpu_after_solver,
                        "pred_extracted": info["pred_raw"],
                        "correct": correct,
                        "debug_numeric": info,
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    stats_pipe = stats[pipe["name"]]
                    stats_pipe["n"] += 1
                    if solver_eos:
                        stats_pipe["ended_with_eos"] += 1
                    if solver_hit_max:
                        stats_pipe["hit_max_new_tokens"] += 1
                    if info["pred_raw"] is None:
                        stats_pipe["parse_fail"] += 1
                    if "\\boxed" not in (solver_out or ""):
                        stats_pipe["no_boxed"] += 1
                    if correct:
                        stats_pipe["correct"] += 1

                    stats_pt = by_pt[pipe["name"]][pt]
                    stats_pt["n"] += 1
                    if solver_eos:
                        stats_pt["ended_with_eos"] += 1
                    if solver_hit_max:
                        stats_pt["hit_max_new_tokens"] += 1
                    if info["pred_raw"] is None:
                        stats_pt["parse_fail"] += 1
                    if "\\boxed" not in (solver_out or ""):
                        stats_pt["no_boxed"] += 1
                    if correct:
                        stats_pt["correct"] += 1

                out_f.flush()

    # finalize
    def finalize(d):
        n = d["n"]
        d["acc"] = (d["correct"] / n) if n else 0.0
        return d

    stats = {k: finalize(v) for k, v in stats.items()}
    by_pt = {pipe: {pt: finalize(v) for pt, v in pts.items()} for pipe, pts in by_pt.items()}

    for mdl, tok in solvers.values():
        free_model(mdl, tok)
    return stats, by_pt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="qintongli/GSM-Plus")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument(
        "--perturbation_types",
        type=str,
        default="all",
        help="Comma-separated list or 'all'.",
    )
    parser.add_argument(
        "--max_per_type",
        type=int,
        default=0,
        help="If >0, subsample this many examples per perturbation type.",
    )
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
    parser.add_argument("--max_new_tokens_hint", type=int, default=int(os.getenv("MAX_NEW_TOKENS_HINT", "4096")))
    parser.add_argument("--max_new_tokens_sol", type=int, default=int(os.getenv("MAX_NEW_TOKENS_SOL", "4096")))
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.7")))
    parser.add_argument("--top_p", type=float, default=float(os.getenv("TOP_P", "0.9")))
    parser.add_argument("--log_gpu_stats", action="store_true", help="Log nvidia-smi snapshots before/after generation.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for hint/solver generation.")

    args = parser.parse_args()

    set_global_seeds(args.seed)
    ensure_dir(args.out_dir)
    records_path = os.path.join(args.out_dir, "records.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.json")
    indices_path = os.path.join(args.out_dir, "selected_indices.json")

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
    ds = load_dataset(args.dataset_name, split=args.split)

    # Choose perturbations
    if args.perturbation_types.strip().lower() == "all":
        wanted_pts = set(GSMPLUS_PERTURBATIONS)
    else:
        wanted_pts = set([p.strip() for p in args.perturbation_types.split(",") if p.strip()])

    # Group indices by perturbation type
    groups = group_indices_by_perturbation(ds)
    selected_indices = []
    rnd = random.Random(args.seed)
    for pt in GSMPLUS_PERTURBATIONS:
        if pt not in wanted_pts:
            continue
        idxs = groups.get(pt, [])
        if args.max_per_type and args.max_per_type > 0 and len(idxs) > args.max_per_type:
            idxs = idxs[:]
            rnd.shuffle(idxs)
            idxs = idxs[: args.max_per_type]
        selected_indices.extend(idxs)

    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "indices": selected_indices,
                "seed": args.seed,
                "perturbation_types": sorted(list(wanted_pts)),
                "max_per_type": args.max_per_type,
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
    # Generate and cache hints once per hint model (non-empty)
    hint_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
    unique_hint_ids = {p["hint_model_id"] for p in pipelines if p.get("hint_model_id")}
    for hint_id in unique_hint_ids:
        cache_path = os.path.join(
            args.out_dir,
            f"hints_{sanitize_model_id_for_path(hint_id)}.jsonl",
        )
        hint_cache[hint_id] = generate_and_cache_hints(
            dataset=ds,
            indices=selected_indices,
            hint_model_id=hint_id,
            cfg=cfg,
            cache_path=cache_path,
            resume=args.resume,
            log_gpu_stats=args.log_gpu_stats,
            batch_size=args.batch_size,
        )

    # Run pipelines using cached hints
    stats, by_pt = run_pipelines_with_cached_hints(
        dataset=ds,
        indices=selected_indices,
        pipelines=pipelines,
        hint_cache=hint_cache,
        cfg=cfg,
        out_jsonl_path=records_path,
        resume=args.resume,
        log_gpu_stats=args.log_gpu_stats,
        batch_size=args.batch_size,
    )

    all_results = {
        "dataset": {"name": args.dataset_name, "split": args.split},
        "selected": {
            "perturbation_types": sorted(list(wanted_pts)),
            "max_per_type": args.max_per_type,
            "n_total": len(selected_indices),
        },
        "gen_cfg": cfg.__dict__,
        "pipelines": pipelines,
        "aggregate": {},
        "indices_path": indices_path,
        "records_path": records_path,
        "hint_caches": {
            hid: os.path.join(
                args.out_dir, f"hints_{sanitize_model_id_for_path(hid)}.jsonl"
            )
            for hid in unique_hint_ids
        },
    }

    for pipe in pipelines:
        all_results["aggregate"][pipe["name"]] = {
            "overall": stats[pipe["name"]],
            "by_perturbation_type": by_pt.get(pipe["name"], {}),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Wrote per-example logs: {records_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
