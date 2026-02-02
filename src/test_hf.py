#!/usr/bin/env python3
import os
import re
import json
import gc
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Config --------------------
# Baseline (non-RLAD) weights (override via env vars if you want)
BASE_HINT_ID = os.getenv("BASE_HINT_ID", "Qwen/Qwen3-4B")
BASE_SOL_ID  = os.getenv("BASE_SOL_ID",  "Qwen/Qwen3-1.7B")

# RLAD weights
RLAD_HINT_ID = os.getenv("RLAD_HINT_ID", "CMU-AIRe/RLAD-Hint-Gen")
RLAD_SOL_ID  = os.getenv("RLAD_SOL_ID",  "CMU-AIRe/RLAD-Sol-Gen")

# Dataset
DATASET_ID   = os.getenv("DATASET_ID", "CMU-AIRe/RLAD-abstractions-benchmarks")
SPLIT        = os.getenv("SPLIT", "AIME2025")
ROW_INDEX    = int(os.getenv("ROW_INDEX", "0"))

# Token budgets
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))
MAX_NEW_TOKENS     = int(os.getenv("MAX_NEW_TOKENS", "512"))

# Runtime
DTYPE = os.getenv("DTYPE", "bfloat16")      # bfloat16 on H100/H200
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
OUT_JSON = os.getenv("OUT_JSON", f"/mnt/lts4/scratch/home/abdelrah/RLAD/rlad_sanity_{SPLIT}_row{ROW_INDEX}.json")


# -------------------- Helpers --------------------
def torch_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown DTYPE={dtype_str}")

def read_model_card_text(repo_id: str, n_lines: int = 80):
    """
    Returns (status, snippet_text). status is 'ok'/'missing'/'error'
    """
    try:
        path = hf_hub_download(repo_id, "README.md")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        snippet = "\n".join(lines[:n_lines]) if lines else ""
        return "ok", snippet
    except EntryNotFoundError:
        return "missing", ""
    except Exception as e:
        return "error", f"Failed to read README.md: {e}"

def load_model_and_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype(DTYPE),
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )
    mdl.eval()
    return mdl, tok

def free_model(mdl, tok):
    del mdl
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def chat_encode(tokenizer, system: str, user: str):
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    raw = f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"
    return tokenizer(raw, return_tensors="pt")["input_ids"]

def truncate_left(input_ids: torch.Tensor, keep_tokens: int):
    if input_ids.shape[-1] <= keep_tokens:
        return input_ids
    return input_ids[:, -keep_tokens:]

@torch.inference_mode()
def generate_text(model, tokenizer, system: str, user: str, max_new_tokens: int):
    input_ids = chat_encode(tokenizer, system=system, user=user)
    keep = max(1, MAX_CONTEXT_TOKENS - max_new_tokens)
    input_ids = truncate_left(input_ids, keep_tokens=keep).to(model.device)

    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic for debugging
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def extract_boxed(text: str):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1).strip() if m else None

def build_hint_prompt(problem: str):
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

def build_solver_prompt(problem: str, cheatsheet: str):
    system = (
        "You are an expert problem-solving assistant.\n"
        "Use the provided cheatsheet (if any) to solve the problem.\n"
        "Always give the final answer in the exact form \\boxed{<answer>} with a single integer.\n"
    )
    user = f"CHEATSHEET:\n{cheatsheet}\n\nPROBLEM:\n{problem}\n"
    return system, user


def run_pipeline(hint_model_id: str, sol_model_id: str, problem: str):
    # hint
    hint_model, hint_tok = load_model_and_tokenizer(hint_model_id)
    hs, hu = build_hint_prompt(problem)
    cheatsheet = generate_text(hint_model, hint_tok, hs, hu, max_new_tokens=MAX_NEW_TOKENS)
    free_model(hint_model, hint_tok)

    # solve
    sol_model, sol_tok = load_model_and_tokenizer(sol_model_id)
    ss, su = build_solver_prompt(problem, cheatsheet)
    solution = generate_text(sol_model, sol_tok, ss, su, max_new_tokens=MAX_NEW_TOKENS)
    free_model(sol_model, sol_tok)

    return {
        "hint_model_id": hint_model_id,
        "solver_model_id": sol_model_id,
        "hint_prompt": {"system": hs, "user": hu},
        "solver_prompt": {"system": ss, "user": su},
        "cheatsheet": cheatsheet,
        "solver_output": solution,
        "boxed_answer": extract_boxed(solution),
    }


def main():
    # Load dataset row
    ds = load_dataset(DATASET_ID, split=SPLIT)
    ex = ds[ROW_INDEX]
    problem = (ex.get("problem") or "").strip()
    gt = (ex.get("answer") or "").strip()

    # Read model cards (READMEs) for RLAD repos (and baseline if you want)
    rlad_hint_card_status, rlad_hint_card = read_model_card_text(RLAD_HINT_ID)
    rlad_sol_card_status, rlad_sol_card = read_model_card_text(RLAD_SOL_ID)
    base_hint_card_status, base_hint_card = read_model_card_text(BASE_HINT_ID)
    base_sol_card_status, base_sol_card = read_model_card_text(BASE_SOL_ID)

    results = {
        "dataset": {"id": DATASET_ID, "split": SPLIT, "row_index": ROW_INDEX},
        "example": {"problem": problem, "ground_truth_answer": gt},
        "settings": {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": DTYPE,
            "device_map": DEVICE_MAP,
        },
        "model_cards": {
            BASE_HINT_ID: {"status": base_hint_card_status, "readme_snippet": base_hint_card},
            BASE_SOL_ID:  {"status": base_sol_card_status,  "readme_snippet": base_sol_card},
            RLAD_HINT_ID: {"status": rlad_hint_card_status, "readme_snippet": rlad_hint_card},
            RLAD_SOL_ID:  {"status": rlad_sol_card_status,  "readme_snippet": rlad_sol_card},
        },
        "runs": {},
    }

    # Baseline run
    results["runs"]["baseline_non_rlad"] = run_pipeline(BASE_HINT_ID, BASE_SOL_ID, problem)

    # RLAD run
    results["runs"]["rlad"] = run_pipeline(RLAD_HINT_ID, RLAD_SOL_ID, problem)

    # Save JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nWrote: {OUT_JSON}")
    print("Baseline boxed answer:", results["runs"]["baseline_non_rlad"]["boxed_answer"])
    print("RLAD boxed answer:", results["runs"]["rlad"]["boxed_answer"])
    print("GT answer:", gt)


if __name__ == "__main__":
    main()
