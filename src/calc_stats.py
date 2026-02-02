import os
import numpy as np
import json
import argparse
from eval_gsm8k import GenConfig


DEFAULT_PIPELINES = [
    {"name":"rlad_gen_qwen3-1.7b_sol","hint_model_id":"CMU-AIRe/RLAD-Hint-Gen","solver_model_id":"Qwen/Qwen3-1.7B"},
    {"name":"rlad_gen_qwen3-4b_sol","hint_model_id":"CMU-AIRe/RLAD-Hint-Gen","solver_model_id":"Qwen/Qwen3-4B"},
    {"name":"rlad_gen_rlad_sol","hint_model_id":"CMU-AIRe/RLAD-Hint-Gen","solver_model_id":"CMU-AIRe/RLAD-Sol-Gen"},
    
    {"name":"no_gen_qwen3-1.7b_sol","hint_model_id":"","solver_model_id":"Qwen/Qwen3-1.7B"},
    {"name":"no_gen_qwen3-4b_sol","hint_model_id":"","solver_model_id":"Qwen/Qwen3-4B"},
    {"name":"no_gen_rlad_sol","hint_model_id":"","solver_model_id":"CMU-AIRe/RLAD-Sol-Gen"}
  ]

def calculate_stats(records_dir: str= "./../runs/gsm8k/outputs.jsonl", pipelines: list = DEFAULT_PIPELINES, out_dir: str = "./../runs/gsm8k/summary.json")-> dict:
    """
    Calculate different statistics, e.g., accuracy, truncated outputs, etc. for the output file in the given directory (dir).
    Args:
        records_dir (str): Path to the directory containing the output .json file.
        pipelines (list): List of pipeline configurations to evaluate. (json format including name, hint_model_id, solver_model_id)
        out_dir (str): Path to save the summary statistics.
    Returns:
        dict: A dictionary containing the calculated statistics.
    """
    
    stats = {
        "n": 0,
        "correct": 0,
        "truncated": 0,
        "parse_fail": 0,
        "no_boxed": 0,
        "accuracy": 0.0
    }
    all_stats = {pipe["name"]: stats.copy() for pipe in pipelines}
    all_rows = set()
    rows_seen_by_pipe = {p["name"]: set() for p in pipelines}
   
    with open(records_dir, "r") as f:
        for line in f:
            record = json.loads(line)
            row_idx = record.get("row_idx")
            if row_idx is None:
                continue
            all_rows.add(row_idx)

            hint_id = record.get("hint_model_id", None)
            solver_id = record.get("solver_model_id", None)

            for pipe in pipelines:
                if hint_id == pipe["hint_model_id"] and solver_id == pipe["solver_model_id"]:
                    name = pipe["name"]
                    all_stats[name]["n"] += 1
                    rows_seen_by_pipe[name].add(row_idx)

                    pred_raw = (record.get("debug_numeric") or {}).get("pred_raw")
                    if pred_raw is None:
                        all_stats[name]["parse_fail"] += 1

                    solver_output = record.get("solver_output", "")
                    if "\\boxed" not in solver_output:
                        all_stats[name]["no_boxed"] += 1

                    if record.get("correct") is True:
                        all_stats[name]["correct"] += 1
                    break 
            
    # Calculate accuracy
    total_examples = len(all_rows)
    for name, stat in all_stats.items():
        stat["accuracy"] = stat["correct"] / stat["n"] if stat["n"] > 0 else 0.0
        stat["truncated"] = total_examples - len(rows_seen_by_pipe[name])

    return all_stats


def main():
    parser = argparse.ArgumentParser()

    # GSM8K dataset defaults
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")  # gsm8k has config "main"
    parser.add_argument("--split", type=str, default="test")

    # optional subsampling
    parser.add_argument("--max_examples", type=int, default= 200, help="Number of examples to eval; 0=all")
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

    args = parser.parse_args()
    
    summary_path = os.path.join(args.out_dir, "summary.json")

    # Recreate the config so we can save it in the summary file - it should match what was used during generation
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


    # Pipelines
    if args.pipelines_json:
        with open(args.pipelines_json, "r", encoding="utf-8") as f:
            pipelines = json.load(f)
    else:
        pipelines = DEFAULT_PIPELINES

    all_results = {
        "dataset": {"name": args.dataset_name, "config": args.dataset_config, "split": args.split},
        "selected": {"max_examples": args.max_examples, "n_total": args.max_examples, "seed": args.seed},
        "gen_cfg": cfg.__dict__,
        "pipelines": pipelines,
        "aggregate": {},
    }

    all_stats = calculate_stats(
        records_dir=os.path.join(args.out_dir, "outputs.jsonl"),
        pipelines=pipelines,
        out_dir=summary_path,
    )
    print("Calculated statistics:", all_stats)
    for pipe in [p["name"] for p in pipelines]:
        all_results["aggregate"][pipe] = {"overall": all_stats[pipe]}

        # Save summary after each pipeline
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Wrote summary: {summary_path}")

if __name__ == "__main__":
    main()