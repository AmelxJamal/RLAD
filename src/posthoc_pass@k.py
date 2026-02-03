#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from fractions import Fraction
from typing import Any, Dict, Optional, List, Tuple
from collections import defaultdict

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

def is_correct(pred_text: str, gt_value: Any) -> bool:
    pred_raw = extract_pred_answer(pred_text)
    gt_raw = "" if gt_value is None else str(gt_value).strip()
    pred_frac = _to_fraction(pred_raw) if pred_raw is not None else None
    gt_frac = _to_fraction(gt_raw)
    if pred_frac is None or gt_frac is None:
        return False
    return pred_frac == gt_frac

def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    # Standard estimator: 1 - C(n-c, k)/C(n, k)
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def flatten_solutions(solutions: List[List[str]]) -> List[str]:
    # solutions is [abs_j][k]
    flat = []
    for abs_list in solutions:
        if not isinstance(abs_list, list):
            continue
        for s in abs_list:
            flat.append("" if s is None else str(s))
    return flat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Path to records.jsonl")
    ap.add_argument("--out", default="", help="Optional output json path")
    ap.add_argument("--problem_type", default="", help="Extract on the lines in the records of this problem type only")
    args = ap.parse_args()

    agg = defaultdict(lambda: {
        "n_examples": 0,
        # boolean-style (empirical) rates
        "pass1_first": 0,
        "pass4_4x1": 0,
        "pass16_any": 0,
        # estimator-style means (order-invariant)
        "pass1_est_from16_sum": 0.0,
        "pass4_est_from16_sum": 0.0,
        "pass16_est_from16_sum": 0.0,
        # sanity stats
        "n_samples_min": None,
        "n_samples_max": None,
    })

    with open(args.records, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            pipe = r.get("pipeline", "UNKNOWN")
            if args.problem_type:
                problem_type = r.get("problem_type", "")
                problem_match = True if any(temp_prob_type == args.problem_type for temp_prob_type in problem_type) else False
            
                if not problem_match:
                    continue
            gt = r.get("ground_truth_answer", None)
            solutions = r.get("solutions", None)
            if not isinstance(solutions, list):
                continue

            # Flatten all candidates (typically 16 when max_n=4)
            flat = flatten_solutions(solutions)
            n = len(flat)
            if n == 0:
                continue

            correct_flags = [is_correct(t, gt) for t in flat]
            c = sum(1 for x in correct_flags if x)

            # "first sample" pass@1 (matches your n=1 metric if ordering is consistent)
            first_ok = bool(correct_flags[0]) if n >= 1 else False

            # 4x1: first solution from each of first 4 abstractions
            ok_4x1 = False
            if len(solutions) >= 4:
                flags_4x1 = []
                for aj in range(4):
                    row = solutions[aj] if isinstance(solutions[aj], list) else []
                    t = row[0] if len(row) >= 1 else ""
                    flags_4x1.append(is_correct(t, gt))
                ok_4x1 = any(flags_4x1)

            # any among all samples (for max_n=4 -> 16)
            any_ok = (c > 0)

            # estimator-style from the n samples available
            p1_est = pass_at_k_estimator(n=n, c=c, k=1)
            p4_est = pass_at_k_estimator(n=n, c=c, k=4)
            p16_est = pass_at_k_estimator(n=n, c=c, k=16)

            d = agg[pipe]
            d["n_examples"] += 1
            d["pass1_first"] += int(first_ok)
            d["pass4_4x1"] += int(ok_4x1)
            d["pass16_any"] += int(any_ok)
            d["pass1_est_from16_sum"] += p1_est
            d["pass4_est_from16_sum"] += p4_est
            d["pass16_est_from16_sum"] += p16_est

            d["n_samples_min"] = n if d["n_samples_min"] is None else min(d["n_samples_min"], n)
            d["n_samples_max"] = n if d["n_samples_max"] is None else max(d["n_samples_max"], n)

    # finalize rates
    out = {}
    for pipe, d in agg.items():
        m = d["n_examples"]
        out[pipe] = {
            "n_examples": m,
            "n_samples_min": d["n_samples_min"],
            "n_samples_max": d["n_samples_max"],

            # empirical, order-dependent variants
            "pass@1_first": d["pass1_first"] / m if m else 0.0,
            "pass@4_4x1": d["pass4_4x1"] / m if m else 0.0,
            "pass@16_any": d["pass16_any"] / m if m else 0.0,

            # estimator, order-invariant variants (computed from whatever n samples exist)
            "pass@1_est_from_n": d["pass1_est_from16_sum"] / m if m else 0.0,
            "pass@4_est_from_n": d["pass4_est_from16_sum"] / m if m else 0.0,
            "pass@16_est_from_n": d["pass16_est_from16_sum"] / m if m else 0.0,
        }

    # print nicely
    print(json.dumps(out, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as wf:
            json.dump(out, wf, indent=2)
        print(f"\nWrote: {args.out}")

if __name__ == "__main__":
    main()