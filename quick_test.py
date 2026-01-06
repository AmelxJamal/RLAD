# # # import json

# # # def is_refusal(text: str) -> bool:
# # #     t = (text or "").lower()
# # #     return any(p in t for p in [
# # #         "not enough information",
# # #         "insufficient information",
# # #         "cannot be determined",
# # #         "can't determine",
# # #         "missing information",
# # #         "not specified",
# # #         "doesn't specify",
# # #     ])

# # # n = ok = 0
# # # for line in open("/mnt/lts4/scratch/home/abdelrah/RLAD/runs/gsmplus/records.jsonl", "r", encoding="utf-8"):
# # #     r = json.loads(line)
# # #     if (r.get("perturbation_type") or "").strip().lower() != "critical thinking":
# # #         continue
# # #     n += 1
# # #     ok += int(is_refusal(r.get("solver_output","")))

# # # print("n:", n, "refusal_acc:", ok/n if n else 0.0)


# # import json
# # from collections import defaultdict

# # REFUSAL_PHRASES = [
# #     "not enough information",
# #     "insufficient information",
# #     "cannot be determined",
# #     "can't be determined",
# #     "cannot determine",
# #     "can't determine",
# #     "does not provide enough information",
# #     "missing information",
# #     "not specified",
# #     "doesn't specify",
# #     "does not specify",
# #     "cannot be uniquely determined",
# # ]

# # def is_refusal(text: str) -> bool:
# #     t = (text or "").lower()
# #     return any(p in t for p in REFUSAL_PHRASES)

# # path = "/mnt/lts4/scratch/home/abdelrah/RLAD/runs/gsmplus/records.jsonl"

# # counts = defaultdict(lambda: {"n": 0, "ok": 0})
# # for line in open(path, "r", encoding="utf-8"):
# #     r = json.loads(line)
# #     if (r.get("perturbation_type") or "").strip().lower() != "critical thinking":
# #         continue
# #     pipe = r.get("pipeline", "UNKNOWN")
# #     counts[pipe]["n"] += 1
# #     counts[pipe]["ok"] += int(is_refusal(r.get("solver_output", "")))

# # for pipe, c in sorted(counts.items()):
# #     acc = c["ok"] / c["n"] if c["n"] else 0.0
# #     print(pipe, "n=", c["n"], "refusal_acc=", acc)



# import json

# REFUSAL_PHRASES = [
#     "not enough information",
#     "insufficient information",
#     "cannot be determined",
#     "can't be determined",
#     "cannot determine",
#     "can't determine",
#     "does not provide enough information",
#     "missing information",
#     "not specified",
#     "doesn't specify",
#     "does not specify",
#     "cannot be uniquely determined",
# ]

# def is_refusal(text: str) -> bool:
#     t = (text or "").lower()
#     return any(p in t for p in REFUSAL_PHRASES)

# path = "/mnt/lts4/scratch/home/abdelrah/RLAD/runs/gsmplus/records.jsonl"
# BASE = "baseline"
# RLAD = "rlad_main"

# # Collect per-example outcomes
# by_row = {}  # row_idx -> {pipeline: bool}
# for line in open(path, "r", encoding="utf-8"):
#     r = json.loads(line)
#     if (r.get("perturbation_type") or "").strip().lower() != "critical thinking":
#         continue
#     row = r["row_idx"]
#     pipe = r.get("pipeline")
#     ok = is_refusal(r.get("solver_output",""))
#     by_row.setdefault(row, {})[pipe] = ok

# both = rlad_only = base_only = neither = missing = 0

# for row, d in by_row.items():
#     if BASE not in d or RLAD not in d:
#         missing += 1
#         continue
#     b = d[BASE]
#     a = d[RLAD]
#     if b and a:
#         both += 1
#     elif (not b) and a:
#         rlad_only += 1
#     elif b and (not a):
#         base_only += 1
#     else:
#         neither += 1

# n_pairs = both + rlad_only + base_only + neither
# print("paired_n:", n_pairs, "missing_pairs:", missing)
# print("both_correct:", both)
# print("rlad_wins:", rlad_only)
# print("baseline_wins:", base_only)
# print("both_wrong:", neither)
# print("net_gain_rlad:", (rlad_only - base_only) / n_pairs if n_pairs else 0.0)


#!/usr/bin/env python3
import json
import argparse
import statistics as stats
from collections import defaultdict

REFUSAL_PHRASES = [
    "not enough information",
    "insufficient information",
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "can't determine",
    "does not provide enough information",
    "missing information",
    "not specified",
    "doesn't specify",
    "does not specify",
    "cannot be uniquely determined",
    "underspecified",
    "under specified",
    "not uniquely determined",
]

def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in REFUSAL_PHRASES)

def summarize(nums):
    if not nums:
        return {"n": 0}
    nums_sorted = sorted(nums)
    def pct(p):
        # simple percentile by index
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=str, required=True, help="Path to records.jsonl")
    ap.add_argument("--perturbation", type=str, default="critical thinking")
    ap.add_argument("--baseline", type=str, default="baseline")
    ap.add_argument("--rlad", type=str, default="rlad_main")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--tokenizer", type=str, default="", help="Optional HF model id for token counts (loads tokenizer only)")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

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

    # bucket rows
    buckets = {
        "both_correct": [],
        "rlad_wins": [],
        "baseline_wins": [],
        "both_wrong": [],
        "missing_pair": [],
    }

    # store lengths for stats
    lens = {b: {"base_chars": [], "rlad_chars": [], "delta_chars": [],
               "base_words": [], "rlad_words": [], "delta_words": [],
               "base_toks": [], "rlad_toks": [], "delta_toks": []}
            for b in buckets}

    def get_lengths(rec):
        cs = rec.get("cheatsheet") or ""
        ch = len(cs)
        wd = len(cs.split())
        tk = None
        if tok is not None:
            tk = len(tok.encode(cs, add_special_tokens=False))
        return ch, wd, tk

    paired_n = 0
    for row, d in by_row.items():
        if args.baseline not in d or args.rlad not in d:
            buckets["missing_pair"].append(row)
            continue

        paired_n += 1
        rb = d[args.baseline]
        rr = d[args.rlad]

        b_ok = is_refusal(rb.get("solver_output", ""))
        r_ok = is_refusal(rr.get("solver_output", ""))

        if b_ok and r_ok:
            bucket = "both_correct"
        elif (not b_ok) and r_ok:
            bucket = "rlad_wins"
        elif b_ok and (not r_ok):
            bucket = "baseline_wins"
        else:
            bucket = "both_wrong"

        buckets[bucket].append(row)

        bch, bwd, btk = get_lengths(rb)
        rch, rwd, rtk = get_lengths(rr)

        lens[bucket]["base_chars"].append(bch)
        lens[bucket]["rlad_chars"].append(rch)
        lens[bucket]["delta_chars"].append(rch - bch)

        lens[bucket]["base_words"].append(bwd)
        lens[bucket]["rlad_words"].append(rwd)
        lens[bucket]["delta_words"].append(rwd - bwd)

        if tok is not None:
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
        if tok is not None:
            print("- tokens")
            print("  base:", summarize(lens[name]["base_toks"]))
            print("  rlad:", summarize(lens[name]["rlad_toks"]))
            print("  delt:", summarize(lens[name]["delta_toks"]))

    # Print stats for the two “interesting” buckets
    print_bucket_stats("baseline_wins")
    print_bucket_stats("rlad_wins")

    # Show examples where baseline wins and RLAD cheatsheet is much longer
    bw_rows = buckets["baseline_wins"]
    if bw_rows:
        def row_score(row):
            rb = by_row[row][args.baseline]
            rr = by_row[row][args.rlad]
            bch = len((rb.get("cheatsheet") or ""))
            rch = len((rr.get("cheatsheet") or ""))
            return (rch - bch, rch)  # prioritize biggest delta
        bw_rows_sorted = sorted(bw_rows, key=row_score, reverse=True)

        print("\n" + "="*80)
        print(f"Top {min(args.topk, len(bw_rows_sorted))} baseline_wins with largest RLAD cheatsheet delta (chars):")
        for row in bw_rows_sorted[: args.topk]:
            rb = by_row[row][args.baseline]
            rr = by_row[row][args.rlad]
            q = (rb.get("question") or "")[:140].replace("\n", " ")
            bch = len((rb.get("cheatsheet") or ""))
            rch = len((rr.get("cheatsheet") or ""))
            b_ref = is_refusal(rb.get("solver_output", ""))
            r_ref = is_refusal(rr.get("solver_output", ""))
            print(f"row={row}  delta={rch-bch:+5d}  base_cs={bch:5d}  rlad_cs={rch:5d}  base_ref={b_ref}  rlad_ref={r_ref}  q='{q}…'")

if __name__ == "__main__":
    main()
