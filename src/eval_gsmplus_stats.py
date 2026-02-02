from datasets import load_dataset
from argparse import ArgumentParser
import random
import json

GSMPLUS_PERTURBATIONS = [
    "problem understanding",
    "numerical substitution",
    "distraction insertion",
    "digit expansion",
    "critical thinking",
    "adding operation",
    "integer-decimal-fraction conversion",
    "reversing operation",
]
def main(args):
    dataset = load_dataset(args.dataset_name, split=args.split)
    print("Dataset stats:")
    print(f"Number of rows: {len(dataset)}")
    print(f"Number of columns: {len(dataset[0].keys())}")
    print(f"Column names: {dataset[0].keys()}")
    print(dataset[0]['perturbation_type'])
    print("*"*100)
    with open("gsmplus_stats.json", "w", encoding="utf-8") as out_f:
        out_f.write("{\"GSMPLUS_PERTURBATIONS\": [{\n")
        # Group indices by perturbation type
        for pt in GSMPLUS_PERTURBATIONS:
            idxs = [i for i, ex in enumerate(dataset) if ex['perturbation_type'] == pt]
            print(f"Number of rows for {pt}: {len(idxs)}")
            out_f.write(json.dumps({pt: len(idxs)}, indent=2, ensure_ascii=False) + "},\n")
            print("*"*100)
        out_f.write("]}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="qintongli/GSM-Plus")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    main(args)
