"""Stage 17: Filter QAs to keep only the "good" ones for downstream eval.

A QA is kept iff:
  - is_underspecified == False, AND
  - closedbook_gemini-2.5-pro.is_correct == False
    (i.e. the closed-book model got it WRONG, so the question is not trivially
    answerable from the LLM's prior knowledge)

Reads the post-processed file from stage 16 and writes a filtered copy.
Cluster records with zero remaining QAs are dropped entirely.

Inputs:  {output_dir}/{version}/qa/final/all_qas.jsonl
Outputs: {output_dir}/{version}/qa/final/good_qas.jsonl

Usage:
    python 17_filter_good_qas.py --config config.yaml
"""

import argparse

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


CLOSEDBOOK_FIELD = "closedbook_gemini-2.5-pro"


def is_good_qa(qa):
    if qa.get("is_underspecified") is not False:
        return False
    closedbook = qa.get(CLOSEDBOOK_FIELD) or {}
    if closedbook.get("is_correct") is not False:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Filter QAs to drop underspecified or zero-shot-correct ones")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, default=None,
                        help="Override path to all_qas.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="Override path to good_qas.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    in_path = args.input or str(output_dir / "qa" / "final" / "all_qas.jsonl")
    out_path = args.output or str(output_dir / "qa" / "final" / "good_qas.jsonl")

    print(f"Loading {in_path}")
    records = load_jsonl(in_path)
    n_clusters_in = len(records)
    n_qas_in = sum(len(r.get("qas", [])) for r in records)
    print(f"  {n_clusters_in} clusters, {n_qas_in} QAs")

    n_underspecified = n_zeroshot_correct = n_both = n_kept = 0
    out_records = []
    for rec in records:
        kept_qas = []
        for qa in rec.get("qas", []):
            us = qa.get("is_underspecified")
            cb_correct = (qa.get(CLOSEDBOOK_FIELD) or {}).get("is_correct")
            if us is True and cb_correct is True:
                n_both += 1
            elif us is True:
                n_underspecified += 1
            elif cb_correct is True:
                n_zeroshot_correct += 1
            if is_good_qa(qa):
                kept_qas.append(qa)
                n_kept += 1
        if kept_qas:
            new_rec = dict(rec)
            new_rec["qas"] = kept_qas
            out_records.append(new_rec)

    save_jsonl(out_records, out_path)
    print(f"\nFilter results:")
    print(f"  underspecified only:        {n_underspecified}")
    print(f"  closedbook correct only:    {n_zeroshot_correct}")
    print(f"  both (counted once above):  {n_both}")
    print(f"  kept (good):                {n_kept}")
    print(f"\nSaved {len(out_records)} cluster records ({n_kept} QAs) to {out_path}")


if __name__ == "__main__":
    main()
