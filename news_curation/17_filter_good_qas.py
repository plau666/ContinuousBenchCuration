"""Stage 17: Filter QAs to "good" ones, flatten, and split into val/test.

Reads the post-processed cluster records from stage 16 and writes THREE files
under qa/final/filtered/:

  good_qas.jsonl   — every "good" QA as a flat list of dicts (NOT grouped by
                     cluster). A QA is "good" iff:
                       - is_underspecified == False, AND
                       - closedbook_gemini-2.5-pro.is_correct == False
                     Each dict has only the eval-relevant fields:
                       question, answer, supports,
                       closedbook_gemini-2.5-pro,
                       openbook_gemini-2.5-flash-lite

  val.jsonl        — half of each cluster's good QAs (per-cluster stratified
                     split with a seeded shuffle), then flattened across clusters
  test.jsonl       — the other half

If a cluster has an odd number of good QAs, val gets floor(n/2) and test gets
ceil(n/2). The shuffle uses random.Random(seed) where seed defaults to
config["seed"]; pass --split-seed to override.

Inputs:  {output_dir}/{version}/qa/final/all_qas.jsonl
Outputs: {output_dir}/{version}/qa/final/filtered/{good_qas,val,test}.jsonl

Usage:
    python 17_filter_good_qas.py --config config.yaml
    python 17_filter_good_qas.py --config config.yaml --split-seed 123
"""

import argparse
import random
from collections import defaultdict

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


CLOSEDBOOK_FIELD = "closedbook_gemini-2.5-pro"
OPENBOOK_FIELD = "openbook_gemini-2.5-flash-lite"


def is_good_qa(qa):
    """A QA is good iff it is fully specified AND the closed-book model got it wrong."""
    if qa.get("is_underspecified") is not False:
        return False
    closedbook = qa.get(CLOSEDBOOK_FIELD) or {}
    if closedbook.get("is_correct") is not False:
        return False
    return True


def slim_qa(qa):
    """Project a QA dict down to the eval-relevant fields only."""
    return {
        "question": qa.get("question"),
        "answer": qa.get("answer"),
        "supports": qa.get("supports") or [],
        CLOSEDBOOK_FIELD: qa.get(CLOSEDBOOK_FIELD) or {},
        OPENBOOK_FIELD: qa.get(OPENBOOK_FIELD) or [],
    }


def main():
    parser = argparse.ArgumentParser(description="Filter, flatten, and split QAs")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Seed for the per-cluster val/test shuffle (default: config.seed)")
    parser.add_argument("--input", type=str, default=None,
                        help="Override path to all_qas.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    seed = args.split_seed if args.split_seed is not None else config["seed"]

    in_path = args.input or str(output_dir / "qa" / "final" / "all_qas.jsonl")
    filtered_dir = output_dir / "qa" / "final" / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path}")
    records = load_jsonl(in_path)
    n_clusters_in = len(records)
    n_qas_in = sum(len(r.get("qas", [])) for r in records)
    print(f"  {n_clusters_in} clusters, {n_qas_in} QAs")

    # ── 1. Filter to good QAs, group by cluster (still in cluster order) ──
    n_underspecified_only = n_closedbook_correct_only = n_both_bad = 0
    good_by_cluster = defaultdict(list)
    cluster_order = []
    for rec in records:
        cid = str(rec.get("cluster_id"))
        if cid not in good_by_cluster:
            cluster_order.append(cid)
        for qa in rec.get("qas", []):
            us = qa.get("is_underspecified")
            cb_correct = (qa.get(CLOSEDBOOK_FIELD) or {}).get("is_correct")
            if us is True and cb_correct is True:
                n_both_bad += 1
            elif us is True:
                n_underspecified_only += 1
            elif cb_correct is True:
                n_closedbook_correct_only += 1
            if is_good_qa(qa):
                good_by_cluster[cid].append(slim_qa(qa))

    n_good = sum(len(v) for v in good_by_cluster.values())
    n_clusters_with_good = sum(1 for v in good_by_cluster.values() if v)
    print(f"\nFilter results:")
    print(f"  underspecified only:        {n_underspecified_only}")
    print(f"  closedbook correct only:    {n_closedbook_correct_only}")
    print(f"  both (counted once above):  {n_both_bad}")
    print(f"  kept (good):                {n_good}  in {n_clusters_with_good} clusters")

    # ── 2. Save flat good_qas.jsonl (no cluster grouping) ──
    flat_good = []
    for cid in cluster_order:
        flat_good.extend(good_by_cluster[cid])
    good_path = filtered_dir / "good_qas.jsonl"
    save_jsonl(flat_good, str(good_path))
    print(f"\nSaved {len(flat_good)} QAs to {good_path}")

    # ── 3. Per-cluster seeded shuffle, split into val/test halves ──
    rng = random.Random(seed)
    val, test = [], []
    for cid in sorted(good_by_cluster.keys()):  # sort for determinism
        group = list(good_by_cluster[cid])
        if not group:
            continue
        rng.shuffle(group)
        half = len(group) // 2
        val.extend(group[:half])
        test.extend(group[half:])

    val_path = filtered_dir / "val.jsonl"
    test_path = filtered_dir / "test.jsonl"
    save_jsonl(val, str(val_path))
    save_jsonl(test, str(test_path))
    print(f"\nSeeded split (seed={seed}):")
    print(f"  val.jsonl:  {len(val)} QAs  -> {val_path}")
    print(f"  test.jsonl: {len(test)} QAs  -> {test_path}")
    print(f"  (clusters with odd count assign the extra QA to test)")


if __name__ == "__main__":
    main()
