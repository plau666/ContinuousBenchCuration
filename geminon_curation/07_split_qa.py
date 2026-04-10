"""Stage 7: Stratified val/test split of QA pairs.

For each geminon, splits its 14 QAs into two equal halves (7 val + 7 test)
using a seeded shuffle. Then concatenates across geminons to produce val.jsonl
and test.jsonl. Two output folders are created — `small` (filtered against
sampled_200k.jsonl) and `medium` (filtered against sampled_1m.jsonl) — where
the `supports` field of each QA is filtered to only include article_idxs that
appear in the corresponding sampled corpus. Folder names match the corpus
slice naming used in stage 5 (corpus/{small,medium,large}/).

Both folders share the same val/test partition (same seed); only the
supports filtering differs.

Usage:
    python 07_split_qa.py --config config.yaml
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


def load_article_idx_set(corpus_path):
    """Load a sampled corpus JSONL and return the set of article_idx values."""
    aidxs = set()
    with open(corpus_path) as f:
        import json
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aidx = rec.get("article_idx")
            if aidx is not None:
                aidxs.add(aidx)
    return aidxs


def split_qas_per_geminon(qas, seed):
    """Group by geminon_idx, shuffle each group with a seeded RNG, split in half.

    Returns (val_qas, test_qas) — flat lists across all geminons.
    """
    by_gidx = defaultdict(list)
    for qa in qas:
        by_gidx[qa["geminon_idx"]].append(qa)

    rng = random.Random(seed)
    val, test = [], []

    # Iterate in sorted geminon idx order for determinism
    for gidx in sorted(by_gidx.keys()):
        group = list(by_gidx[gidx])
        rng.shuffle(group)
        half = len(group) // 2
        val.extend(group[:half])
        test.extend(group[half:])

    return val, test


def filter_supports(qas, allowed_aidxs):
    """Return new QA records with `supports` filtered to allowed_aidxs."""
    out = []
    for qa in qas:
        new_qa = dict(qa)
        new_qa["supports"] = [a for a in qa["supports"] if a in allowed_aidxs]
        out.append(new_qa)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stratified val/test split of QAs")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Seed for the val/test split (default: config seed)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    qa_dir = output_dir / "qa"
    corpus_dir = output_dir / "corpus"

    seed = args.split_seed if args.split_seed is not None else config["seed"]

    # Load public + sensitive QAs separately
    public_qas = load_jsonl(str(qa_dir / "public_qas.jsonl"))
    sensitive_qas = load_jsonl(str(qa_dir / "sensitive_qas.jsonl"))
    print(f"Loaded {len(public_qas)} public + {len(sensitive_qas)} sensitive QAs")

    # Split each set independently per geminon (same partition for both 200k and 1m folders)
    public_val, public_test = split_qas_per_geminon(public_qas, seed)
    sensitive_val, sensitive_test = split_qas_per_geminon(sensitive_qas, seed)
    print(f"Split with seed {seed}:")
    print(f"  public:    {len(public_val)} val + {len(public_test)} test")
    print(f"  sensitive: {len(sensitive_val)} val + {len(sensitive_test)} test")

    # Load article_idx sets for the two sampled corpora
    print("\nLoading sampled corpus article_idx sets...")
    aidxs_200k = load_article_idx_set(corpus_dir / "sampled_200k.jsonl")
    aidxs_1m = load_article_idx_set(corpus_dir / "sampled_1m.jsonl")
    print(f"  sampled_200k: {len(aidxs_200k)} article_idxs")
    print(f"  sampled_1m:   {len(aidxs_1m)} article_idxs")

    # Build filtered versions for each folder. Folder names match the
    # corpus slice naming convention used in stage 5 (corpus/{small,medium}/):
    #   small  ← supports filtered to sampled_200k
    #   medium ← supports filtered to sampled_1m
    folders = {
        "small":  aidxs_200k,
        "medium": aidxs_1m,
    }

    splits = {
        "public_val": public_val,
        "public_test": public_test,
        "sensitive_val": sensitive_val,
        "sensitive_test": sensitive_test,
    }

    def support_stats(qas):
        sizes = [len(q["supports"]) for q in qas]
        n_zero = sum(1 for s in sizes if s == 0)
        avg = sum(sizes) / len(sizes) if sizes else 0
        return avg, n_zero, max(sizes) if sizes else 0

    for folder_name, allowed in folders.items():
        folder = qa_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\n  {folder_name}/")
        for split_name, split_qas in splits.items():
            filtered = filter_supports(split_qas, allowed)
            save_jsonl(filtered, str(folder / f"{split_name}.jsonl"))
            avg, n_zero, mx = support_stats(filtered)
            print(f"    {split_name}.jsonl  ({len(filtered)} QAs): avg {avg:.1f} supports, {n_zero} with 0, max {mx}")


if __name__ == "__main__":
    main()
