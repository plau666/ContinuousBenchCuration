"""Stage 18: Split the cleaned corpus into three eval-ready slices.

Produces three JSONL files in {output_dir}/{version}/corpus/:

  large.jsonl   — every cleaned article (full corpus, byte-for-byte copy of
                  cleaned/articles.jsonl)
  median.jsonl  — the union of all articles that landed in any cluster (from
                  clustered/clustered_articles.json), deduped by article_idx
                  and sorted ascending by article_idx
  small.jsonl   — only the articles that support at least one good QA, i.e.
                  the union of every `supports` list across
                  qa/final/filtered/good_qas.jsonl, again deduped and sorted

Each slice has the same per-article schema as cleaned/articles.jsonl
(carrying `article_idx`).

Inputs:  {output_dir}/{version}/cleaned/articles.jsonl
         {output_dir}/{version}/clustered/clustered_articles.json
         {output_dir}/{version}/qa/final/filtered/good_qas.jsonl
Outputs: {output_dir}/{version}/corpus/{large,median,small}.jsonl

Usage:
    python 18_split_corpus.py --config config.yaml
"""

import argparse
import json
import shutil
from pathlib import Path

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


def main():
    parser = argparse.ArgumentParser(description="Split cleaned corpus into large/median/small")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "cleaned" / "articles.jsonl"
    clustered_path = output_dir / "clustered" / "clustered_articles.json"
    good_qas_path = output_dir / "qa" / "final" / "filtered" / "good_qas.jsonl"

    # ── 1. large.jsonl: byte-copy the entire cleaned corpus (idempotent) ──
    large_path = corpus_dir / "large.jsonl"
    if large_path.exists() and large_path.stat().st_size > 0:
        n_large = sum(1 for _ in open(large_path))
        print(f"[1/3] large.jsonl already exists ({n_large:,} articles), skipping copy")
    else:
        print(f"[1/3] large.jsonl ← copying {cleaned_path}")
        shutil.copyfile(cleaned_path, large_path)
        n_large = sum(1 for _ in open(large_path))
        size_mb = large_path.stat().st_size / 1e6
        print(f"      {n_large:,} articles, {size_mb:.1f} MB")

    # ── 2. median.jsonl: union of articles in clustered_articles.json (idempotent) ──
    median_path = corpus_dir / "median.jsonl"
    if median_path.exists() and median_path.stat().st_size > 0:
        median_count = sum(1 for _ in open(median_path))
        print(f"\n[2/3] median.jsonl already exists ({median_count:,} articles), skipping flatten")
    else:
        print(f"\n[2/3] median.jsonl ← flattening {clustered_path}")
        with open(clustered_path) as f:
            clustered = json.load(f)
        n_clusters = len(clustered)
        median_by_idx = {}
        n_visited = 0
        for cid, arts in clustered.items():
            for art in arts:
                n_visited += 1
                aidx = art.get("article_idx")
                if aidx is None:
                    continue
                median_by_idx[int(aidx)] = art   # dedup by article_idx
        median = [median_by_idx[k] for k in sorted(median_by_idx.keys())]
        save_jsonl(median, str(median_path))
        median_count = len(median)
        size_mb = median_path.stat().st_size / 1e6
        print(f"      {n_clusters:,} clusters, {n_visited:,} (cluster, article) pairs visited")
        print(f"      {median_count:,} unique articles after dedup, {size_mb:.1f} MB")

    # ── 3. small.jsonl: union of `supports` across good_qas.jsonl (idempotent) ──
    small_path = corpus_dir / "small.jsonl"
    if small_path.exists() and small_path.stat().st_size > 0:
        small_count = sum(1 for _ in open(small_path))
        print(f"\n[3/3] small.jsonl already exists ({small_count:,} articles), skipping rebuild")
    else:
        print(f"\n[3/3] small.jsonl ← collecting `supports` from {good_qas_path}")
        good_qas = load_jsonl(str(good_qas_path))
        needed_idxs = set()
        n_qa_with_support = 0
        for qa in good_qas:
            sup = qa.get("supports") or []
            if sup:
                n_qa_with_support += 1
            for aidx in sup:
                needed_idxs.add(int(aidx))
        print(f"      {len(good_qas):,} good QAs, {n_qa_with_support:,} with at least one support")
        print(f"      {len(needed_idxs):,} unique supporting article_idx values")

        # Stream cleaned/articles.jsonl and pull only the needed lines.
        # Look up by explicit `article_idx` field (set in stage 3); fall back to line position.
        found = {}
        with open(cleaned_path) as f:
            for line_pos, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                art = json.loads(line)
                aidx = int(art.get("article_idx", line_pos))
                if aidx in needed_idxs:
                    found[aidx] = art
                if len(found) == len(needed_idxs):
                    break

        if len(found) != len(needed_idxs):
            missing = needed_idxs - set(found)
            print(f"      WARNING: {len(missing)} supporting article_idx not found in cleaned/articles.jsonl")
            print(f"      first few missing: {sorted(missing)[:10]}")

        small = [found[k] for k in sorted(found.keys())]
        save_jsonl(small, str(small_path))
        small_count = len(small)
        size_mb = small_path.stat().st_size / 1e6
        print(f"      {small_count:,} articles, {size_mb:.1f} MB")

    # ── 4. Build large/medium/small subfolders + train/val/test (90/5/5) ──
    # Inside each subfolder the source file is symlinked as `all.jsonl` so the
    # 4 files are uniformly named regardless of which slice you're looking at:
    #   {large,medium,small}/{all,train,val,test}.jsonl
    print("\n[4/4] Train/val/test split (90/5/5)")
    from tools.split import split_train_val_test
    seed = config.get("seed", 42)
    slice_specs = [
        ("large",  large_path),
        ("medium", median_path),
        ("small",  small_path),
    ]
    for folder_name, source in slice_specs:
        if not source.exists():
            print(f"  [{folder_name}] skipping — source not found ({source})")
            continue
        info = split_train_val_test(
            source, corpus_dir / folder_name, source_basename="all.jsonl", seed=seed
        )
        print(f"  [{folder_name}] {source.name} → all.jsonl: n={info['n_total']:,} → "
              f"train={info['n_train']:,}, val={info['n_val']:,}, test={info['n_test']:,}")

    # ── Summary ──
    print(f"\nDone. Outputs in {corpus_dir}/")
    print(f"  large.jsonl  {n_large:>10,} articles")
    print(f"  median.jsonl {median_count:>10,} articles")
    print(f"  small.jsonl  {small_count:>10,} articles")


if __name__ == "__main__":
    main()
