"""Stage 18: Split the cleaned corpus into three eval-ready slices.

Produces three JSONL files in {output_dir}/{version}/corpus/:

  large.jsonl   — every cleaned article (full corpus, byte-for-byte copy of
                  cleaned/articles.jsonl)
  small.jsonl   — only the articles that support at least one good QA, i.e.
                  the union of every `supports` list across
                  qa/final/filtered/good_qas.jsonl, deduped and sorted by
                  article_idx
  medium.jsonl  — the union of (a) every article that landed in any cluster
                  (from clustered/clustered_articles.json) AND (b) every
                  article in small.jsonl. The second leg guarantees the
                  invariant `small ⊆ medium ⊆ large` even though the
                  support-retrieval in stage 14/15 runs over the full
                  cleaned corpus.

Each slice has the same per-article schema as cleaned/articles.jsonl
(carrying `article_idx`).

Inputs:  {output_dir}/{version}/cleaned/articles.jsonl
         {output_dir}/{version}/clustered/clustered_articles.json
         {output_dir}/{version}/qa/final/filtered/good_qas.jsonl
Outputs: {output_dir}/{version}/corpus/{large,medium,small}.jsonl
         {output_dir}/{version}/corpus/{large,medium,small}/{train,val,test}.jsonl

Usage:
    python 18_split_corpus.py --config config.yaml
"""

import argparse
import json

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


def _read_idxs(path):
    return {json.loads(line)["article_idx"] for line in open(path)}


def main():
    parser = argparse.ArgumentParser(description="Split cleaned corpus into large/medium/small")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "cleaned" / "articles.jsonl"
    clustered_path = output_dir / "clustered" / "clustered_articles.json"
    good_qas_path = output_dir / "qa" / "final" / "filtered" / "good_qas.jsonl"

    large_path  = corpus_dir / "large.jsonl"
    small_path  = corpus_dir / "small.jsonl"
    medium_path = corpus_dir / "medium.jsonl"

    # ── 1. large.jsonl: copy cleaned/articles.jsonl, assigning article_idx
    #       from the line position when the source doesn't already carry it.
    #       Every article must have an article_idx — downstream stages and
    #       the released HF dataset rely on it as the corpus-global join key.
    if large_path.exists() and large_path.stat().st_size > 0:
        n_large = sum(1 for _ in open(large_path))
        print(f"[1/3] large.jsonl already exists ({n_large:,} articles), skipping")
    else:
        print(f"[1/3] large.jsonl ← copying {cleaned_path} (assigning article_idx if missing)")
        n_large = 0
        with open(cleaned_path) as fin, open(large_path, "w") as fout:
            for line_pos, line in enumerate(fin):
                rec = json.loads(line)
                rec.setdefault("article_idx", line_pos)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_large += 1
        size_mb = large_path.stat().st_size / 1e6
        print(f"      {n_large:,} articles, {size_mb:.1f} MB")

    # ── 2. small.jsonl: union of `supports` across good_qas.jsonl ────────
    if small_path.exists() and small_path.stat().st_size > 0:
        small_idxs = _read_idxs(small_path)
        small_count = len(small_idxs)
        print(f"\n[2/3] small.jsonl already exists ({small_count:,} articles), skipping")
    else:
        print(f"\n[2/3] small.jsonl ← collecting `supports` from {good_qas_path}")
        good_qas = load_jsonl(str(good_qas_path))
        needed_idxs = set()
        n_qa_with_support = 0
        for qa in good_qas:
            sup = qa.get("supports") or []
            if sup:
                n_qa_with_support += 1
            for aidx in sup:
                needed_idxs.add(int(aidx))
        print(f"      {len(good_qas):,} good QAs, {n_qa_with_support:,} with ≥1 support")
        print(f"      {len(needed_idxs):,} unique supporting article_idx values")

        # Single ordered pass over large.jsonl (which now has article_idx
        # everywhere). Records come out sorted by article_idx for free.
        small = []
        with open(large_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["article_idx"] in needed_idxs:
                    small.append(rec)
                    if len(small) == len(needed_idxs):
                        break
        small_idxs = {r["article_idx"] for r in small}
        if small_idxs != needed_idxs:
            missing = needed_idxs - small_idxs
            print(f"      WARNING: {len(missing)} supporting article_idx not found in {large_path.name}")
            print(f"      first few missing: {sorted(missing)[:10]}")
        save_jsonl(small, str(small_path))
        small_count = len(small)
        size_mb = small_path.stat().st_size / 1e6
        print(f"      {small_count:,} articles, {size_mb:.1f} MB")

    # ── 3. medium.jsonl: clustered articles ∪ small (so small ⊆ medium) ─
    #       Without unioning small in, articles that the support-retrieval
    #       (stage 14/15) pulled from outside the clustered subset would be
    #       in `small` but missing from `medium`. We want every retrieval
    #       target to be reachable from `medium`, so we explicitly add them.
    if medium_path.exists() and medium_path.stat().st_size > 0:
        medium_count = sum(1 for _ in open(medium_path))
        print(f"\n[3/3] medium.jsonl already exists ({medium_count:,} articles), skipping")
    else:
        print(f"\n[3/3] medium.jsonl ← clustered ∪ small")
        with open(clustered_path) as f:
            clustered = json.load(f)
        clustered_idxs = set()
        for arts in clustered.values():
            for art in arts:
                aidx = art.get("article_idx")
                if aidx is not None:
                    clustered_idxs.add(int(aidx))
        target_idxs = clustered_idxs | small_idxs
        added_from_small = len(small_idxs - clustered_idxs)
        print(f"      |clustered| = {len(clustered_idxs):,}, "
              f"|small \\ clustered| = {added_from_small:,}, "
              f"|union| = {len(target_idxs):,}")

        medium = []
        with open(large_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["article_idx"] in target_idxs:
                    medium.append(rec)
                    if len(medium) == len(target_idxs):
                        break
        save_jsonl(medium, str(medium_path))
        medium_count = len(medium)
        size_mb = medium_path.stat().st_size / 1e6
        print(f"      {medium_count:,} articles, {size_mb:.1f} MB")

    # ── 4. Build {large,medium,small}/{train,val,test}.jsonl (90/5/5) ────
    print("\n[4/4] Train/val/test split (90/5/5)")
    from tools.split import split_train_val_test
    seed = config.get("seed", 42)
    for folder_name, source in [
        ("large",  large_path),
        ("medium", medium_path),
        ("small",  small_path),
    ]:
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
    print(f"  large.jsonl   {n_large:>10,} articles")
    print(f"  medium.jsonl  {medium_count:>10,} articles  (clustered ∪ small)")
    print(f"  small.jsonl   {small_count:>10,} articles")


if __name__ == "__main__":
    main()
