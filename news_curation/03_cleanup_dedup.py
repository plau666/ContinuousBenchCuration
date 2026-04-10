"""Stage 3: Normalize text, filter low quality, and exact + near dedup.

Reads all per-WARC JSONL files from {output_dir}/{version}/extracted/,
merges them, cleans + filters, then runs SHA-256 exact dedup followed by
MinHash LSH near-dedup (containment similarity, configurable threshold).

Output: {output_dir}/{version}/cleaned/articles.jsonl

Usage:
    python 03_cleanup_dedup.py --config config.yaml
    python 03_cleanup_dedup.py --config config.yaml --exact-only
"""

import argparse
import glob
import json
import time
from pathlib import Path

from tqdm import tqdm

from utils.io import load_config, save_jsonl, ensure_output_dir
from utils.normalize import clean_article, is_valid
from utils.dedup import exact_dedup, near_dedup


def main():
    parser = argparse.ArgumentParser(description="Cleanup + dedup news articles")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--exact-only", action="store_true", help="Skip near-dedup")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    extracted_dir = output_dir / "extracted"
    cleaned_dir = output_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    cfg = config["cleanup"]
    min_text_length = cfg["min_text_length"]
    min_word_count = cfg["min_word_count"]

    # 1. Load and merge per-WARC JSONL files
    files = sorted(glob.glob(str(extracted_dir / "*.jsonl")))
    print(f"Loading {len(files)} extracted JSONL files from {extracted_dir}")
    articles = []
    for fp in tqdm(files, desc="  Loading", unit="file"):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    articles.append(json.loads(line))
    print(f"  Loaded {len(articles):,} articles")

    # 2. Normalize text
    print("\n--- Step 1: Text normalization ---")
    t0 = time.time()
    articles = [clean_article(a) for a in tqdm(articles, desc="  Cleaning", unit="article")]
    print(f"  Done in {time.time()-t0:.1f}s")

    # 3. Quality filter
    print("\n--- Step 2: Quality filter ---")
    before = len(articles)
    articles = [a for a in articles if is_valid(a, min_text_length, min_word_count)]
    print(f"  Removed {before - len(articles):,} invalid articles ({len(articles):,} remain)")

    # 4. Exact dedup
    print("\n--- Step 3: Exact dedup (SHA-256) ---")
    t0 = time.time()
    before = len(articles)
    articles = exact_dedup(articles)
    print(f"  Removed {before - len(articles):,} exact duplicates ({len(articles):,} remain) in {time.time()-t0:.1f}s")

    # 5. Near dedup
    if not args.exact_only:
        print(f"\n--- Step 4: Near-dedup (containment >= {cfg['dedup_threshold']}, "
              f"{cfg['dedup_shingle_size']}-gram, {cfg['dedup_num_perm']} perms) ---")
        t0 = time.time()
        before = len(articles)
        articles = near_dedup(
            articles,
            threshold=cfg["dedup_threshold"],
            num_perm=cfg["dedup_num_perm"],
            shingle_size=cfg["dedup_shingle_size"],
            workers=cfg["dedup_workers"],
        )
        print(f"  Removed {before - len(articles):,} near-duplicates "
              f"({len(articles):,} remain) in {time.time()-t0:.1f}s")

    # 6. Assign a stable article_idx to each kept article (carries through later stages)
    for i, art in enumerate(articles):
        art["article_idx"] = i

    # 7. Save
    output_path = cleaned_dir / "articles.jsonl"
    save_jsonl(articles, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved {len(articles):,} articles ({size_mb:.1f} MB) to {output_path}")
    print(f"  Each article carries an `article_idx` field (0..{len(articles)-1}) for downstream tracking")


if __name__ == "__main__":
    main()
