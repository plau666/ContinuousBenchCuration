"""Stage 5: Process corpus responses (parse, normalize, dedup, balanced sample).

Reads response JSONL files from tools.query_gemini output, parses LLM responses,
normalizes feature names, deduplicates, and produces balanced samples.

Usage:
    python 05_process_corpus.py --config config.yaml
    python 05_process_corpus.py --config config.yaml --skip-dedup
    python 05_process_corpus.py --config config.yaml --skip-sampling
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, save_jsonl, clean_and_parse_response, ensure_output_dir
from utils.normalization import parse_info, INFO_KEYS
from utils.dedup import exact_dedup, near_dedup
from utils.balanced_sampler import balanced_sample_feature_aware, sample_sensitive_articles


# Map response filename pattern -> corpus type
TYPE_MAP = {
    "wiki": "wiki",
    "journal": "journal",
    "comparison": "comparison",
    "chain": "evolution",
    "sensitive_wiki": "sensitive_wiki",
}


def infer_type_from_filename(filename):
    """Infer corpus type from response filename."""
    name = filename.lower()
    # Check sensitive first since it contains "wiki"
    if "sensitive" in name:
        return "sensitive_wiki"
    for key, val in TYPE_MAP.items():
        if key in name:
            return val
    return "unknown"


def parse_response_file(response_path, corpus_type):
    """Parse one response file into normalized records."""
    records = []
    n_errors = 0
    n_total = 0

    raw = load_jsonl(str(response_path))
    for entry in raw:
        n_total += 1
        tag_idxs = entry["tag"]
        parsed, err = clean_and_parse_response(entry["response"])
        if parsed is None:
            n_errors += 1
            continue

        for item in parsed:
            if not isinstance(item, dict) or "text" not in item:
                continue

            tag_list = []
            for i, idx in enumerate(tag_idxs):
                info_key = INFO_KEYS[i] if i < len(INFO_KEYS) else None
                if info_key:
                    raw_info = item.get(info_key, [])
                    info, _ = parse_info(raw_info)
                else:
                    info = []
                tag_list.append({"idx": idx, "info": info})

            records.append({
                "text": item["text"],
                "tag": tag_list,
                "type": corpus_type,
            })

    return records, n_errors, n_total


def main():
    parser = argparse.ArgumentParser(description="Process corpus responses")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--responses-dir", type=str, default=None,
                        help="Directory containing response JSONL files (default: {output_dir}/responses)")
    parser.add_argument("--input-all", type=str, default=None,
                        help="Path to a pre-built all.jsonl. If provided, skips response parsing.")
    parser.add_argument("--sensitive-index", type=str, default=None,
                        help="Path to sensitive_geminon_index.jsonl (default: {output_dir}/sensitive_geminon_index.jsonl)")
    parser.add_argument("--skip-sampling", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    proc_cfg = config["processing"]

    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Build (or load) the full merged corpus ---
    if args.input_all:
        print(f"Loading pre-built corpus from {args.input_all}")
        full_corpus = load_jsonl(args.input_all)
        all_merged = [r for r in full_corpus if r["type"] != "sensitive_wiki"]
        sensitive_merged = [r for r in full_corpus if r["type"] == "sensitive_wiki"]
        print(f"  Loaded {len(full_corpus)} records ({len(all_merged)} public, {len(sensitive_merged)} sensitive)")
    else:
        responses_dir = Path(args.responses_dir) if args.responses_dir else output_dir / "responses"
        print(f"Parsing response files in {responses_dir}/")
        all_merged = []
        sensitive_merged = []

        for resp_file in sorted(responses_dir.glob("*.jsonl")):
            corpus_type = infer_type_from_filename(resp_file.name)
            if corpus_type == "unknown":
                print(f"  Skipping (unknown type): {resp_file.name}")
                continue

            records, n_errors, n_total = parse_response_file(resp_file, corpus_type)
            print(f"  {resp_file.name}: {n_total} prompts -> {len(records)} records ({n_errors} parse errors)")

            if corpus_type == "sensitive_wiki":
                sensitive_merged.extend(records)
            else:
                all_merged.extend(records)

        print(f"\nTotal: {len(all_merged)} public records, {len(sensitive_merged)} sensitive records")

        # --- Step 2: Save full merged corpus ---
        full_corpus = all_merged + sensitive_merged
        save_jsonl(full_corpus, str(corpus_dir / "all.jsonl"))
        print(f"Saved {len(full_corpus)} records to {corpus_dir / 'all.jsonl'}")

    # --- Step 3: Dedup ---
    print("\n--- Dedup ---")
    before = len(full_corpus)

    print(f"Step 1: Exact dedup (SHA-256)")
    deduped = exact_dedup(full_corpus)
    print(f"  Removed {before - len(deduped):,} exact duplicates")
    before2 = len(deduped)

    print(f"Step 2: Near-dedup (containment >= {proc_cfg['dedup_threshold']})")
    deduped = near_dedup(
        deduped,
        threshold=proc_cfg["dedup_threshold"],
        num_perm=proc_cfg["dedup_num_perm"],
        shingle_size=proc_cfg["dedup_shingle_size"],
        workers=proc_cfg["dedup_workers"],
    )
    print(f"  Removed {before2 - len(deduped):,} near-duplicates")

    # Assign a stable article_idx to each deduped article (carried into samples)
    for i, art in enumerate(deduped):
        art["article_idx"] = i

    save_jsonl(deduped, str(corpus_dir / "all_deduped.jsonl"))
    print(f"Saved {len(deduped)} deduped records to {corpus_dir / 'all_deduped.jsonl'}")

    # Re-split into public/sensitive after dedup for sampling
    all_merged = [r for r in deduped if r["type"] != "sensitive_wiki"]
    sensitive_merged = [r for r in deduped if r["type"] == "sensitive_wiki"]

    # --- Step 4: Balanced sampling ---
    if not args.skip_sampling:
        print("\n--- Balanced sampling ---")

        # Load sensitive index for sensitive article selection
        sensitive_index_path = args.sensitive_index or str(output_dir / "sensitive_geminon_index.jsonl")
        sensitive_index = load_jsonl(sensitive_index_path)

        rng = np.random.default_rng(config["seed"])

        by_type = defaultdict(list)
        for item in all_merged:
            by_type[item["type"]].append(item)

        for size_label, targets in [
            ("200k", proc_cfg["sample_targets_200k"]),
            ("1m", proc_cfg["sample_targets_1m"]),
        ]:
            print(f"\n  Sampling {size_label}...")
            sampled = []
            for ctype, target_n in targets.items():
                entries = by_type.get(ctype, [])
                n_rounds = proc_cfg["sampling_rounds"] if len(entries) < 200_000 else 5
                subset = balanced_sample_feature_aware(entries, target_n, rng, n_rounds=n_rounds)
                sampled.extend(subset)
                print(f"    {ctype}: {len(subset)} / {target_n} requested")

            # Add sensitive samples
            sensitive_sampled = sample_sensitive_articles(sensitive_merged, sensitive_index)
            sampled.extend(sensitive_sampled)
            print(f"    sensitive: {len(sensitive_sampled)}")

            rng.shuffle(sampled)

            output_path = corpus_dir / f"sampled_{size_label}.jsonl"
            save_jsonl(sampled, str(output_path))
            print(f"  Saved {len(sampled)} records to {output_path}")

    # --- Step 5: Build large/medium/small subfolders + train/val/test (90/5/5) ---
    # Inside each subfolder the source file is symlinked as `all.jsonl` so the
    # 4 files are uniformly named regardless of which slice you're looking at:
    #   {large,medium,small}/{all,train,val,test}.jsonl
    print("\n--- Train/val/test split (90/5/5) ---")
    from tools.split import split_train_val_test
    seed = config.get("seed", 42)
    slice_specs = [
        ("large",  corpus_dir / "all_deduped.jsonl"),
        ("medium", corpus_dir / "sampled_1m.jsonl"),
        ("small",  corpus_dir / "sampled_200k.jsonl"),
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

    print("\nDone!")


if __name__ == "__main__":
    main()
