"""Stage 8: Compute stats and plots over the corpus and QA splits.

Two outputs:
  1. Token-count distributions (overlaid by type) for all_deduped,
     sampled_200k, and sampled_1m. Tokenizes with Gemma3Tokenizer if
     available; otherwise falls back to whitespace splitting.
  2. Per-attribute support count stats (mean/median/p25/p75/max/n_zero)
     for the val+test QA splits in qa/small and qa/medium.

Outputs (under {output_dir}/{version}/stats/):
  token_counts_{name}.json
  token_dist_overlaid_{name}.png
  support_stats_{folder}.json

Usage:
  python 08_compute_stats.py --config config.yaml
  python 08_compute_stats.py --config config.yaml --tokenizer-workers 32
  python 08_compute_stats.py --config config.yaml --skip-tokens
"""

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, ensure_output_dir


# ─── Token-count display config (mirrors plot_token_dist.py) ────────────────
ORDER = ["journal", "evolution", "comparison", "wiki", "sensitive_wiki"]
LABELS = {
    "wiki": "Public Wiki",
    "sensitive_wiki": "Sensitive Wiki",
    "journal": "Field Journal",
    "comparison": "Comparison",
    "evolution": "Evolution Log",
}
COLORS = {
    "journal": "#2CA02C",
    "evolution": "#FF7F0E",
    "comparison": "#D62728",
    "wiki": "#1F77B4",
    "sensitive_wiki": "#17BECF",
}


# ─── Tokenization ───────────────────────────────────────────────────────────
def _init_gemma_worker():
    global _tok
    from gemma.gm.text import Gemma3Tokenizer
    _tok = Gemma3Tokenizer()


def _gemma_batch(batch):
    return [(t, len(_tok.encode(txt))) for t, txt in batch]


def _whitespace_batch(batch):
    return [(t, len(txt.split())) for t, txt in batch]


def _whitespace_init():
    pass


def compute_token_counts(corpus_path, n_workers=44, batch_size=512, use_gemma=True):
    """Tokenize every text in a corpus JSONL and group counts by type.

    Returns dict: {type: [n_tokens, ...]}.
    """
    print(f"  Reading {corpus_path}")
    cur, batches = [], []
    with open(corpus_path) as f:
        for line in f:
            item = json.loads(line)
            cur.append((item["type"], item["text"]))
            if len(cur) == batch_size:
                batches.append(cur)
                cur = []
    if cur:
        batches.append(cur)
    total = sum(len(b) for b in batches)
    print(f"  {total:,} articles in {len(batches):,} batches, {n_workers} workers")

    if use_gemma:
        try:
            initializer = _init_gemma_worker
            worker = _gemma_batch
            print("  Using Gemma3Tokenizer")
        except Exception as e:
            print(f"  Gemma3 unavailable ({e}), falling back to whitespace")
            initializer = _whitespace_init
            worker = _whitespace_batch
    else:
        initializer = _whitespace_init
        worker = _whitespace_batch
        print("  Using whitespace tokenizer")

    t0 = time.time()
    results = defaultdict(list)
    done = 0
    with mp.Pool(n_workers, initializer=initializer) as pool:
        for batch_result in pool.imap_unordered(worker, batches, chunksize=4):
            for art_type, n_tok in batch_result:
                results[art_type].append(n_tok)
            done += len(batch_result)
            if done % 200_000 == 0:
                print(f"    {done:,}/{total:,} ({done/(time.time()-t0):.0f} art/s)")
    print(f"  Done in {time.time()-t0:.1f}s")
    return dict(results)


# ─── Plotting ───────────────────────────────────────────────────────────────
def plot_token_dist(token_counts, title, out_path, x_max=256):
    """Render an overlaid KDE plot of token counts by article type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    x_grid = np.linspace(0, x_max, 1000)
    fig, ax = plt.subplots(figsize=(11, 5))

    for t in ORDER:
        if t not in token_counts or not token_counts[t]:
            continue
        arr = np.clip(np.array(token_counts[t], dtype=np.float32), 0, x_max)
        mu, sd = arr.mean(), arr.std()
        label = f"{LABELS[t]}  (μ={mu:.0f}, σ={sd:.0f}, n={len(arr):,})"
        color = COLORS[t]

        rng = np.random.default_rng(0)
        sample = rng.choice(arr, size=min(50_000, len(arr)), replace=False)
        # KDE needs at least 2 distinct points
        if len(set(sample.tolist())) < 2:
            continue
        kde = gaussian_kde(sample, bw_method=0.15)
        y = kde(x_grid)

        is_sensitive = (t == "sensitive_wiki")
        if is_sensitive:
            ax.plot(x_grid, y, color=color, linewidth=2.2, linestyle="--", label=label, zorder=4)
            ax.fill_between(x_grid, y, alpha=0.10, color=color, zorder=3)
        else:
            ax.fill_between(x_grid, y, alpha=0.30, color=color, zorder=2)
            ax.plot(x_grid, y, color=color, linewidth=2.0, linestyle="-", label=label, zorder=3)

        ls = "--" if is_sensitive else "-"
        ax.axvline(mu, color=color, linewidth=1.2, linestyle=ls, alpha=0.8, zorder=5)

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Token count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ─── Question -> feature parser ─────────────────────────────────────────────
def question_to_feature(q):
    """Map a QA question to its canonical feature label."""
    if "classification of" in q:
        return "classification"
    if "types of" in q:
        return "types"
    if "ability of" in q:
        return "ability"
    if "HP stat of" in q:
        return "hp"
    if "special attack stat of" in q:
        return "special attack"
    if "special defense stat of" in q:
        return "special defense"
    if "attack stat of" in q:
        return "attack"
    if "defense stat of" in q:
        return "defense"
    if "speed stat of" in q:
        return "speed"
    if "base stat total stat of" in q:
        return "base_stat_total"
    if "move of" in q:
        return "move"
    if "weight" in q:
        return "weight"
    if "height" in q:
        return "height"
    if "evolution line of" in q:
        return "evolution_line"
    return "unknown"


def support_stats_for_qas(qas):
    """Group by feature, compute support count percentile stats."""
    by_feature = defaultdict(list)
    for q in qas:
        feat = question_to_feature(q["question"])
        by_feature[feat].append(len(q.get("supports", [])))

    out = {}
    for feat, sizes in sorted(by_feature.items()):
        arr = np.array(sizes)
        out[feat] = {
            "n_qas": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": int(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": int(arr.max()),
            "n_zero_supports": int((arr == 0).sum()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute stats and plots")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--tokenizer-workers", type=int, default=44)
    parser.add_argument("--no-gemma", action="store_true",
                        help="Skip Gemma3 tokenizer; use whitespace")
    parser.add_argument("--skip-tokens", action="store_true",
                        help="Skip token computation/plots")
    parser.add_argument("--skip-supports", action="store_true",
                        help="Skip support stats")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = output_dir / "corpus"
    qa_dir = output_dir / "qa"

    # ── Token counts + plots ────────────────────────────────────────────
    if not args.skip_tokens:
        print("\n=== Token counts + distribution plots ===")
        corpora = {
            "all_deduped": corpus_dir / "all_deduped.jsonl",
            "sampled_200k": corpus_dir / "sampled_200k.jsonl",
            "sampled_1m": corpus_dir / "sampled_1m.jsonl",
        }
        for name, path in corpora.items():
            if not path.exists():
                print(f"\n[{name}] Skipping — file not found: {path}")
                continue
            print(f"\n[{name}]")
            counts = compute_token_counts(
                path, n_workers=args.tokenizer_workers, use_gemma=not args.no_gemma
            )
            counts_path = stats_dir / f"token_counts_{name}.json"
            with open(counts_path, "w") as f:
                json.dump(counts, f)
            print(f"  Saved {counts_path}")

            for t, c in sorted(counts.items()):
                print(f"    {t:<20}: {len(c):>9,} articles")

            plot_path = stats_dir / f"token_dist_overlaid_{name}.png"
            plot_token_dist(counts, f"Token Count Distributions — {name}", plot_path)

    # ── Per-attribute support stats ────────────────────────────────────
    if not args.skip_supports:
        print("\n=== Per-attribute support count stats ===")
        for folder_name in ["small", "medium"]:
            folder = qa_dir / folder_name
            if not folder.exists():
                print(f"\n[{folder_name}] Skipping — folder not found")
                continue

            print(f"\n[{folder_name}]")
            split_files = [
                "public_val.jsonl", "public_test.jsonl",
                "sensitive_val.jsonl", "sensitive_test.jsonl",
            ]
            stats = {}
            for fname in split_files:
                path = folder / fname
                if not path.exists():
                    continue
                qas = load_jsonl(str(path))
                split_stats = support_stats_for_qas(qas)
                stats[fname.replace(".jsonl", "")] = split_stats
                # Print summary
                print(f"  {fname}:")
                for feat, s in split_stats.items():
                    print(f"    {feat:<20}: n={s['n_qas']:>4}, mean={s['mean']:>7.1f}, "
                          f"median={s['median']:>6.1f}, max={s['max']:>5}, n_zero={s['n_zero_supports']}")

            out_path = stats_dir / f"support_stats_{folder_name}.json"
            with open(out_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  Saved {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
