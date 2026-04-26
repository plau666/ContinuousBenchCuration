"""Stage 19: Per-slice corpus token-count distributions + QA support-count stats.

What it does:
  1. For each corpus slice (`large`, `median`, `small`), tokenizes every
     article with the Gemma 3 tokenizer (or whitespace fallback) and writes
     the raw counts, a percentile summary, and a KDE plot with inline
     cutoff annotations at 256 / 512 / 1024 / 2048 / 4096 tokens.
  2. For the released QA splits (`val`, `test`), reports per-split
     support-count summaries and a combined histogram.

Inputs (under {output_dir}/{version}/):
  qa/final/all_qas.jsonl                cluster-grouped raw QAs
                                        (schema: {cluster_id, qas: [{supports: [...]}]})
  qa/final/filtered/{val,test}.jsonl    released QA splits, flat
                                        (schema: {question, supports: [...]})
  corpus/{large,median,small}.jsonl     per-slice full corpora

Outputs (under {output_dir}/{version}/stats/):
  qa_summary.json                        cluster + QA counts and per-split support stats
  support_stats_per_split.json           {val, test} → {n_qas, mean, median, p25, p75, std, max}
  support_count_dist.png                 histogram across val + test
  token_counts_{large,median,small}.json   raw per-article token counts
  token_stats_{large,median,small}.json    summary {n, mean, std, p25/median/p75/p90/p99, max}
  token_dist_News_{Large,Medium,Small}.png KDE plot per slice (x_max=4096)

Usage:
    # Standard run on the version named in config.yaml, 32 workers, Gemma 3
    python 19_compute_stats.py --config config.yaml --workers 32

    # Pin a specific version (overrides config.yaml)
    python 19_compute_stats.py --config config.yaml --version 2025_09

    # Whitespace fallback (no Gemma install required)
    python 19_compute_stats.py --config config.yaml --no-gemma

    # Re-compute support stats only (skip tokenization)
    python 19_compute_stats.py --config config.yaml --skip-tokens

    # Only one or two slices (e.g. iterating on large)
    python 19_compute_stats.py --config config.yaml --slices large
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, ensure_output_dir


SLICE_NAMES = ["large", "median", "small"]
# All slices use the same blue (matches geminon's aggregate plot style).
SLICE_COLOR = "#1F77B4"
# Display name shown in the plot title and used as the filename slug, e.g.
# `News_Large` → `token_dist_News_Large.png`. Matches geminon's
# `token_dist_Geminon{,200k,1m}.png` naming.
SLICE_DISPLAY = {
    "large":  "News_Large",
    "median": "News_Medium",
    "small":  "News_Small",
}


# ─── Tokenization workers ───────────────────────────────────────────────────
def _whitespace_init():
    pass


def _whitespace_batch(batch):
    return [len(t.split()) for t in batch]


def _gemma_init():
    global _tok
    from gemma.gm.text import Gemma3Tokenizer
    _tok = Gemma3Tokenizer()


def _gemma_batch(batch):
    return [len(_tok.encode(t)) for t in batch]


def compute_token_counts(corpus_path, n_workers=32, batch_size=512, use_gemma=True):
    print(f"  Reading {corpus_path}")
    cur, batches = [], []
    with open(corpus_path) as f:
        for line in f:
            item = json.loads(line)
            cur.append(item.get("text", ""))
            if len(cur) == batch_size:
                batches.append(cur)
                cur = []
    if cur:
        batches.append(cur)
    total = sum(len(b) for b in batches)
    print(f"  {total:,} articles in {len(batches):,} batches, {n_workers} workers")

    if use_gemma:
        # Probe the import in the parent — see the matching note in
        # geminon_curation/08_compute_stats.py. Without this, a missing
        # `gemma` package causes the worker pool to enter an infinite
        # fork loop, burning CPU silently.
        try:
            from gemma.gm.text import Gemma3Tokenizer  # noqa: F401
            initializer = _gemma_init
            worker = _gemma_batch
            print("  Using Gemma3Tokenizer")
        except Exception as e:
            print(f"  Gemma3 unavailable in parent ({type(e).__name__}: {e}), "
                  "falling back to whitespace")
            initializer, worker = _whitespace_init, _whitespace_batch
    else:
        initializer, worker = _whitespace_init, _whitespace_batch
        print("  Using whitespace tokenizer")

    counts = []
    with mp.Pool(n_workers, initializer=initializer) as pool:
        for batch_result in pool.imap_unordered(worker, batches, chunksize=4):
            counts.extend(batch_result)
    return counts


def token_summary(counts):
    """Return dict of summary statistics for a list of token counts."""
    if not counts:
        return {"n": 0}
    arr = np.array(counts, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def plot_token_dist(counts, label, out_path, color=SLICE_COLOR, x_max=4096,
                    reference_marks=(256, 512, 1024, 2048, 4096)):
    """Single-slice KDE plot with inline cutoff markers.

    Default `x_max=4096` covers ~99.5% of news article token counts. Pass
    `x_max=None` to use the actual maximum (no clipping) or any other
    integer to truncate the x-axis.

    For each cutoff `c` in `reference_marks` (within `x_max`) we drop a
    dotted vertical line at c, label it with the cutoff number in bold
    grey near the top of the curve, and write the cumulative percentage
    of articles with `tokens <= c` in the curve's color just below.
    Cumulative % is always computed from raw (unclipped) counts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    raw = np.array(counts, dtype=np.float32)
    if len(raw) == 0:
        return
    if x_max is None:
        x_max = int(raw.max())
    arr = np.clip(raw, 0, x_max)
    mu, med, sd = float(arr.mean()), float(np.median(arr)), float(arr.std())

    rng = np.random.default_rng(0)
    sample = rng.choice(arr, size=min(80_000, len(arr)), replace=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    y = None
    if len(set(sample.tolist())) >= 2:
        kde = gaussian_kde(sample, bw_method=0.15)
        x = np.linspace(0, x_max, 1000)
        y = kde(x)
        ax.fill_between(x, y, alpha=0.25, color=color, zorder=2)
        ax.plot(x, y, color=color, linewidth=2.0, zorder=3,
                label=(f"Per-article  (mean={mu:.0f}, median={med:.0f}, "
                       f"std={sd:.0f}, n={len(raw):,})"))

    y_top = float(y.max()) if y is not None else 1.0
    x_offset = max(2.0, x_max * 0.01)
    for x_mark in reference_marks:
        if x_mark > x_max:
            continue
        pct = float((raw <= x_mark).mean()) * 100
        ax.axvline(x_mark, color="#888888", linewidth=1.0, linestyle=":", alpha=0.6, zorder=1)
        ax.text(x_mark + x_offset, y_top * 0.92, f"{x_mark}",
                fontsize=10, color="#555555", fontweight="bold", va="top")
        ax.text(x_mark + x_offset, y_top * 0.82, f"{pct:.1f}%",
                fontsize=9.5, color=color, va="top")

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Token count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Token Count Distributions — {label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_support_count_dist(counts, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array(counts)
    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(0, max(50, np.percentile(arr, 99) + 1), 50) if len(arr) else 10
    ax.hist(arr, bins=bins, color="#FF7F0E", alpha=0.85)
    ax.set_xlabel("# supporting articles per QA")
    ax.set_ylabel("Count")
    ax.set_title(f"Support-count distribution (n={len(arr):,})",
                 fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def _support_stats(counts):
    if not counts:
        return {"n_qas": 0}
    arr = np.array(counts)
    return {
        "n_qas": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "max": int(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="QA + per-slice corpus stats")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--version", type=str, default=None,
                        help="Override config['version'] for this run.")
    parser.add_argument("--no-gemma", action="store_true")
    parser.add_argument("--skip-tokens", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--slices", type=str, nargs="+", default=None,
                        help=f"Subset of corpus slices to process (default: {SLICE_NAMES})")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.version:
        config["version"] = args.version
    output_dir = ensure_output_dir(config)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. QA + support stats ──────────────────────────────────────────
    # 2025_09 layout:  qa/final/all_qas.jsonl  (cluster-grouped)
    #                  qa/final/filtered/{good_qas,val,test}.jsonl  (flat)
    print("\n=== QA / support counts ===")
    summary = {}
    qa_root = output_dir / "qa" / "final"

    # Cluster-level totals (kept in qa_summary.json for provenance only —
    # the per-split tables below are what the README displays).
    all_qas_path = qa_root / "all_qas.jsonl"
    if all_qas_path.exists():
        recs = load_jsonl(str(all_qas_path))
        n_clusters = len(recs)
        n_qas = sum(len(r.get("qas", [])) for r in recs)
        summary["all_qas"] = {"n_clusters": n_clusters, "n_qas": n_qas}
        print(f"  all_qas (raw): {n_clusters} clusters, {n_qas} QAs")

    # Per-split support stats. Released splits only (val + test) — all_qas
    # and good_qas include filtered-out items (underspecified / zeroshot),
    # so reporting them in the README would be misleading.
    support_stats = {}
    val_test_counts = []
    for split_name, fname in [("val", "val.jsonl"), ("test", "test.jsonl")]:
        path = qa_root / "filtered" / fname
        if not path.exists():
            continue
        recs = load_jsonl(str(path))
        counts = [len(r.get("supports") or []) for r in recs]
        support_stats[split_name] = _support_stats(counts)
        val_test_counts.extend(counts)
        print(f"  {split_name:<5}: n={len(counts):,}, mean={np.mean(counts):.1f}, "
              f"max={max(counts) if counts else 0}, n_zero={sum(1 for c in counts if c==0)}")
    if val_test_counts:
        plot_support_count_dist(val_test_counts, stats_dir / "support_count_dist.png")

    if support_stats:
        with open(stats_dir / "support_stats_per_split.json", "w") as f:
            json.dump(support_stats, f, indent=2)
        print(f"  Saved {stats_dir / 'support_stats_per_split.json'}")
        summary["support_stats_per_split"] = support_stats

    # ── 2. Per-slice token stats ────────────────────────────────────────
    if not args.skip_tokens:
        print("\n=== Token counts (per corpus slice) ===")
        slices = args.slices or SLICE_NAMES
        corpus_dir = output_dir / "corpus"
        counts_by_slice = {}
        token_stats_by_slice = {}

        for name in slices:
            slice_path = corpus_dir / f"{name}.jsonl"
            if not slice_path.exists():
                print(f"\n[{name}] Skipping — not found at {slice_path}")
                continue

            print(f"\n[{name}]")
            counts = compute_token_counts(
                slice_path, n_workers=args.workers, use_gemma=not args.no_gemma
            )
            counts_by_slice[name] = counts

            counts_path = stats_dir / f"token_counts_{name}.json"
            with open(counts_path, "w") as f:
                json.dump(counts, f)
            print(f"  Saved {counts_path}")

            stats = token_summary(counts)
            token_stats_by_slice[name] = stats
            stats_path = stats_dir / f"token_stats_{name}.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  Saved {stats_path}  "
                  f"(n={stats['n']:,}, mean={stats.get('mean', 0):.0f}, "
                  f"median={stats.get('median', 0):.0f}, p99={stats.get('p99', 0):.0f})")

            display = SLICE_DISPLAY[name]
            plot_path = stats_dir / f"token_dist_{display}.png"
            plot_token_dist(counts, display, plot_path)

        if token_stats_by_slice:
            summary["token_stats_per_slice"] = token_stats_by_slice

    with open(stats_dir / "qa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {stats_dir / 'qa_summary.json'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
