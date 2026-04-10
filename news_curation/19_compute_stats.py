"""Stage 19: Compute QA stats and token distributions for each corpus slice.

Two output groups:

  1. QA-stage counts (clusters, total QAs, judged breakdown, supports stats).
     Written to {stats_dir}/qa_summary.json plus support_count_dist.png.

  2. Per-slice token stats. For each of the corpus slices produced by
     stage 18 (large, median, small), this script computes the per-article
     token count via the Gemma3 tokenizer (or whitespace fallback), saves
     the raw counts list, a per-slice summary (n, mean, std, percentiles),
     a per-slice KDE plot, and a single overlay plot comparing all three.

Inputs:  {output_dir}/{version}/qa/{qas,qas_judged,qas_with_supports}.jsonl
         {output_dir}/{version}/corpus/{large,median,small}.jsonl
Outputs: {output_dir}/{version}/stats/qa_summary.json
         {output_dir}/{version}/stats/support_count_dist.png
         {output_dir}/{version}/stats/token_counts_{large,median,small}.json
         {output_dir}/{version}/stats/token_stats_{large,median,small}.json
         {output_dir}/{version}/stats/token_dist_{large,median,small}.png
         {output_dir}/{version}/stats/token_dist_overlay.png

Usage:
    python 19_compute_stats.py --config config.yaml
    python 19_compute_stats.py --config config.yaml --no-gemma
    python 19_compute_stats.py --config config.yaml --skip-tokens
    python 19_compute_stats.py --config config.yaml --slices large median
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, ensure_output_dir


SLICE_NAMES = ["large", "median", "small"]
SLICE_COLORS = {
    "large":  "#1F77B4",  # blue
    "median": "#2CA02C",  # green
    "small":  "#D62728",  # red
}
SLICE_LABELS = {
    "large":  "large (full corpus)",
    "median": "median (clustered)",
    "small":  "small (supports)",
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
        try:
            initializer = _gemma_init
            worker = _gemma_batch
            print("  Using Gemma3Tokenizer")
        except Exception as e:
            print(f"  Gemma3 unavailable ({e}), falling back to whitespace")
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


def plot_token_dist(counts, label, color, out_path, x_max=2048):
    """Single-slice KDE plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    arr = np.clip(np.array(counts, dtype=np.float32), 0, x_max)
    if len(arr) == 0:
        return
    mu, sd = arr.mean(), arr.std()

    rng = np.random.default_rng(0)
    sample = rng.choice(arr, size=min(50_000, len(arr)), replace=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    if len(set(sample.tolist())) >= 2:
        kde = gaussian_kde(sample, bw_method=0.15)
        x = np.linspace(0, x_max, 1000)
        ax.fill_between(x, kde(x), alpha=0.30, color=color)
        ax.plot(x, kde(x), color=color, linewidth=2.0,
                label=f"{label}  (μ={mu:.0f}, σ={sd:.0f}, n={len(arr):,})")
    ax.axvline(mu, color=color, linewidth=1.2, alpha=0.8)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Token count")
    ax.set_ylabel("Density")
    ax.set_title(f"Token Count Distribution — {label}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_overlay(counts_by_slice, out_path, x_max=2048):
    """One overlay KDE plot showing all slices on the same axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.linspace(0, x_max, 1000)
    rng = np.random.default_rng(0)
    for name in SLICE_NAMES:
        if name not in counts_by_slice:
            continue
        counts = counts_by_slice[name]
        if not counts:
            continue
        arr = np.clip(np.array(counts, dtype=np.float32), 0, x_max)
        if len(arr) == 0:
            continue
        mu, sd = arr.mean(), arr.std()
        sample = rng.choice(arr, size=min(50_000, len(arr)), replace=False)
        if len(set(sample.tolist())) < 2:
            continue
        kde = gaussian_kde(sample, bw_method=0.15)
        color = SLICE_COLORS[name]
        ax.fill_between(x, kde(x), alpha=0.20, color=color)
        ax.plot(x, kde(x), color=color, linewidth=2.0,
                label=f"{SLICE_LABELS[name]}  (μ={mu:.0f}, σ={sd:.0f}, n={len(arr):,})")
        ax.axvline(mu, color=color, linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Token count")
    ax.set_ylabel("Density")
    ax.set_title("Token Count Distribution — corpus slices", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
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


def qa_summary(records, label):
    n_clusters = len(records)
    n_qas = sum(len(r.get("qas", [])) for r in records)
    print(f"  {label}: {n_clusters} clusters, {n_qas} QAs")
    return {"n_clusters": n_clusters, "n_qas": n_qas}


def main():
    parser = argparse.ArgumentParser(description="QA + per-slice corpus stats")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-gemma", action="store_true")
    parser.add_argument("--skip-tokens", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--slices", type=str, nargs="+", default=None,
                        help=f"Subset of corpus slices to process (default: {SLICE_NAMES})")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. QA counts at each stage ─────────────────────────────────────
    print("\n=== QA stage counts ===")
    summary = {}
    for label, fname in [
        ("qas", "qas.jsonl"),
        ("qas_with_zeroshot", "qas_with_zeroshot.jsonl"),
        ("qas_judged", "qas_judged.jsonl"),
        ("qas_with_supports", "qas_with_supports.jsonl"),
    ]:
        path = output_dir / "qa" / fname
        if path.exists():
            recs = load_jsonl(str(path))
            summary[label] = qa_summary(recs, label)

    # Judged breakdown
    judged_path = output_dir / "qa" / "qas_judged.jsonl"
    if judged_path.exists():
        recs = load_jsonl(str(judged_path))
        n_zeroshot_correct = n_underspecified = n_good = 0
        for r in recs:
            for qa in r.get("qas", []):
                if qa.get("is_zeroshot_correct"):
                    n_zeroshot_correct += 1
                if qa.get("is_underspecified"):
                    n_underspecified += 1
                if qa.get("is_zeroshot_correct") is False and qa.get("is_underspecified") is False:
                    n_good += 1
        summary["judged_breakdown"] = {
            "zeroshot_correct": n_zeroshot_correct,
            "underspecified": n_underspecified,
            "good": n_good,
        }
        print(f"  Judged: zeroshot_correct={n_zeroshot_correct}, "
              f"underspecified={n_underspecified}, good={n_good}")

    # Support counts per QA
    sup_path = output_dir / "qa" / "qas_with_supports.jsonl"
    if sup_path.exists():
        recs = load_jsonl(str(sup_path))
        sup_counts = []
        for r in recs:
            for qa in r.get("qas", []):
                sup_counts.append(len(qa.get("supports") or []))
        if sup_counts:
            arr = np.array(sup_counts)
            summary["support_count_stats"] = {
                "n_qas": int(len(arr)),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "max": int(arr.max()),
                "n_zero": int((arr == 0).sum()),
            }
            print(f"  Supports: avg {arr.mean():.1f} per QA, max {arr.max()}, n_zero {(arr==0).sum()}")
            plot_support_count_dist(sup_counts, stats_dir / "support_count_dist.png")

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

            plot_path = stats_dir / f"token_dist_{name}.png"
            plot_token_dist(counts, SLICE_LABELS[name], SLICE_COLORS[name], plot_path)

        # Overlay plot comparing all available slices on one axis
        if counts_by_slice:
            plot_overlay(counts_by_slice, stats_dir / "token_dist_overlay.png")

        if token_stats_by_slice:
            summary["token_stats_per_slice"] = token_stats_by_slice

    with open(stats_dir / "qa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {stats_dir / 'qa_summary.json'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
