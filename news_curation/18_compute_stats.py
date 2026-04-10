"""Stage 17: Compute QA stats and corpus token distributions.

Two outputs:
  1. Per-stage QA counts (clusters, total QAs, judged stats, supports stats)
  2. Token-count distribution for the cleaned corpus, with KDE plot

Inputs:  {output_dir}/{version}/cleaned/articles.jsonl
         {output_dir}/{version}/qa/{qas,qas_judged,qas_with_supports}.jsonl
Outputs: {output_dir}/{version}/stats/qa_summary.json
         {output_dir}/{version}/stats/support_count_dist.png
         {output_dir}/{version}/stats/token_counts.json
         {output_dir}/{version}/stats/token_dist.png

Usage:
    python 17_compute_stats.py --config config.yaml
    python 17_compute_stats.py --config config.yaml --no-gemma
    python 17_compute_stats.py --config config.yaml --skip-tokens
"""

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, ensure_output_dir


# ─── Tokenization ───────────────────────────────────────────────────────────
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


def plot_token_dist(counts, title, out_path, x_max=2048):
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
        ax.fill_between(x, kde(x), alpha=0.30, color="#1F77B4")
        ax.plot(x, kde(x), color="#1F77B4", linewidth=2.0,
                label=f"cleaned corpus (μ={mu:.0f}, σ={sd:.0f}, n={len(arr):,})")
    ax.axvline(mu, color="#1F77B4", linewidth=1.2, alpha=0.8)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Token count")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
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
    parser = argparse.ArgumentParser(description="QA + corpus stats")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-gemma", action="store_true")
    parser.add_argument("--skip-tokens", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
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

    with open(stats_dir / "qa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {stats_dir / 'qa_summary.json'}")

    # ── 2. Token counts + plot for cleaned corpus ──────────────────────
    if not args.skip_tokens:
        print("\n=== Token counts ===")
        cleaned_path = output_dir / "cleaned" / "articles.jsonl"
        if cleaned_path.exists():
            counts = compute_token_counts(
                cleaned_path, n_workers=args.workers, use_gemma=not args.no_gemma
            )
            with open(stats_dir / "token_counts.json", "w") as f:
                json.dump(counts, f)
            print(f"  Saved {stats_dir / 'token_counts.json'}")
            plot_token_dist(
                counts, "Token Count Distribution — cleaned corpus",
                stats_dir / "token_dist.png"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
