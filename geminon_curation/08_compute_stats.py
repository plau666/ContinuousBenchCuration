"""Stage 8: Compute stats and plots over the corpus and QA splits.

Three outputs:
  1. Token-count distributions (overlaid by article type) for all_deduped,
     sampled_200k, and sampled_1m. Tokenizes with Gemma3Tokenizer if
     available; otherwise falls back to whitespace splitting.
  2. Summary stats JSON per slice — per article type, with
     {n, mean, std, min, p25, median, p75, p90, p99, max}.
  3. Per-attribute support count stats (mean/median/p25/p75/max/n_zero)
     for the val+test QA splits in qa/small and qa/medium.

Outputs (under {output_dir}/{version}/stats/):
  token_counts_{name}.json         raw per-article counts grouped by type
  token_stats_{name}.json          summary stats per type
  token_dist_overlaid_{name}.png   KDE plot, per-type curves overlaid (no mean lines)
  token_dist_{name}.png            Aggregate KDE plot over all types + cumulative
                                   % at reference cutoffs 64/128/256
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
        # Probe the import IN THE PARENT before spawning workers — otherwise
        # an ImportError fires repeatedly inside each worker (invisible to
        # the parent), the pool respawns failing workers in a tight loop,
        # and we burn CPU forever instead of falling back. cf. the 2025-04-26
        # fork-loop incident.
        try:
            from gemma.gm.text import Gemma3Tokenizer  # noqa: F401
            initializer = _init_gemma_worker
            worker = _gemma_batch
            print("  Using Gemma3Tokenizer")
        except Exception as e:
            print(f"  Gemma3 unavailable in parent ({type(e).__name__}: {e}), "
                  "falling back to whitespace")
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


# ─── Summary stats ──────────────────────────────────────────────────────────
def token_summary(counts):
    """Return summary stats for a flat list of per-article token counts."""
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


def token_stats_per_type(token_counts):
    """Return {article_type: summary_dict} for the dict-of-lists token counts."""
    return {t: token_summary(counts) for t, counts in token_counts.items()}


# ─── Plotting ───────────────────────────────────────────────────────────────
def plot_token_dist_overlaid(token_counts, title, out_path, x_max=256):
    """Per-type KDE curves overlaid on one axis. No per-type vertical mean
    lines (the means are shown numerically in the legend).
    Saved as `token_dist_overlaid_*.png`.
    """
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

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Token count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_token_dist(token_counts, title, out_path, x_max=320,
                    reference_marks=(64, 128, 256)):
    """Aggregate KDE plot: concatenate counts across all article types into
    one distribution, plot a single KDE curve, and annotate vertical
    reference lines at fixed token thresholds (default 64 / 128 / 256) with
    cumulative-percentage labels (`X% ≤ mark`). Saved as `token_dist_*.png`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # Aggregate all article types into one flat array
    all_counts = []
    for t in ORDER:
        if t in token_counts and token_counts[t]:
            all_counts.extend(token_counts[t])
    # Include any types not in ORDER too
    for t, vals in token_counts.items():
        if t not in ORDER and vals:
            all_counts.extend(vals)

    if not all_counts:
        print(f"  No counts to plot for {out_path}")
        return

    raw = np.array(all_counts, dtype=np.float32)
    arr = np.clip(raw, 0, x_max)
    mu, med, sd = float(arr.mean()), float(np.median(arr)), float(arr.std())

    fig, ax = plt.subplots(figsize=(11, 5))
    color = "#1F77B4"
    x_grid = np.linspace(0, x_max, 1000)

    rng = np.random.default_rng(0)
    sample = rng.choice(arr, size=min(80_000, len(arr)), replace=False)
    y = None
    if len(set(sample.tolist())) >= 2:
        kde = gaussian_kde(sample, bw_method=0.15)
        y = kde(x_grid)
        ax.fill_between(x_grid, y, alpha=0.25, color=color, zorder=2)
        ax.plot(x_grid, y, color=color, linewidth=2.0, zorder=3,
                label=(f"Per-article  (mean={mu:.0f}, median={med:.0f}, "
                       f"std={sd:.0f}, n={len(arr):,})"))

    # Inline cutoff markers, matching the news style: dotted vertical line,
    # bold cutoff number near the top of the curve, percentage just below in
    # the curve color. Positioned slightly to the right of each line so the
    # text doesn't sit on top of it. Cumulative % is computed from raw counts
    # so cutoffs above x_max still reflect everything below them correctly.
    y_top = float(y.max()) if y is not None else 1.0
    x_offset = max(2.0, x_max * 0.01)  # nudge labels right of the line
    for x in reference_marks:
        if x > x_max:
            continue
        pct = float((raw <= x).mean()) * 100
        ax.axvline(x, color="#888888", linewidth=1.0, linestyle=":", alpha=0.6, zorder=1)
        ax.text(x + x_offset, y_top * 0.92, f"{x}",
                fontsize=10, color="#555555", fontweight="bold", va="top")
        ax.text(x + x_offset, y_top * 0.82, f"{pct:.1f}%",
                fontsize=9.5, color=color, va="top")

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Token count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
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
    # if "evolution line of" in q:
    #     return "evolution_line"
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
        # Display name → corpus path. Mirrors the news naming convention
        # (`News_Large/Medium/Small`): `large` is the full deduped corpus,
        # `medium` is the 1M sample, `small` is the 200k sample.
        corpora = {
            "Geminon_Large":  corpus_dir / "all_deduped.jsonl",
            "Geminon_Medium": corpus_dir / "sampled_1m.jsonl",
            "Geminon_Small":  corpus_dir / "sampled_200k.jsonl",
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

            stats = token_stats_per_type(counts)
            stats_path = stats_dir / f"token_stats_{name}.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  Saved {stats_path}")

            for t in ORDER:
                if t not in stats:
                    continue
                s = stats[t]
                print(f"    {t:<20}: n={s['n']:>9,}  mean={s.get('mean', 0):>5.0f}  "
                      f"median={s.get('median', 0):>5.0f}  p99={s.get('p99', 0):>5.0f}")

            # Two plots per slice:
            #   token_dist_overlaid_{name}.png  — original style with per-type mean lines
            #   token_dist_{name}.png           — reference marks at 64/128/256, no per-type means
            overlaid_path = stats_dir / f"token_dist_overlaid_{name}.png"
            plot_token_dist_overlaid(counts, f"Token Count Distributions — {name}", overlaid_path)

            ref_path = stats_dir / f"token_dist_{name}.png"
            plot_token_dist(counts, f"Token Count Distributions — {name}", ref_path)

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
                          f"median={s['median']:>6.1f}, max={s['max']:>5}")

            out_path = stats_dir / f"support_stats_{folder_name}.json"
            with open(out_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  Saved {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
