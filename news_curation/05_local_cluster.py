"""Stage 5: Local windowed kNN + Leiden clustering for event detection.

Reads the cleaned articles + embeddings produced by stages 3-4, runs
windowed clustering, merges across windows, and writes
{output_dir}/{version}/clustered/clustered_articles.json (the canonical
input format for the QA pipeline downstream).

This is a thin wrapper around an external clustering implementation
(~800 lines of GPU+Leiden code) — duplicating it here would be wasteful.
The wrapper constructs the right CLI args from config.yaml and shells out.

Point it at your copy of the source script via either:
  - the `CC_LOCAL_CLUSTER_SCRIPT` environment variable, or
  - the `--source-script` flag.

Usage:
    CC_LOCAL_CLUSTER_SCRIPT=/path/to/local_cluster.py \\
        python 05_local_cluster.py --config config.yaml

    # or
    python 05_local_cluster.py --config config.yaml --source-script /path/to/local_cluster.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils.io import load_config, ensure_output_dir


def main():
    parser = argparse.ArgumentParser(description="Local clustering wrapper")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--source-script", type=str,
        default=os.environ.get("CC_LOCAL_CLUSTER_SCRIPT"),
        help="Path to the external local_cluster.py implementation. "
             "Defaults to $CC_LOCAL_CLUSTER_SCRIPT if set; required otherwise.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    args = parser.parse_args()

    if not args.source_script:
        print("ERROR: --source-script not set and $CC_LOCAL_CLUSTER_SCRIPT is empty.")
        print("Point this at your copy of local_cluster.py and re-run.")
        sys.exit(2)

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    embeds_path = output_dir / "embeds" / "text_embeds.npy"
    corpus_path = output_dir / "cleaned" / "articles.jsonl"
    clustered_dir = output_dir / "clustered"
    clustered_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.source_script).exists():
        print(f"ERROR: source script not found: {args.source_script}")
        print("Pass --source-script PATH to point at your local_cluster.py")
        sys.exit(1)
    if not embeds_path.exists():
        print(f"ERROR: embeddings missing — run 04_compute_embeddings.py first ({embeds_path})")
        sys.exit(1)
    if not corpus_path.exists():
        print(f"ERROR: corpus missing — run 03_cleanup_dedup.py first ({corpus_path})")
        sys.exit(1)

    cfg = config["clustering"]
    cmd = [
        sys.executable, args.source_script,
        "--embeds", str(embeds_path),
        "--corpus", str(corpus_path),
        "--output_dir", str(clustered_dir),
        "--window-days", str(cfg["window_days"]),
        "--window-step", str(cfg["window_step"]),
        "--k-search", str(cfg["k_search"]),
        "--k-graph", str(cfg["k_graph"]),
        "--sim-threshold", str(cfg["sim_threshold"]),
        "--leiden-resolution", str(cfg["leiden_resolution"]),
        "--leiden-iterations", str(cfg["leiden_iterations"]),
        "--min-cluster-size", str(cfg["min_cluster_size"]),
        "--jaccard-threshold", str(cfg["jaccard_threshold"]),
        "--centroid-threshold", str(cfg["centroid_threshold"]),
        "--span-cap-days", str(cfg["span_cap_days"]),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print(f"\nClustering complete. Outputs in {clustered_dir}/")


if __name__ == "__main__":
    main()
