"""Push curation outputs to a HuggingFace dataset repo.

Uploads selected files/folders from a local curation output directory to
a HuggingFace dataset repo. Each curation (geminon, news, ...) gets its
own subfolder inside the repo, and each version gets its own subfolder
inside that.

Default behavior uploads `corpus/`, `qa/`, and the geminon index files
to `{curation}/{version}/...` in the dataset repo.

Usage:
    # Set HF_TOKEN env var or pass --token
    export HF_TOKEN=hf_xxx

    # Push everything from output/v9/ in geminon_curation
    python -m tools.push_to_hf \
        --repo pl666/ContinuousBench \
        --curation geminon \
        --version v9 \
        --local-dir /home/peihanliu/ContinuousBenchCuration/geminon_curation/output/v9

    # Push only specific paths
    python -m tools.push_to_hf \
        --repo pl666/ContinuousBench \
        --curation geminon \
        --version v9 \
        --local-dir output/v9 \
        --include qa corpus/sampled_200k.jsonl corpus/sampled_1m.jsonl

    # Dry run
    python -m tools.push_to_hf ... --dry-run
"""

import argparse
import os
from pathlib import Path


# Default subpaths to upload (relative to --local-dir)
DEFAULT_INCLUDES = [
    "geminon_index.jsonl",
    "public_geminon_index.jsonl",
    "sensitive_geminon_index.jsonl",
    "qa",
    "corpus/all_deduped.jsonl",
    "corpus/sampled_200k.jsonl",
    "corpus/sampled_1m.jsonl",
]


def collect_files(local_dir, includes):
    """Expand include paths into a list of (local_path, repo_path) tuples.

    repo_path is relative to local_dir (preserves subdirectory structure).
    """
    local_dir = Path(local_dir).resolve()
    files = []
    for inc in includes:
        full = local_dir / inc
        if not full.exists():
            print(f"  Warning: {full} does not exist, skipping")
            continue
        if full.is_file():
            files.append((full, full.relative_to(local_dir).as_posix()))
        elif full.is_dir():
            for f in sorted(full.rglob("*")):
                if f.is_file():
                    files.append((f, f.relative_to(local_dir).as_posix()))
    return files


def main():
    parser = argparse.ArgumentParser(description="Push curation outputs to HuggingFace dataset repo")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace dataset repo id, e.g. pl666/ContinuousBench")
    parser.add_argument("--curation", type=str, required=True,
                        help="Curation name (e.g. geminon, news) — becomes top-level subfolder in the repo")
    parser.add_argument("--version", type=str, required=True,
                        help="Version label (e.g. v9) — becomes second-level subfolder")
    parser.add_argument("--local-dir", type=str, required=True,
                        help="Local source directory (e.g. geminon_curation/output/v9)")
    parser.add_argument("--include", type=str, nargs="+", default=None,
                        help="Subpaths to upload (default: corpus, qa, indices). "
                             "Can be files or directories, relative to --local-dir.")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (defaults to HF_TOKEN env var)")
    parser.add_argument("--commit-message", type=str, default=None,
                        help="Commit message (default: 'Upload {curation}/{version}')")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private if it doesn't exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files that would be uploaded without uploading")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("Error: Provide --token or set HF_TOKEN env var")
        return

    local_dir = Path(args.local_dir).resolve()
    if not local_dir.exists():
        print(f"Error: --local-dir does not exist: {local_dir}")
        return

    includes = args.include or DEFAULT_INCLUDES
    print(f"Source: {local_dir}")
    print(f"Repo:   {args.repo}")
    print(f"Path in repo: {args.curation}/{args.version}/")
    print(f"Includes: {includes}")

    files = collect_files(local_dir, includes)
    if not files:
        print("No files found to upload")
        return

    total_size = sum(f.stat().st_size for f, _ in files)
    print(f"\n{len(files)} files, {total_size / 1e6:.1f} MB total:")
    for local_f, rel in files[:20]:
        size = local_f.stat().st_size
        print(f"  {rel:<60} {size/1e6:>8.2f} MB")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")

    if args.dry_run:
        print("\n[DRY RUN] No upload performed")
        return

    # Lazy import so the script can be examined / dry-run without the dep
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Ensure repo exists
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=token,
        )
    except Exception as e:
        print(f"  Note: create_repo: {e}")

    commit_msg = args.commit_message or f"Upload {args.curation}/{args.version}"
    print(f"\nUploading {len(files)} files to {args.repo} ({commit_msg})...")

    # Upload as a folder for atomicity. We need to mirror the source layout
    # under {curation}/{version}/ in the repo. The simplest way is per-file
    # uploads via upload_file with explicit path_in_repo.
    for i, (local_f, rel) in enumerate(files, 1):
        path_in_repo = f"{args.curation}/{args.version}/{rel}"
        try:
            api.upload_file(
                path_or_fileobj=str(local_f),
                path_in_repo=path_in_repo,
                repo_id=args.repo,
                repo_type="dataset",
                token=token,
                commit_message=commit_msg if i == 1 else None,
            )
            print(f"  [{i}/{len(files)}] {path_in_repo}")
        except Exception as e:
            print(f"  [{i}/{len(files)}] FAILED: {path_in_repo}: {e}")

    print(f"\nDone! View at https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
