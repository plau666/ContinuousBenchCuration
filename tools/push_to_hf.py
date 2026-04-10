"""Push curation outputs to HuggingFace as proper datasets with configs + tags.

Two upload targets, both private by default:
  - geminon → pl666/ContinuousBench-Geminon
  - news    → pl666/ContinuousBench-News

After upload, the user can do:

    # Geminon corpus (3 sizes × 4 splits)
    load_dataset("pl666/ContinuousBench-Geminon", "corpus_large",
                 split="train", revision="v9")

    # Geminon QA (2 sizes × 4 splits)
    load_dataset("pl666/ContinuousBench-Geminon", "qa_small",
                 split="public_val", revision="v9")

    # News corpus (3 sizes × 4 splits)
    load_dataset("pl666/ContinuousBench-News", "corpus_large",
                 split="train", revision="v5")

    # News QA (default config — no config_name needed)
    load_dataset("pl666/ContinuousBench-News",
                 split="val", revision="v5")

The script:
  1. Creates the repo (private if not yet existing)
  2. Resolves all source files (following symlinks)
  3. Builds + uploads a README.md whose YAML frontmatter declares every config
  4. Uploads each data file to its repo path
  5. Tags the resulting commit with the version label

Usage:
    export HF_TOKEN=hf_xxx

    # Push geminon v9 (everything)
    python -m tools.push_to_hf --curation geminon --version v9

    # Push news v5 (everything)
    python -m tools.push_to_hf --curation news --version v5

    # Subset / dry run
    python -m tools.push_to_hf --curation geminon --version v9 --skip-qa
    python -m tools.push_to_hf --curation news --version v5 --dry-run

    # Override the local source directory
    python -m tools.push_to_hf --curation geminon --version v9 \
        --local-dir /path/to/geminon_curation/output/v9
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────
# Default repo names + version → revision mapping
# ───────────────────────────────────────────────────────────────────────────
DEFAULT_REPOS = {
    "geminon": "pl666/ContinuousBench-Geminon",
    "news":    "pl666/ContinuousBench-News",
}

DEFAULT_LOCAL_ROOT = {
    "geminon": "geminon_curation/output",
    "news":    "news_curation/output",
}


# ───────────────────────────────────────────────────────────────────────────
# Spec types
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class ConfigSpec:
    """A single dataset config (e.g. corpus_large) with its split files."""
    name: str
    splits: dict           # {split_name: local_Path}
    is_default: bool = False


@dataclass
class UploadSpec:
    """The full upload plan for one repo."""
    repo: str
    version: str
    local_dir: Path
    private: bool
    configs: list = field(default_factory=list)


# ───────────────────────────────────────────────────────────────────────────
# Build the upload spec for each curation
# ───────────────────────────────────────────────────────────────────────────
def build_geminon_spec(version, local_dir, repo, private,
                      include_corpus=True, include_qa=True):
    """Geminon: 3 corpus configs × 4 splits + 2 qa configs × 4 splits."""
    spec = UploadSpec(repo=repo, version=version, local_dir=local_dir, private=private)
    corpus_dir = local_dir / "corpus"
    qa_dir = local_dir / "qa"

    if include_corpus:
        for slice_name in ["large", "medium", "small"]:
            slice_dir = corpus_dir / slice_name
            spec.configs.append(ConfigSpec(
                name=f"corpus_{slice_name}",
                splits={
                    "train": slice_dir / "train.jsonl",
                    "val":   slice_dir / "val.jsonl",
                    "test":  slice_dir / "test.jsonl",
                    "all":   slice_dir / "all.jsonl",
                },
            ))

    if include_qa:
        for slice_name in ["small", "medium"]:
            slice_dir = qa_dir / slice_name
            spec.configs.append(ConfigSpec(
                name=f"qa_{slice_name}",
                splits={
                    "public_val":     slice_dir / "public_val.jsonl",
                    "public_test":    slice_dir / "public_test.jsonl",
                    "sensitive_val":  slice_dir / "sensitive_val.jsonl",
                    "sensitive_test": slice_dir / "sensitive_test.jsonl",
                },
            ))

    return spec


def build_news_spec(version, local_dir, repo, private,
                   include_corpus=True, include_qa=True):
    """News: 3 corpus configs × 4 splits + 1 default qa config × 2 splits."""
    spec = UploadSpec(repo=repo, version=version, local_dir=local_dir, private=private)
    corpus_dir = local_dir / "corpus"
    qa_dir = local_dir / "qa" / "final" / "filtered"

    if include_qa:
        # Marked as default so users can call load_dataset(repo, split=...)
        # without specifying a config_name. The config is still NAMED "qa"
        # so the folder in the repo is qa/ and an explicit
        # load_dataset(repo, "qa", split=...) call works too.
        spec.configs.append(ConfigSpec(
            name="qa",
            splits={
                "val":  qa_dir / "val.jsonl",
                "test": qa_dir / "test.jsonl",
            },
            is_default=True,
        ))

    if include_corpus:
        for slice_name in ["large", "medium", "small"]:
            slice_dir = corpus_dir / slice_name
            spec.configs.append(ConfigSpec(
                name=f"corpus_{slice_name}",
                splits={
                    "train": slice_dir / "train.jsonl",
                    "val":   slice_dir / "val.jsonl",
                    "test":  slice_dir / "test.jsonl",
                    "all":   slice_dir / "all.jsonl",
                },
            ))

    return spec


# ───────────────────────────────────────────────────────────────────────────
# README + YAML config generation
# ───────────────────────────────────────────────────────────────────────────
def render_readme(spec, curation_label):
    """Build a README.md whose YAML frontmatter declares every config + split.

    HuggingFace's automatic data file detection picks up the `configs` block
    in the YAML frontmatter so `load_dataset(repo, config_name, split=...)`
    just works.
    """
    lines = ["---"]
    lines.append("configs:")
    for cfg in spec.configs:
        lines.append(f"- config_name: {cfg.name}")
        if cfg.is_default:
            lines.append("  default: true")
        lines.append("  data_files:")
        for split_name, local_path in cfg.splits.items():
            # Path inside the repo: {config_name}/{split_name}.jsonl
            repo_path = f"{cfg.name}/{split_name}.jsonl"
            lines.append(f"  - split: {split_name}")
            lines.append(f"    path: {repo_path}")
    lines.append("license: apache-2.0")
    lines.append("tags:")
    lines.append("- continuousbench")
    lines.append(f"- {curation_label}")
    lines.append("---")
    lines.append("")
    lines.append(f"# ContinuousBench — {curation_label.title()} ({spec.version})")
    lines.append("")
    lines.append("This dataset was generated by the [ContinuousBenchCuration](https://github.com/) pipeline.")
    lines.append("")
    lines.append("## Configs")
    lines.append("")
    for cfg in spec.configs:
        marker = "  *(default)*" if cfg.is_default else ""
        lines.append(f"- **`{cfg.name}`**{marker} — splits: " + ", ".join(f"`{s}`" for s in cfg.splits))
    lines.append("")
    lines.append("## Loading")
    lines.append("")
    lines.append("```python")
    lines.append("from datasets import load_dataset")
    lines.append("")
    sample_cfg = next((c for c in spec.configs if not c.is_default), spec.configs[0])
    sample_split = next(iter(sample_cfg.splits))
    if sample_cfg.is_default:
        lines.append(f'ds = load_dataset("{spec.repo}", split="{sample_split}", revision="{spec.version}")')
    else:
        lines.append(
            f'ds = load_dataset("{spec.repo}", "{sample_cfg.name}", '
            f'split="{sample_split}", revision="{spec.version}")'
        )
    lines.append("```")
    lines.append("")
    lines.append(f"## Version: `{spec.version}`")
    lines.append("")
    lines.append(f"Pinned via the `{spec.version}` git tag — pass `revision=\"{spec.version}\"` to `load_dataset`.")
    lines.append("")
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────────────
# Upload
# ───────────────────────────────────────────────────────────────────────────
def collect_upload_pairs(spec):
    """Return list of (resolved_local_path, repo_path, size_bytes)."""
    pairs = []
    for cfg in spec.configs:
        for split_name, local_path in cfg.splits.items():
            local = Path(local_path).resolve()  # follows symlinks
            if not local.exists():
                print(f"  WARNING: missing {local} for {cfg.name}/{split_name}", file=sys.stderr)
                continue
            repo_path = f"{cfg.name}/{split_name}.jsonl"
            size = local.stat().st_size
            pairs.append((local, repo_path, size))
    return pairs


def do_upload(spec, curation_label, token, dry_run, skip_tag):
    pairs = collect_upload_pairs(spec)
    total_size = sum(s for _, _, s in pairs)

    print(f"\nRepo:    {spec.repo}")
    print(f"Version: {spec.version}  (will be tagged after upload)")
    print(f"Source:  {spec.local_dir}")
    print(f"Configs: {len(spec.configs)}")
    for cfg in spec.configs:
        marker = "  (default)" if cfg.is_default else ""
        print(f"  - {cfg.name}{marker}: {len(cfg.splits)} splits")
    print(f"Files:   {len(pairs)}, total {total_size / 1e9:.2f} GB")

    print("\nUpload plan:")
    for local, repo_path, size in pairs:
        size_str = f"{size / 1e6:.1f} MB" if size < 1e9 else f"{size / 1e9:.2f} GB"
        print(f"  {repo_path:<40} {size_str:>10}  ←  {local}")

    if dry_run:
        print("\n[DRY RUN] No upload performed.")
        return

    if not token:
        print("\nERROR: --token or HF_TOKEN required for live upload.")
        sys.exit(1)

    # Lazy import so dry-run doesn't need the dep
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # 1. Ensure repo exists
    print(f"\nEnsuring repo {spec.repo} exists (private={spec.private})")
    create_repo(
        repo_id=spec.repo,
        repo_type="dataset",
        private=spec.private,
        exist_ok=True,
        token=token,
    )

    # 2. Upload README first (so YAML configs are visible if a user browses
    #    the repo while uploads are in progress)
    readme = render_readme(spec, curation_label)
    print(f"Uploading README.md ({len(readme)} chars)")
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=spec.repo,
        repo_type="dataset",
        token=token,
        commit_message=f"Update README for {spec.version}",
    )

    # 3. Upload each data file
    print(f"\nUploading {len(pairs)} data files")
    for i, (local, repo_path, size) in enumerate(pairs, 1):
        size_str = f"{size / 1e6:.1f} MB" if size < 1e9 else f"{size / 1e9:.2f} GB"
        print(f"  [{i}/{len(pairs)}] {repo_path:<40} ({size_str})")
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_path,
                repo_id=spec.repo,
                repo_type="dataset",
                token=token,
                commit_message=f"Add {repo_path} ({spec.version})",
            )
        except Exception as e:
            print(f"      FAILED: {e}", file=sys.stderr)

    # 4. Tag the resulting main commit with the version
    if not skip_tag:
        print(f"\nTagging main as {spec.version}")
        try:
            api.create_tag(
                repo_id=spec.repo,
                tag=spec.version,
                tag_message=f"Release {spec.version}",
                repo_type="dataset",
                token=token,
                exist_ok=True,
            )
        except TypeError:
            # Older huggingface_hub versions don't support exist_ok
            try:
                api.create_tag(
                    repo_id=spec.repo,
                    tag=spec.version,
                    tag_message=f"Release {spec.version}",
                    repo_type="dataset",
                    token=token,
                )
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"  tag {spec.version} already exists; deleting and recreating")
                    api.delete_tag(spec.repo, tag=spec.version, repo_type="dataset", token=token)
                    api.create_tag(
                        repo_id=spec.repo,
                        tag=spec.version,
                        tag_message=f"Release {spec.version}",
                        repo_type="dataset",
                        token=token,
                    )
                else:
                    raise

    print(f"\nDone! View at https://huggingface.co/datasets/{spec.repo}/tree/{spec.version}")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Push ContinuousBench curation outputs to HuggingFace")
    parser.add_argument("--curation", type=str, required=True, choices=["geminon", "news"],
                        help="Which curation to push")
    parser.add_argument("--version", type=str, required=True,
                        help="Version label, e.g. v9 (Geminon) or v5 (News). Becomes the git tag.")
    parser.add_argument("--repo", type=str, default=None,
                        help="Override repo id (default: pl666/ContinuousBench-{Geminon,News})")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="Override local source directory (default: {geminon,news}_curation/output/{version})")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--public", action="store_true",
                        help="Push as public (default: private)")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="Skip corpus configs")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip QA configs")
    parser.add_argument("--skip-tag", action="store_true",
                        help="Don't create the version tag after upload")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the upload plan without uploading anything")
    args = parser.parse_args()

    repo = args.repo or DEFAULT_REPOS[args.curation]
    local_dir = Path(args.local_dir) if args.local_dir else (
        Path(DEFAULT_LOCAL_ROOT[args.curation]) / args.version
    )
    local_dir = local_dir.resolve()
    if not local_dir.exists():
        print(f"ERROR: local directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    private = not args.public
    token = args.token or os.environ.get("HF_TOKEN")

    if args.curation == "geminon":
        spec = build_geminon_spec(
            version=args.version,
            local_dir=local_dir,
            repo=repo,
            private=private,
            include_corpus=not args.skip_corpus,
            include_qa=not args.skip_qa,
        )
    else:
        spec = build_news_spec(
            version=args.version,
            local_dir=local_dir,
            repo=repo,
            private=private,
            include_corpus=not args.skip_corpus,
            include_qa=not args.skip_qa,
        )

    do_upload(spec, args.curation, token=token, dry_run=args.dry_run, skip_tag=args.skip_tag)


if __name__ == "__main__":
    main()
