"""Fetch HuggingFace's auto-generated Croissant metadata for a curation dataset,
merge in the curation-specific RAI fields, and (optionally) upload the result
back to HF as `croissant.json` so consumers see the augmented metadata.

The RAI text for each curation lives next to its pipeline:
    geminon_curation/rai_metadata.py    (exports GEMINON_RAI: dict)
    news_curation/rai_metadata.py       (exports NEWS_RAI: dict)

This script is intentionally thin: it loads the right dict based on
`--curation`, GETs the auto-Croissant from HF, sets the RAI namespace + spec
conformance, drops every key from the dict in (overwriting if present), saves
the merged JSON locally, and optionally uploads it back to HF.

Usage:
    # Dry-run — fetch + merge + save locally; no upload
    python -m tools.meta_data --curation geminon --version 2025_09

    # Custom output path
    python -m tools.meta_data --curation news --version v5 \\
        --output news_curation/output/v5/croissant.json

    # Upload the merged metadata to HF as `croissant.json` at the repo root
    python -m tools.meta_data --curation geminon --version 2025_09 --upload

    # Override the repo
    python -m tools.meta_data --curation geminon --version 2025_09 \\
        --repo pl666/ContinuousBench-Geminon
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import requests


# ───────────────────────────────────────────────────────────────────────────
# Defaults — match what tools/push_to_hf.py uses
# ───────────────────────────────────────────────────────────────────────────
DEFAULT_REPOS = {
    "geminon": "ContinuousBench/Geminon",
    "news":    "ContinuousBench/News",
}

# Where each curation's RAI dict + variable name lives
RAI_SOURCES = {
    "geminon": ("geminon_curation/rai_metadata.py", "GEMINON_RAI"),
    "news":    ("news_curation/rai_metadata.py",    "NEWS_RAI"),
}

# Where the merged JSON lands by default (next to the curation's output)
DEFAULT_OUTPUT_TEMPLATE = "{curation}_curation/output/{version}/croissant.json"


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────
def load_rai_dict(curation):
    """Import the RAI dict for the given curation by file path."""
    if curation not in RAI_SOURCES:
        raise ValueError(f"Unknown curation: {curation!r}. Valid: {list(RAI_SOURCES)}")
    rel_path, var_name = RAI_SOURCES[curation]
    project_root = Path(__file__).resolve().parent.parent
    module_path = project_root / rel_path
    if not module_path.exists():
        raise FileNotFoundError(f"RAI module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(f"_rai_{curation}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, var_name):
        raise AttributeError(f"{module_path} does not export `{var_name}`")
    return getattr(module, var_name)


def fetch_croissant(repo_id, token=None):
    """GET https://huggingface.co/api/datasets/{repo_id}/croissant"""
    url = f"https://huggingface.co/api/datasets/{repo_id}/croissant"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=60)
    if not r.ok:
        print(f"GET {url}\n  {r.status_code}: {r.text[:500]}", file=sys.stderr)
        r.raise_for_status()
    return r.json()


def merge_rai(croissant, rai_dict):
    """Merge RAI fields into the Croissant dict (in-place + return).

    Adds namespace declarations for any prefix used in the dict's keys
    (rai, dct, prov, ...) so the resulting JSON-LD is well-formed.
    """
    # JSON-LD `@context` can be either a dict (preferred) or a string URL
    # shorthand for `{"@vocab": "<url>"}`. HF's auto-Croissant for the news
    # repo returns a string, so promote it to a dict before adding prefixes.
    ctx = croissant.get("@context", {})
    if isinstance(ctx, str):
        ctx = {"@vocab": ctx}
    elif isinstance(ctx, list):
        # Some serializers use a list; pull strings into @vocab and merge dicts.
        merged = {}
        for item in ctx:
            if isinstance(item, str):
                merged.setdefault("@vocab", item)
            elif isinstance(item, dict):
                merged.update(item)
        ctx = merged
    croissant["@context"] = ctx

    # Always declare these — they're cited in dct:conformsTo and the standard RAI fields
    ctx["rai"] = "http://mlcommons.org/croissant/RAI/"
    ctx["dct"] = "http://purl.org/dc/terms/"

    # Declare any other known namespaces if their prefix appears as a key
    known_namespaces = {
        "prov":   "http://www.w3.org/ns/prov#",
        "schema": "https://schema.org/",
        "foaf":   "http://xmlns.com/foaf/0.1/",
        "dcat":   "http://www.w3.org/ns/dcat#",
    }
    used_prefixes = {k.split(":", 1)[0] for k in rai_dict if ":" in k}
    for prefix in used_prefixes:
        if prefix in known_namespaces and prefix not in ctx:
            ctx[prefix] = known_namespaces[prefix]

    croissant["dct:conformsTo"] = "http://mlcommons.org/croissant/RAI/1.0"
    for k, v in rai_dict.items():
        croissant[k] = v
    return croissant


def upload_croissant(repo_id, local_path, token=None, commit_message=None):
    """Upload the merged JSON to HF as croissant.json at the repo root."""
    from huggingface_hub import HfApi
    api = HfApi(token=token) if token else HfApi()
    msg = commit_message or "Upload Croissant metadata with RAI extension"
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo="croissant.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=msg,
    )
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/croissant.json"


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch HF Croissant + merge curation-specific RAI fields"
    )
    parser.add_argument("--curation", choices=list(RAI_SOURCES), required=True,
                        help="Which curation's RAI dict to merge")
    parser.add_argument("--version", required=True,
                        help="Version label (used only for the default output path)")
    parser.add_argument("--repo", default=None,
                        help="Override the HF repo id (default: ContinuousBench/{Geminon,News})")
    parser.add_argument("--output", default=None,
                        help=f"Output path for the merged JSON (default: {DEFAULT_OUTPUT_TEMPLATE})")
    parser.add_argument("--token", default=None,
                        help="HF token (default: $HF_TOKEN or ~/.cache/huggingface/token)")
    parser.add_argument("--upload", action="store_true",
                        help="After saving locally, also upload to HF as croissant.json")
    args = parser.parse_args()

    repo_id = args.repo or DEFAULT_REPOS[args.curation]
    output_path = Path(args.output) if args.output else Path(
        DEFAULT_OUTPUT_TEMPLATE.format(curation=args.curation, version=args.version)
    )
    token = args.token or os.environ.get("HF_TOKEN")

    # 1. Load the curation's RAI dict
    print(f"Loading RAI dict for curation={args.curation}")
    rai_dict = load_rai_dict(args.curation)
    print(f"  {len(rai_dict)} RAI fields: {sorted(rai_dict)}")

    # 2. Fetch HF's auto-Croissant
    print(f"\nFetching auto-generated Croissant from HF: {repo_id}")
    croissant = fetch_croissant(repo_id, token=token)
    print(f"  got {len(croissant)} top-level keys")

    # 3. Merge
    print("\nMerging RAI fields into Croissant")
    merged = merge_rai(croissant, rai_dict)

    # 4. Save locally
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    size = output_path.stat().st_size
    print(f"\nWrote {output_path}  ({size:,} bytes)")

    # 5. Optionally upload
    if args.upload:
        print(f"\nUploading {output_path} → {repo_id}/croissant.json")
        url = upload_croissant(repo_id, output_path, token=token)
        print(f"  {url}")
    else:
        print("\nNo --upload flag set; the merged JSON stays local.")
        print(f"To push it, re-run with --upload, or upload manually:")
        print(f"  hf upload {repo_id} {output_path} croissant.json --repo-type dataset")


if __name__ == "__main__":
    main()
