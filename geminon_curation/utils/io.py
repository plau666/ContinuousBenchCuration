"""I/O for geminon_curation.

Generic I/O lives in tools.io. This module re-exports those plus the
geminon-specific `ensure_output_dir` helper.
"""

from pathlib import Path

from tools.io import (
    NumpyEncoder,
    load_config,
    load_jsonl,
    save_jsonl,
    load_csv,
    load_template,
    clean_and_parse_response,
)

__all__ = [
    "NumpyEncoder",
    "load_config",
    "load_jsonl",
    "save_jsonl",
    "load_csv",
    "load_template",
    "clean_and_parse_response",
    "ensure_output_dir",
]


def ensure_output_dir(config):
    """Create versioned output directory and standard subdirs. Returns the base path."""
    base = Path(config["output_dir"]) / config["version"]
    for subdir in ["prompts", "responses", "corpus", "qa"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    return base
