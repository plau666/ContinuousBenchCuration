"""I/O for news_curation.

Re-exports generic I/O from tools.io and adds the news-specific
`ensure_output_dir` helper.
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
    for subdir in [
        "warcs",
        "extracted",
        "cleaned",
        "embeds",
        "clustered",
        "prompts",
        "responses",
        "qa",
        "support",
        "stats",
    ]:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    return base
