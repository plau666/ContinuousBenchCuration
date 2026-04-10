"""Generic I/O utilities shared across all curation pipelines."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_config(path, cli_overrides=None):
    """Load YAML config and merge with CLI overrides."""
    with open(path) as f:
        config = yaml.safe_load(f)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                # Support nested keys like "index.seed"
                keys = key.split(".")
                d = config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
    return config


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(data, path):
    """Save a list of dicts to JSONL with numpy type safety."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, cls=NumpyEncoder, ensure_ascii=False) + "\n")


def load_csv(path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def load_template(path):
    """Read and return the contents of a template file."""
    with open(path) as f:
        return f.read().strip()


def clean_and_parse_response(response_str):
    """Strip markdown fences and parse JSON response.

    Returns (parsed_list, error_string). On success error is None.
    """
    s = response_str.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    try:
        return json.loads(s), None
    except json.JSONDecodeError:
        pass
    # Fallback: find first '[' and last ']'
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1]), None
        except json.JSONDecodeError as e:
            return None, str(e)
    return None, "no valid JSON array found"
