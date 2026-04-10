"""Stage 3: Apply LLM naming responses, deduplicate, and split public/sensitive.

Reads the unnamed index and naming responses, applies names, checks for
duplicates, and outputs the final named index files.

Usage:
    python 03_apply_names.py --config config.yaml --responses output/v9/naming_responses.jsonl
    python 03_apply_names.py --config config.yaml --responses output/v9/naming_responses.jsonl \
        --name-requery-responses output/v9/naming_requery_responses.jsonl
    python 03_apply_names.py --config config.yaml --responses output/v9/naming_responses.jsonl \
        --cls-requery-responses output/v9/classification_requery_responses.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from utils.io import (
    load_config, load_jsonl, save_jsonl, load_template,
    clean_and_parse_response, ensure_output_dir,
)


def parse_naming_response(response_str):
    """Parse a naming JSONL response into a list of {stage, name, classification}."""
    # Try parsing as JSONL (one JSON object per line)
    results = []
    lines = response_str.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip markdown fences
        if line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            if "name" in obj and "classification" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    if results:
        return results

    # Fallback: try parsing as JSON array
    parsed, err = clean_and_parse_response(response_str)
    if parsed and isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict) and "name" in r]

    return []


def format_line_data(line_records):
    """Format evolution line data for prompts."""
    parts = []
    for i, g in enumerate(line_records):
        type_str = g["type1"] + (f" / {g['type2']}" if g["type2"] else "")
        part = (
            f"Stage {i + 1}:\n"
            f"  Types: {type_str}\n"
            f"  Ability: {g['ability']}\n"
            f"  Move: {g['move']['name']}\n"
            f"  HP: {g['hp']}, Attack: {g['attack']}, Defense: {g['defense']}\n"
            f"  Special Attack: {g['special attack']}, Special Defense: {g['special defense']}, Speed: {g['speed']}\n"
            f"  Base Stat Total: {g['base_stat_total']}\n"
            f"  Weight: {g['weight']} lbs, Height: {g['height']} m"
        )
        parts.append(part)
    return "\n\n".join(parts)


def format_line_with_names(line_records):
    """Format evolution line with current names for classification re-query."""
    parts = []
    for i, g in enumerate(line_records):
        type_str = g["type1"] + (f" / {g['type2']}" if g["type2"] else "")
        parts.append(
            f"Stage {i + 1}: Name={g['name']}\n"
            f"  Types: {type_str}\n"
            f"  Ability: {g['ability']}\n"
            f"  Move: {g['move']['name']}"
        )
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Apply naming responses and split")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--responses", type=str, required=True,
                        help="Path to naming_responses.jsonl from tools.query_gemini")
    parser.add_argument("--name-requery-responses", type=str, default=None,
                        help="Path to naming_requery_responses.jsonl (if re-querying names)")
    parser.add_argument("--cls-requery-responses", type=str, default=None,
                        help="Path to classification_requery_responses.jsonl (if re-querying)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    template_dir = Path(__file__).parent / "templates"

    # Load unnamed index
    index_path = output_dir / "geminon_index_unnamed.jsonl"
    geminons = load_jsonl(str(index_path))
    idx_to_geminon = {g["idx"]: g for g in geminons}

    # Load naming prompts to get line groupings
    naming_prompts = load_jsonl(str(output_dir / "prompts" / "naming_prompts.jsonl"))
    prompt_idx_to_prompt = {p["idx"]: p for p in naming_prompts}

    # Load responses
    responses = load_jsonl(args.responses)
    resp_by_idx = {r["idx"]: r for r in responses}

    # Load requery responses if provided
    name_requery_resps = {}
    if args.name_requery_responses:
        for r in load_jsonl(args.name_requery_responses):
            name_requery_resps[r["idx"]] = r

    cls_requery_resps = {}
    if args.cls_requery_responses:
        for r in load_jsonl(args.cls_requery_responses):
            cls_requery_resps[r["idx"]] = r

    # --- Apply names from responses ---
    lines_by_idx = {}  # prompt_idx -> list of geminon records
    parse_errors = 0

    for prompt in naming_prompts:
        pidx = prompt["idx"]
        tag = prompt["tag"]
        line_records = [idx_to_geminon[gi] for gi in tag]

        # Use requery response if available, else original
        if pidx in name_requery_resps:
            resp = name_requery_resps[pidx]
        elif pidx in cls_requery_resps:
            resp = cls_requery_resps[pidx]
        elif pidx in resp_by_idx:
            resp = resp_by_idx[pidx]
        else:
            print(f"  Warning: No response for prompt idx {pidx}")
            parse_errors += 1
            lines_by_idx[pidx] = line_records
            continue

        # Parse response
        naming_results = parse_naming_response(resp["response"])
        if len(naming_results) < len(line_records):
            print(f"  Warning: prompt {pidx} expected {len(line_records)} names, got {len(naming_results)}")
            parse_errors += 1

        # Apply names
        for i, g in enumerate(line_records):
            if i < len(naming_results):
                g["name"] = naming_results[i].get("name")
                g["classification"] = naming_results[i].get("classification")

        # Fill evolution_line with actual names
        names = [g["name"] for g in line_records]
        for g in line_records:
            g["evolution_line"] = names

        lines_by_idx[pidx] = line_records

    print(f"Applied names to {len(lines_by_idx)} lines ({parse_errors} parse errors)")

    # --- Check for duplicate names ---
    name_to_pidxs = defaultdict(list)
    for pidx, records in lines_by_idx.items():
        for r in records:
            if r["name"]:
                name_to_pidxs[r["name"]].append(pidx)

    dup_names = {n: sorted(set(pids)) for n, pids in name_to_pidxs.items() if len(set(pids)) > 1}

    if dup_names and not args.name_requery_responses:
        print(f"\n  Found {len(dup_names)} duplicate names across lines!")
        # Determine which lines to re-query (keep first, re-query rest)
        lines_to_requery = set()
        for name, pidxs in dup_names.items():
            for pidx in pidxs[1:]:
                lines_to_requery.add(pidx)

        # Build forbidden name set
        used_names = set()
        for pidx, records in lines_by_idx.items():
            if pidx not in lines_to_requery:
                for r in records:
                    if r["name"]:
                        used_names.add(r["name"])

        # Generate requery prompts
        requery_template = load_template(template_dir / "naming_requery.txt")
        requery_prompts = []
        for pidx in sorted(lines_to_requery):
            records = lines_by_idx[pidx]
            line_data = format_line_data(records)
            prompt_text = requery_template.replace(
                "{forbidden_names}", "\n".join(sorted(used_names))
            ).replace("{evolution_line_data}", line_data)
            requery_prompts.append({
                "idx": pidx,
                "prompt": prompt_text,
                "tag": [g["idx"] for g in records],
            })

        requery_path = output_dir / "prompts" / "naming_requery_prompts.jsonl"
        save_jsonl(requery_prompts, str(requery_path))
        requery_resp_path = output_dir / "responses" / "naming_requery_responses.jsonl"
        print(f"  Saved {len(requery_prompts)} re-query prompts to {requery_path}")
        print(f"  Run: python -m tools.query_gemini --input {requery_path} --output {requery_resp_path} --api-keys ...")
        print(f"  Then: python 03_apply_names.py --config {args.config} --responses {args.responses} --name-requery-responses {requery_resp_path}")
        return

    # --- Check for duplicate classification tuples ---
    cls_tuple_to_pidxs = defaultdict(list)
    for pidx, records in lines_by_idx.items():
        if len(records) >= 2:
            cls_tuple = tuple(r["classification"] for r in records)
            cls_tuple_to_pidxs[cls_tuple].append(pidx)

    dup_cls = {t: pids for t, pids in cls_tuple_to_pidxs.items() if len(pids) > 1}

    if dup_cls and not args.cls_requery_responses:
        print(f"\n  Found {len(dup_cls)} duplicate classification tuples!")
        lines_to_requery_cls = set()
        for cls_tuple, pidxs in dup_cls.items():
            for pidx in sorted(pidxs)[1:]:
                lines_to_requery_cls.add(pidx)

        used_tuples = set()
        for pidx, records in lines_by_idx.items():
            if len(records) >= 2 and pidx not in lines_to_requery_cls:
                used_tuples.add(tuple(r["classification"] for r in records))

        requery_template = load_template(template_dir / "classification_requery.txt")
        requery_prompts = []
        for pidx in sorted(lines_to_requery_cls):
            records = lines_by_idx[pidx]
            line_data = format_line_with_names(records)
            forbidden_str = "\n".join(str(t) for t in sorted(used_tuples))
            prompt_text = requery_template.replace(
                "{forbidden_tuples}", forbidden_str
            ).replace("{evolution_line_data_with_names}", line_data)
            requery_prompts.append({
                "idx": pidx,
                "prompt": prompt_text,
                "tag": [g["idx"] for g in records],
            })

        requery_path = output_dir / "prompts" / "classification_requery_prompts.jsonl"
        save_jsonl(requery_prompts, str(requery_path))
        requery_resp_path = output_dir / "responses" / "classification_requery_responses.jsonl"
        print(f"  Saved {len(requery_prompts)} classification re-query prompts to {requery_path}")
        print(f"  Run: python -m tools.query_gemini --input {requery_path} --output {requery_resp_path} --api-keys ...")
        print(f"  Then: python 03_apply_names.py --config {args.config} --responses {args.responses} --cls-requery-responses {requery_resp_path}")
        return

    # --- No duplicates: proceed to split ---
    print("\nNo duplicate names or classifications found. Proceeding to split.")

    # Rebuild line lists by stage count
    final_3stage, final_2stage, final_1stage = [], [], []
    for pidx, records in lines_by_idx.items():
        if len(records) == 3:
            final_3stage.append(records)
        elif len(records) == 2:
            final_2stage.append(records)
        else:
            final_1stage.append(records)

    # Sensitive/public split
    split_cfg = config["split"]
    random.seed(config["sensitive_split_seed"])

    sensitive_3 = set(random.sample(range(len(final_3stage)), split_cfg["num_sensitive_lines_3stage"]))
    sensitive_2 = set(random.sample(range(len(final_2stage)), split_cfg["num_sensitive_lines_2stage"]))
    sensitive_1 = set(random.sample(range(len(final_1stage)), split_cfg["num_sensitive_lines_1stage"]))

    sensitive_geminons, public_geminons = [], []

    for i, records in enumerate(final_3stage):
        (sensitive_geminons if i in sensitive_3 else public_geminons).extend(records)
    for i, records in enumerate(final_2stage):
        (sensitive_geminons if i in sensitive_2 else public_geminons).extend(records)
    for i, records in enumerate(final_1stage):
        (sensitive_geminons if i in sensitive_1 else public_geminons).extend(records)

    sensitive_geminons.sort(key=lambda g: g["idx"])
    public_geminons.sort(key=lambda g: g["idx"])
    all_geminons = sorted(sensitive_geminons + public_geminons, key=lambda g: g["idx"])

    # Save
    save_jsonl(all_geminons, str(output_dir / "geminon_index.jsonl"))
    save_jsonl(public_geminons, str(output_dir / "public_geminon_index.jsonl"))
    save_jsonl(sensitive_geminons, str(output_dir / "sensitive_geminon_index.jsonl"))

    print(f"\nSaved:")
    print(f"  geminon_index.jsonl          ({len(all_geminons)} records)")
    print(f"  public_geminon_index.jsonl   ({len(public_geminons)} records)")
    print(f"  sensitive_geminon_index.jsonl ({len(sensitive_geminons)} records)")


if __name__ == "__main__":
    main()
