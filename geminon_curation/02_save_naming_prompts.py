"""Stage 2: Save naming prompts as JSONL (no LLM calls).

Reads the unnamed geminon index, groups by evolution line, and saves
one prompt per line using the naming template.

Usage:
    python 02_save_naming_prompts.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

from utils.io import load_config, load_jsonl, save_jsonl, load_template, ensure_output_dir


def format_line_data(line_records):
    """Format an evolution line's data for the naming prompt."""
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


def group_by_evolution_line(geminons):
    """Group geminon records by evolution line (adjacent indices)."""
    lines = []
    current_line = []

    for g in sorted(geminons, key=lambda x: x["idx"]):
        if current_line and g["idx"] != current_line[-1]["idx"] + 1:
            lines.append(current_line)
            current_line = []
        current_line.append(g)

    if current_line:
        lines.append(current_line)

    # Filter: only keep lines where all members share the same evolution_line length
    # (i.e., they're actually part of the same line)
    filtered = []
    for line in lines:
        line_len = len(line[0]["evolution_line"])
        if all(len(g["evolution_line"]) == line_len for g in line) and len(line) == line_len:
            filtered.append(line)
        else:
            # Split into individual lines based on evolution_line length
            by_len = {}
            for g in line:
                l = len(g["evolution_line"])
                by_len.setdefault(l, []).append(g)
            for records in by_len.values():
                # Sub-group by contiguous idx
                sub_line = []
                for g in sorted(records, key=lambda x: x["idx"]):
                    if sub_line and g["idx"] != sub_line[-1]["idx"] + 1:
                        if len(sub_line) == len(sub_line[0]["evolution_line"]):
                            filtered.append(sub_line)
                        sub_line = []
                    sub_line.append(g)
                if sub_line and len(sub_line) == len(sub_line[0]["evolution_line"]):
                    filtered.append(sub_line)

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Save naming prompts as JSONL")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    # Load unnamed index
    index_path = output_dir / "geminon_index_unnamed.jsonl"
    geminons = load_jsonl(str(index_path))
    print(f"Loaded {len(geminons)} geminons from {index_path}")

    # Group by evolution line
    lines = group_by_evolution_line(geminons)
    print(f"Found {len(lines)} evolution lines")

    # Load template
    template_dir = Path(__file__).parent / "templates"
    naming_template = load_template(template_dir / "naming.txt")

    # Generate prompts
    prompts = []
    for line_idx, line_records in enumerate(lines):
        line_data = format_line_data(line_records)
        prompt_text = naming_template.replace("{evolution_line_data}", line_data)

        prompts.append({
            "idx": line_idx,
            "prompt": prompt_text,
            "line_key": f"{len(line_records)}stage_{line_idx}",
            "line_length": len(line_records),
            "tag": [g["idx"] for g in line_records],
        })

    # Save
    output_path = output_dir / "prompts" / "naming_prompts.jsonl"
    save_jsonl(prompts, str(output_path))

    n3 = sum(1 for p in prompts if p["line_length"] == 3)
    n2 = sum(1 for p in prompts if p["line_length"] == 2)
    n1 = sum(1 for p in prompts if p["line_length"] == 1)
    print(f"\nSaved {len(prompts)} naming prompts to {output_path}")
    print(f"  3-stage: {n3}, 2-stage: {n2}, 1-stage: {n1}")
    print(f"\nNext: run `python -m tools.query_gemini --input {output_path} ...` to get naming responses.")


if __name__ == "__main__":
    main()
