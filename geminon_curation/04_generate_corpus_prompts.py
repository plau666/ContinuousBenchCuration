"""Stage 4: Generate corpus prompt JSONL files (no LLM calls).

Creates 5 types of prompts from the named geminon index:
- wiki, journal, chain, comparison (public), sensitive_wiki

Usage:
    python 04_generate_corpus_prompts.py --config config.yaml
    python 04_generate_corpus_prompts.py --config config.yaml --prompt-types wiki,journal
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

from utils.io import load_config, load_jsonl, save_jsonl, load_template, ensure_output_dir


def geminon_to_json_str(g):
    """Convert geminon record to formatted JSON string for prompts."""
    return json.dumps(g, indent=2, ensure_ascii=False)


def group_by_evolution_line(geminons):
    """Group geminons into evolution lines (contiguous idx with same evolution_line)."""
    lines = []
    current_line = []
    for g in sorted(geminons, key=lambda x: x["idx"]):
        if current_line and g["idx"] != current_line[-1]["idx"] + 1:
            lines.append(current_line)
            current_line = []
        current_line.append(g)
    if current_line:
        lines.append(current_line)
    return lines


def generate_wiki_prompts(geminons, template, num_entries):
    """One prompt per geminon."""
    prompts = []
    for i, g in enumerate(geminons):
        prompt_text = template.replace("{data_json}", geminon_to_json_str(g))
        prompt_text = prompt_text.replace("{num_entries}", str(num_entries))
        prompts.append({
            "idx": i,
            "prompt": prompt_text,
            "tag": [g["idx"]],
        })
    return prompts


def generate_journal_prompts(geminons, template, num_entries):
    """One prompt per geminon."""
    prompts = []
    for i, g in enumerate(geminons):
        prompt_text = template.replace("{data_json}", geminon_to_json_str(g))
        prompt_text = prompt_text.replace("{num_entries}", str(num_entries))
        prompts.append({
            "idx": i,
            "prompt": prompt_text,
            "tag": [g["idx"]],
        })
    return prompts


def generate_chain_prompts(geminons, template, num_entries):
    """One prompt per evolution line with >=2 stages."""
    lines = group_by_evolution_line(geminons)
    prompts = []
    prompt_idx = 0
    for line in lines:
        if len(line) < 2:
            continue
        g1 = geminon_to_json_str(line[0])
        g2 = geminon_to_json_str(line[1]) if len(line) >= 2 else "null"
        g3 = geminon_to_json_str(line[2]) if len(line) >= 3 else "null"
        prompt_text = template.replace("{data_json_1}", g1)
        prompt_text = prompt_text.replace("{data_json_2}", g2)
        prompt_text = prompt_text.replace("{data_json_3}", g3)
        prompt_text = prompt_text.replace("{num_entries}", str(num_entries))
        prompts.append({
            "idx": prompt_idx,
            "prompt": prompt_text,
            "tag": [g["idx"] for g in line],
        })
        prompt_idx += 1
    return prompts


def generate_comparison_prompts(geminons, template, num_entries):
    """One prompt per pair of geminons (all C(N,2) combinations)."""
    prompts = []
    for prompt_idx, (g1, g2) in enumerate(combinations(geminons, 2)):
        prompt_text = template.replace("{data_json_1}", geminon_to_json_str(g1))
        prompt_text = prompt_text.replace("{data_json_2}", geminon_to_json_str(g2))
        prompt_text = prompt_text.replace("{num_entries}", str(num_entries))
        prompts.append({
            "idx": prompt_idx,
            "prompt": prompt_text,
            "tag": [g1["idx"], g2["idx"]],
        })
    return prompts


PROMPT_GENERATORS = {
    "wiki": ("wiki.txt", generate_wiki_prompts, "wiki_entries_per_prompt"),
    "journal": ("journal.txt", generate_journal_prompts, "journal_entries_per_prompt"),
    "chain": ("chain.txt", generate_chain_prompts, "chain_entries_per_prompt"),
    "comparison": ("comparison.txt", generate_comparison_prompts, "comparison_entries_per_prompt"),
    "sensitive_wiki": ("wiki.txt", generate_wiki_prompts, "wiki_entries_per_prompt"),
}


def main():
    parser = argparse.ArgumentParser(description="Generate corpus prompt JSONL files")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--prompt-types", type=str, default=None,
                        help="Comma-separated list of prompt types to generate (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    template_dir = Path(__file__).parent / "templates"
    prompt_cfg = config["prompts"]

    # Load indices
    public = load_jsonl(str(output_dir / "public_geminon_index.jsonl"))
    sensitive = load_jsonl(str(output_dir / "sensitive_geminon_index.jsonl"))
    print(f"Loaded: {len(public)} public, {len(sensitive)} sensitive geminons")

    # Determine which types to generate
    if args.prompt_types:
        types_to_gen = [t.strip() for t in args.prompt_types.split(",")]
    else:
        types_to_gen = list(PROMPT_GENERATORS.keys())

    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for ptype in types_to_gen:
        if ptype not in PROMPT_GENERATORS:
            print(f"Unknown prompt type: {ptype}, skipping")
            continue

        template_file, generator_fn, entries_key = PROMPT_GENERATORS[ptype]
        template = load_template(template_dir / template_file)
        num_entries = prompt_cfg[entries_key]

        # Use sensitive geminons for sensitive_wiki, public for everything else
        geminons = sensitive if ptype == "sensitive_wiki" else public

        prompts = generator_fn(geminons, template, num_entries)

        output_path = prompts_dir / f"{'public' if ptype != 'sensitive_wiki' else 'sensitive'}_{ptype.replace('sensitive_', '')}_prompts.jsonl"
        save_jsonl(prompts, str(output_path))
        print(f"  {ptype}: {len(prompts)} prompts -> {output_path}")

    print(f"\nDone! Prompts saved to {prompts_dir}/")
    print("Next: run `python -m tools.query_gemini --input ... --output ...` on each prompt file.")


if __name__ == "__main__":
    main()
