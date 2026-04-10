"""Stage 12: Save closed-book judge prompts (one per cluster, batched).

Each prompt contains all of a cluster's QAs with their ground-truth answers
and zero-shot responses. The judge model returns a JSON array with
`is_zeroshot_correct` and `is_underspecified` per QA.

Inputs:  {output_dir}/{version}/qa/qas_with_zeroshot.jsonl
Outputs: {output_dir}/{version}/prompts/judge_prompts.jsonl

Usage:
    python 12_save_judge_prompts.py --config config.yaml
"""

import argparse
import json

from utils.io import load_config, load_jsonl, save_jsonl, load_template, ensure_output_dir


def format_qas_for_prompt(qas, zeroshot_field):
    items = []
    for i, qa in enumerate(qas, 1):
        zeroshot = qa.get(zeroshot_field) or "(no response)"
        items.append({
            "id": i,
            "question": qa.get("question"),
            "ground_truth": qa.get("answer"),
            "zeroshot_response": zeroshot,
        })
    return json.dumps(items, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Save judge prompts")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    zeroshot_field = config["zeroshot"]["model_label"]

    qas_path = output_dir / "qa" / "qas_with_zeroshot.jsonl"
    records = load_jsonl(str(qas_path))
    total_qas = sum(len(r.get("qas", [])) for r in records)
    print(f"Loaded {len(records)} clusters, {total_qas} QAs")

    template = load_template("templates/judge_closedbook.txt")

    prompts = []
    for prompt_idx, rec in enumerate(records):
        qas = rec.get("qas", [])
        if not qas:
            continue
        formatted = format_qas_for_prompt(qas, zeroshot_field)
        prompt = template.replace("{qas_json}", formatted)
        prompts.append({
            "idx": prompt_idx,
            "prompt": prompt,
            "tag": {
                "cluster_id": rec.get("cluster_id"),
                "n_qas": len(qas),
            },
        })

    output_path = output_dir / "prompts" / "judge_prompts.jsonl"
    save_jsonl(prompts, str(output_path))
    print(f"\nSaved {len(prompts)} judge prompts to {output_path}")
    print(f"\nNext: python -m tools.query_gemini --input {output_path} --output {output_dir}/responses/judge_responses.jsonl --api-keys ...")


if __name__ == "__main__":
    main()
