"""Stage 10: Save zero-shot evaluation prompts (one per QA).

For each QA in qas.jsonl, build a prompt asking the model to answer the
question with no context (best guess). The prompt's `tag` field carries the
(cluster_id, qa_idx) so the responses can be merged back in stage 11.

Inputs:  {output_dir}/{version}/qa/qas.jsonl
Outputs: {output_dir}/{version}/prompts/zeroshot_prompts.jsonl

Usage:
    python 10_save_zeroshot_prompts.py --config config.yaml
"""

import argparse

from utils.io import load_config, load_jsonl, save_jsonl, load_template, ensure_output_dir


def main():
    parser = argparse.ArgumentParser(description="Save zero-shot prompts")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    qas_path = output_dir / "qa" / "qas.jsonl"
    print(f"Loading {qas_path}")
    records = load_jsonl(str(qas_path))
    total_qas = sum(len(r.get("qas", [])) for r in records)
    print(f"  {len(records)} clusters, {total_qas} total QAs")

    template = load_template("templates/zeroshot.txt")

    prompts = []
    prompt_idx = 0
    for r in records:
        cid = r.get("cluster_id")
        for qi, qa in enumerate(r.get("qas", [])):
            question = qa.get("question") or ""
            if not question:
                continue
            prompt = template.replace("{question}", question)
            prompts.append({
                "idx": prompt_idx,
                "prompt": prompt,
                "tag": {"cluster_id": cid, "qa_idx": qi},
            })
            prompt_idx += 1

    output_path = output_dir / "prompts" / "zeroshot_prompts.jsonl"
    save_jsonl(prompts, str(output_path))
    print(f"\nSaved {len(prompts)} zero-shot prompts to {output_path}")
    print(f"\nNext: python -m tools.query_gemini --input {output_path} --output {output_dir}/responses/zeroshot_responses.jsonl --api-keys ...")


if __name__ == "__main__":
    main()
