"""Stage 11: Merge zero-shot responses into qas.jsonl.

Reads zeroshot_responses.jsonl, indexes by (cluster_id, qa_idx), then walks
qas.jsonl and attaches the response under the configured field name.

Inputs:  {output_dir}/{version}/qa/qas.jsonl
         {output_dir}/{version}/responses/zeroshot_responses.jsonl
Outputs: {output_dir}/{version}/qa/qas_with_zeroshot.jsonl

Usage:
    python 11_apply_zeroshot.py --config config.yaml
"""

import argparse

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


def main():
    parser = argparse.ArgumentParser(description="Apply zero-shot responses")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--responses", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    field = config["zeroshot"]["model_label"]

    qas_path = output_dir / "qa" / "qas.jsonl"
    records = load_jsonl(str(qas_path))

    responses_path = args.responses or str(output_dir / "responses" / "zeroshot_responses.jsonl")
    responses = load_jsonl(responses_path)
    print(f"Loaded {len(records)} cluster records, {len(responses)} zero-shot responses")

    by_key = {}
    for r in responses:
        tag = r.get("tag", {})
        key = (str(tag.get("cluster_id")), tag.get("qa_idx"))
        text = (r.get("response") or "").strip()
        by_key[key] = text

    n_filled = n_missing = 0
    for rec in records:
        cid = str(rec.get("cluster_id"))
        for qi, qa in enumerate(rec.get("qas", [])):
            key = (cid, qi)
            if key in by_key:
                qa[field] = by_key[key]
                n_filled += 1
            else:
                qa[field] = None
                n_missing += 1

    output_path = output_dir / "qa" / "qas_with_zeroshot.jsonl"
    save_jsonl(records, str(output_path))
    print(f"\nSaved {output_path}")
    print(f"  filled={n_filled}, missing={n_missing}")


if __name__ == "__main__":
    main()
