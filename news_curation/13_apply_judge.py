"""Stage 13: Merge judge responses into qas → qas_judged.jsonl.

Reads judge_responses.jsonl, parses each response as a JSON array of
{id, is_zeroshot_correct, is_underspecified}, and attaches the judgments
back to the corresponding cluster's QAs by id (1-indexed within cluster).

Inputs:  {output_dir}/{version}/qa/qas_with_zeroshot.jsonl
         {output_dir}/{version}/responses/judge_responses.jsonl
Outputs: {output_dir}/{version}/qa/qas_judged.jsonl

Usage:
    python 13_apply_judge.py --config config.yaml
"""

import argparse
import json
import re

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


def parse_judgments(text):
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            results = json.loads(bracket_match.group())
            if isinstance(results, list):
                return [r for r in results if isinstance(r, dict) and "id" in r]
        except json.JSONDecodeError:
            pass
    return []


def main():
    parser = argparse.ArgumentParser(description="Apply judge responses")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--responses", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    qas_path = output_dir / "qa" / "qas_with_zeroshot.jsonl"
    records = load_jsonl(str(qas_path))

    responses_path = args.responses or str(output_dir / "responses" / "judge_responses.jsonl")
    responses = load_jsonl(responses_path)
    judgments_by_cid = {}
    for r in responses:
        cid = str(r.get("tag", {}).get("cluster_id"))
        judgments_by_cid[cid] = parse_judgments(r.get("response", "") or "")
    print(f"Loaded {len(records)} clusters, {len(judgments_by_cid)} judge responses")

    n_judged = n_missing = 0
    for rec in records:
        cid = str(rec.get("cluster_id"))
        qas = rec.get("qas", [])
        judgments = judgments_by_cid.get(cid, [])
        by_id = {int(j["id"]): j for j in judgments if "id" in j}
        for i, qa in enumerate(qas):
            j = by_id.get(i + 1)
            if j is not None:
                qa["is_zeroshot_correct"] = j.get("is_zeroshot_correct")
                qa["is_underspecified"] = j.get("is_underspecified")
                n_judged += 1
            else:
                qa["is_zeroshot_correct"] = None
                qa["is_underspecified"] = None
                n_missing += 1

    output_path = output_dir / "qa" / "qas_judged.jsonl"
    save_jsonl(records, str(output_path))
    print(f"\nSaved {output_path}")
    print(f"  judged={n_judged}, missing={n_missing}")


if __name__ == "__main__":
    main()
