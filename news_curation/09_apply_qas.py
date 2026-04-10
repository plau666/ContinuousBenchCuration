"""Stage 9: Parse QA generation responses → qas.jsonl.

Reads {output_dir}/{version}/responses/qa_responses.jsonl, parses each
response as JSON {qas: [...]}, and writes one cluster record per response.

Output format:
    {
      "cluster_id": "...",
      "article_count": N,
      "articles_used": M,
      "facts_used": K,
      "qas": [
        {"question": "...", "answer": "...", "supporting_articles_ids": [...]},
        ...
      ]
    }

Usage:
    python 09_apply_qas.py --config config.yaml
"""

import argparse
import json
import re

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


def parse_qas(text):
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace_match:
        return []
    try:
        obj = json.loads(brace_match.group())
    except json.JSONDecodeError:
        return []
    qas = obj.get("qas", [])
    if not isinstance(qas, list):
        return []

    out = []
    for q in qas:
        if not isinstance(q, dict) or "question" not in q or "answer" not in q:
            continue
        # Canonicalize the field name: prefer root_articles_ids, accept legacy
        # supporting_articles_ids for backwards compatibility.
        if "root_articles_ids" not in q and "supporting_articles_ids" in q:
            q["root_articles_ids"] = q.pop("supporting_articles_ids")
        elif "supporting_articles_ids" in q and "root_articles_ids" in q:
            q.pop("supporting_articles_ids")
        out.append(q)
    return out


def main():
    parser = argparse.ArgumentParser(description="Apply QA generation responses")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--responses", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    responses_path = args.responses or str(output_dir / "responses" / "qa_responses.jsonl")
    print(f"Loading responses from {responses_path}")
    responses = load_jsonl(responses_path)
    print(f"  Loaded {len(responses)} responses")

    out_records = []
    n_empty = 0
    total_qas = 0
    for r in responses:
        tag = r.get("tag", {})
        text = r.get("response", "") or ""
        qas = parse_qas(text)
        if not qas:
            n_empty += 1
        total_qas += len(qas)
        out_records.append({
            "cluster_id": tag.get("cluster_id"),
            "article_count": tag.get("article_count"),
            "articles_used": tag.get("articles_used"),
            "facts_used": tag.get("facts_used"),
            "qas": qas,
        })

    output_path = output_dir / "qa" / "qas.jsonl"
    save_jsonl(out_records, str(output_path))
    print(f"\nSaved {len(out_records)} cluster records ({total_qas} total QAs) to {output_path}")
    print(f"  Empty/failed: {n_empty}")


if __name__ == "__main__":
    main()
