"""Stage 7: Parse fact extraction responses → facts.jsonl.

Reads {output_dir}/{version}/responses/fact_responses.jsonl (produced by
tools.query_gemini run on fact_prompts.jsonl), parses each response into
a list of fact strings, and writes one record per cluster.

Output format:
    {"cluster_id": "...", "article_count": N, "articles_used": M, "facts": [...]}

Usage:
    python 07_apply_facts.py --config config.yaml
"""

import argparse
import json
import re

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


def parse_facts(text):
    """Parse model response into list of fact strings.

    Tries JSON array first, falls back to numbered/bulleted text.
    """
    if not text:
        return []
    text = text.strip()

    # Try JSON list
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            facts = json.loads(bracket_match.group())
            if isinstance(facts, list) and all(isinstance(f, str) for f in facts):
                return [f.strip() for f in facts if f.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: line-by-line, strip numbering / bullets
    facts = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        if len(line) > 20:
            facts.append(line)
    return facts


def main():
    parser = argparse.ArgumentParser(description="Apply fact extraction responses")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--responses", type=str, default=None,
                        help="Override path to fact_responses.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    responses_path = args.responses or str(output_dir / "responses" / "fact_responses.jsonl")
    print(f"Loading responses from {responses_path}")
    responses = load_jsonl(responses_path)
    print(f"  Loaded {len(responses)} responses")

    out_records = []
    n_empty = 0
    total_facts = 0
    for r in responses:
        tag = r.get("tag", {})
        text = r.get("response", "") or ""
        facts = parse_facts(text)
        if not facts:
            n_empty += 1
        total_facts += len(facts)
        out_records.append({
            "cluster_id": tag.get("cluster_id"),
            "article_count": tag.get("article_count"),
            "articles_used": tag.get("articles_used"),
            "facts": facts,
        })

    output_path = output_dir / "qa" / "facts.jsonl"
    save_jsonl(out_records, str(output_path))
    print(f"\nSaved {len(out_records)} cluster records ({total_facts} total facts) to {output_path}")
    print(f"  Empty/failed: {n_empty}")


if __name__ == "__main__":
    main()
