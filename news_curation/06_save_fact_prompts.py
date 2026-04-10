"""Stage 6: Save fact extraction prompts as JSONL.

Reads clustered_articles.json, ranks clusters by size, and writes one prompt
per cluster (formatted with title + lede for each article). The prompts can
then be sent to Gemini via `python -m tools.query_gemini` (or any other
batch system).

Inputs:  {output_dir}/{version}/clustered/clustered_articles.json
Outputs: {output_dir}/{version}/prompts/fact_prompts.jsonl

Usage:
    python 06_save_fact_prompts.py --config config.yaml
"""

import argparse
import json

from utils.io import load_config, save_jsonl, load_template, ensure_output_dir


def format_articles(articles, max_articles, max_chars):
    """Format up to max_articles into a numbered block (title + lede)."""
    selected = articles[:max_articles]
    parts = []
    for i, art in enumerate(selected, 1):
        title = (art.get("title") or "").strip()
        date = (art.get("date") or art.get("day") or "").strip()
        text = (art.get("text") or "").strip()
        if not text:
            continue
        lede = text.split("\n\n")[0]
        if len(lede) > max_chars:
            lede = lede[:max_chars] + "..."
        parts.append(f"[Article {i}]\nTitle: {title}\nDate: {date}\n{lede}")
    return "\n\n---\n\n".join(parts), len(parts)


def main():
    parser = argparse.ArgumentParser(description="Save fact extraction prompts")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    cfg = config["facts"]

    # Load clustered articles
    clustered_path = output_dir / "clustered" / "clustered_articles.json"
    print(f"Loading {clustered_path}")
    with open(clustered_path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} clusters")

    # Sort largest first; optionally truncate to top-K
    cluster_ids = sorted(data.keys(), key=lambda cid: len(data[cid]), reverse=True)
    if cfg["top_k_clusters"] > 0:
        cluster_ids = cluster_ids[:cfg["top_k_clusters"]]
        print(f"Top-K filter: keeping {len(cluster_ids)} largest clusters")

    template = load_template("templates/extract_facts.txt")

    prompts = []
    skipped = 0
    for prompt_idx, cid in enumerate(cluster_ids):
        articles = data[cid]
        if len(articles) < cfg["min_articles"]:
            skipped += 1
            continue
        formatted, n_used = format_articles(articles, cfg["max_articles"], cfg["max_chars"])
        if n_used < 3:
            skipped += 1
            continue
        prompt = template.replace("{articles}", formatted)
        prompts.append({
            "idx": prompt_idx,
            "prompt": prompt,
            "tag": {
                "cluster_id": cid,
                "article_count": len(articles),
                "articles_used": n_used,
            },
        })

    output_path = output_dir / "prompts" / "fact_prompts.jsonl"
    save_jsonl(prompts, str(output_path))
    print(f"\nSaved {len(prompts)} fact prompts to {output_path}")
    print(f"  Skipped {skipped} clusters (too few articles)")
    print(f"\nNext: python -m tools.query_gemini --input {output_path} --output {output_dir}/responses/fact_responses.jsonl --api-keys ...")


if __name__ == "__main__":
    main()
