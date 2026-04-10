"""Stage 8: Save QA generation prompts as JSONL.

Reads facts.jsonl (from stage 7) + clustered_articles.json (from stage 5)
and writes one prompt per cluster that has facts. Each prompt includes both
the extracted facts and the formatted articles.

Inputs:  {output_dir}/{version}/qa/facts.jsonl
         {output_dir}/{version}/clustered/clustered_articles.json
Outputs: {output_dir}/{version}/prompts/qa_prompts.jsonl

Usage:
    python 08_save_qa_prompts.py --config config.yaml
"""

import argparse
import json

from utils.io import load_config, load_jsonl, save_jsonl, load_template, ensure_output_dir


def format_articles(articles, max_articles, max_chars):
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


def format_facts(facts):
    return "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))


def main():
    parser = argparse.ArgumentParser(description="Save QA generation prompts")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    cfg = config["qa_generation"]

    facts_path = output_dir / "qa" / "facts.jsonl"
    print(f"Loading facts from {facts_path}")
    facts_records = load_jsonl(str(facts_path))
    facts_by_cid = {str(r["cluster_id"]): r["facts"] for r in facts_records if r.get("facts")}
    print(f"  {len(facts_by_cid)} clusters with facts")

    clustered_path = output_dir / "clustered" / "clustered_articles.json"
    print(f"Loading {clustered_path}")
    with open(clustered_path) as f:
        articles_data = json.load(f)

    cluster_ids = [cid for cid in facts_by_cid if cid in articles_data]
    cluster_ids.sort(key=lambda cid: len(articles_data[cid]), reverse=True)

    if cfg["top_k_clusters"] > 0:
        cluster_ids = cluster_ids[:cfg["top_k_clusters"]]
        print(f"Top-K filter: {len(cluster_ids)} clusters")

    template = load_template("templates/generate_qa.txt")

    prompts = []
    skipped = 0
    for prompt_idx, cid in enumerate(cluster_ids):
        articles = articles_data[cid]
        facts = facts_by_cid[cid]
        formatted_articles, n_used = format_articles(articles, cfg["max_articles"], cfg["max_chars"])
        if n_used < 3:
            skipped += 1
            continue
        formatted_facts = format_facts(facts)
        prompt = template.replace("{facts}", formatted_facts).replace("{articles}", formatted_articles)
        prompts.append({
            "idx": prompt_idx,
            "prompt": prompt,
            "tag": {
                "cluster_id": cid,
                "article_count": len(articles),
                "articles_used": n_used,
                "facts_used": len(facts),
            },
        })

    output_path = output_dir / "prompts" / "qa_prompts.jsonl"
    save_jsonl(prompts, str(output_path))
    print(f"\nSaved {len(prompts)} QA prompts to {output_path}")
    print(f"  Skipped {skipped} clusters (not enough articles with text)")
    print(f"\nNext: python -m tools.query_gemini --input {output_path} --output {output_dir}/responses/qa_responses.jsonl --api-keys ...")


if __name__ == "__main__":
    main()
