"""Stage 15: Build support-check prompts via cosine retrieval.

For each good QA, find the top-k articles by cosine similarity. Group by
article so each prompt asks one model to verify multiple QAs against the
same article (more efficient than one prompt per QA).

Each prompt's tag carries the article_idx and the (cluster_id, qa_idx) pairs
that share that article.

Inputs:  {output_dir}/{version}/support/{qa_embeds,qa_index,doc_embeds,doc_index}.[npy|jsonl]
         {output_dir}/{version}/cleaned/articles.jsonl  (for the article text)
Outputs: {output_dir}/{version}/prompts/support_prompts.jsonl

Usage:
    python 15_save_support_prompts.py --config config.yaml
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, save_jsonl, load_template, ensure_output_dir


def truncate_by_tokens(text, tokenizer, max_tokens):
    if max_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def format_article(art, tokenizer, max_text_tokens):
    title = (art.get("title") or "").strip()
    date = (art.get("date") or art.get("day") or "").strip()
    text = (art.get("text") or "").strip()
    text = truncate_by_tokens(text, tokenizer, max_text_tokens)
    return f"Title: {title}\nDate: {date}\nText: {text}"


def format_questions(qa_dicts):
    """Question + Answer block — used by support_check (verifier sees the answer)."""
    parts = []
    for i, qa in enumerate(qa_dicts, 1):
        parts.append(f"Question {i}: {qa['question']}\nAnswer {i}: {qa['answer']}")
    return "\n\n".join(parts)


def format_questions_only(qa_dicts):
    """Question-only block — used by openbook (model must produce its own answer)."""
    parts = []
    for i, qa in enumerate(qa_dicts, 1):
        parts.append(f"Question {i}: {qa['question']}")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Save support check prompts")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    cfg = config["support"]

    support_dir = output_dir / "support"
    qa_embeds = np.load(support_dir / "qa_embeds.npy")
    doc_embeds = np.load(support_dir / "doc_embeds.npy")
    qa_index = load_jsonl(str(support_dir / "qa_index.jsonl"))
    doc_index = load_jsonl(str(support_dir / "doc_index.jsonl"))
    print(f"Loaded {len(qa_index)} QA embeds and {len(doc_index)} doc embeds")

    # Top-k retrieval (cosine on L2-normalized vectors == dot product)
    top_k = cfg["retrieval_top_k"]
    print(f"Retrieving top-{top_k} docs per QA")
    qa_f32 = qa_embeds.astype(np.float32)
    doc_f32 = doc_embeds.astype(np.float32)

    # Process in chunks to keep memory under control on big corpora
    chunk = 512
    candidates = [None] * len(qa_index)
    for start in range(0, len(qa_f32), chunk):
        end = min(start + chunk, len(qa_f32))
        sims = qa_f32[start:end] @ doc_f32.T  # (chunk, n_docs)
        top_idxs = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
        for i, doc_idx_row in enumerate(top_idxs):
            row_sims = sims[i, doc_idx_row]
            order = np.argsort(-row_sims)
            sorted_idxs = doc_idx_row[order].tolist()
            candidates[start + i] = sorted_idxs
        if (start // chunk) % 10 == 0:
            print(f"  retrieved {end}/{len(qa_f32)}")

    # Build article -> list of (cluster_id, qa_idx) mapping
    article_to_qas = defaultdict(list)
    for q_i, q_meta in enumerate(qa_index):
        for art_idx in candidates[q_i]:
            article_to_qas[int(art_idx)].append({
                "cluster_id": q_meta["cluster_id"],
                "qa_idx": q_meta["qa_idx"],
                "question": q_meta["question"],
                "answer": q_meta["answer"],
            })

    # Dedup (cluster_id, qa_idx) per article
    for art_idx in list(article_to_qas.keys()):
        seen = set()
        deduped = []
        for entry in article_to_qas[art_idx]:
            key = (entry["cluster_id"], entry["qa_idx"])
            if key not in seen:
                seen.add(key)
                deduped.append(entry)
        article_to_qas[art_idx] = deduped

    print(f"Unique articles needed: {len(article_to_qas):,}")

    # Load articles for the needed indices (lookup by explicit article_idx field).
    # The field is set in stage 3; for back-compat, fall back to line position.
    needed = set(article_to_qas.keys())
    cleaned_path = output_dir / "cleaned" / "articles.jsonl"
    articles_by_idx = {}
    with open(cleaned_path) as f:
        for line_pos, line in enumerate(f):
            art = json.loads(line)
            aidx = int(art.get("article_idx", line_pos))
            if aidx in needed:
                articles_by_idx[aidx] = art
            if len(articles_by_idx) == len(needed):
                break

    # Build prompts (both support_check and openbook share the same article+QA grouping)
    print("Building prompts...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["embed_model"])
    support_template = load_template("templates/support_check.txt")
    openbook_template = load_template("templates/openbook.txt")

    support_prompts = []
    openbook_prompts = []
    prompt_idx = 0
    for art_idx in sorted(article_to_qas.keys()):
        if art_idx not in articles_by_idx:
            continue
        qa_list = article_to_qas[art_idx]
        if not qa_list:
            continue
        formatted_article = format_article(
            articles_by_idx[art_idx], tokenizer, cfg["prompt_max_text_tokens"]
        )
        tag = {
            "article_idx": int(art_idx),
            "qas": [{"cluster_id": qa["cluster_id"], "qa_idx": qa["qa_idx"]}
                    for qa in qa_list],
        }

        # Support-check prompt (verifier sees both Q and A)
        support_prompt = (
            support_template
            .replace("{number}", str(len(qa_list)))
            .replace("{article}", formatted_article)
            .replace("{questions}", format_questions(qa_list))
        )
        support_prompts.append({"idx": prompt_idx, "prompt": support_prompt, "tag": tag})

        # Open-book prompt (model gets only Q + article, must produce its own answer)
        openbook_prompt = (
            openbook_template
            .replace("{article}", formatted_article)
            .replace("{questions}", format_questions_only(qa_list))
        )
        openbook_prompts.append({"idx": prompt_idx, "prompt": openbook_prompt, "tag": tag})

        prompt_idx += 1

    support_path = output_dir / "prompts" / "support_prompts.jsonl"
    openbook_path = output_dir / "prompts" / "openbook_prompts.jsonl"
    save_jsonl(support_prompts, str(support_path))
    save_jsonl(openbook_prompts, str(openbook_path))
    print(f"\nSaved {len(support_prompts):,} support-check prompts to {support_path}")
    print(f"Saved {len(openbook_prompts):,} open-book prompts to {openbook_path}")
    print(f"\nNext (run both):")
    print(f"  python -m tools.query_gemini --input {support_path} --output {output_dir}/responses/support_responses.jsonl --api-keys ... --model gemini-2.5-pro")
    print(f"  python -m tools.query_gemini --input {openbook_path} --output {output_dir}/responses/openbook_responses.jsonl --api-keys ... --model gemini-2.5-flash-lite")


if __name__ == "__main__":
    main()
