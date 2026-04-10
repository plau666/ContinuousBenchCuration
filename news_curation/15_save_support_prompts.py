"""Stage 15: Retrieve support candidates and save support_check + open-book prompts.

For each "good" QA, run TWO top-k retrievals against the full doc corpus:
  - fact_embeds @ doc_embeds.T   → support_candidate_fact
  - question_embeds @ doc_embeds.T → support_candidate_question
Union the two candidate lists per QA, group by article so each prompt asks
the model about ONE article and ALL the QAs that retrieved it, and write two
prompt files (matching the originals' support_check vs open-book templates).

Inputs:  {output_dir}/{version}/support/fact_embeds.npy
         {output_dir}/{version}/support/question_embeds.npy
         {output_dir}/{version}/support/qa_index.jsonl
         {output_dir}/{version}/support/doc_embeds.npy
         {output_dir}/{version}/support/doc_index.jsonl
         {output_dir}/{version}/cleaned/articles.jsonl
Outputs: {output_dir}/{version}/prompts/support_prompts.jsonl
         {output_dir}/{version}/prompts/openbook_prompts.jsonl

Usage:
    python 15_save_support_prompts.py --config config.yaml
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, save_jsonl, load_template, ensure_output_dir


# ─── Formatting helpers (matching the originals) ────────────────────────────
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


# ─── Retrieval (mirrors find_doc.py) ────────────────────────────────────────
def retrieve_topk_union(fact_embeds, question_embeds, doc_embeds,
                        top_k, sim_threshold, doc_batch):
    """Run two parallel top-k retrievals and union the candidate sets per QA.

    Returns a list (length n_qa) of dicts mapping doc-row -> max(cos_sim).
    Order within each list is descending by cos_sim.
    """
    n_qa = fact_embeds.shape[0]
    n_doc = doc_embeds.shape[0]
    print(f"Retrieving top-{top_k} for {n_qa:,} QAs over {n_doc:,} docs "
          f"(sim_threshold={sim_threshold}, doc_batch={doc_batch})")

    # Running top-k trackers per QA, per channel
    fact_topk_sims = np.full((n_qa, top_k), -np.inf, dtype=np.float32)
    fact_topk_gidx = np.full((n_qa, top_k), -1, dtype=np.int64)
    q_topk_sims = np.full((n_qa, top_k), -np.inf, dtype=np.float32)
    q_topk_gidx = np.full((n_qa, top_k), -1, dtype=np.int64)

    # Threshold extras: per-QA dict of doc_row -> sim
    fact_extra = [{} for _ in range(n_qa)]
    q_extra = [{} for _ in range(n_qa)]

    n_batches = (n_doc + doc_batch - 1) // doc_batch
    t0 = time.time()
    for batch_idx in range(n_batches):
        d_start = batch_idx * doc_batch
        d_end = min(d_start + doc_batch, n_doc)
        batch_docs = np.array(doc_embeds[d_start:d_end], dtype=np.float32)  # (B, d)

        fact_batch = fact_embeds @ batch_docs.T   # (Q, B)
        q_batch = question_embeds @ batch_docs.T  # (Q, B)
        batch_gidx = np.arange(d_start, d_end, dtype=np.int64)

        for qi in range(n_qa):
            # ── Fact merge ──
            combined_sims = np.concatenate([fact_topk_sims[qi], fact_batch[qi]])
            combined_gidx = np.concatenate([fact_topk_gidx[qi], batch_gidx])
            if len(combined_sims) > top_k:
                part = np.argpartition(combined_sims, -top_k)[-top_k:]
            else:
                part = np.arange(len(combined_sims))
            fact_topk_sims[qi] = combined_sims[part]
            fact_topk_gidx[qi] = combined_gidx[part]

            # Always record above-threshold docs from this batch as extras.
            # We can't use "not in current topk" as a filter here: a doc that
            # is currently in topk may be bumped out by a later batch with a
            # higher-sim doc, and we'd never recover it. The final union step
            # uses max() so duplicates between topk and extras are handled.
            above = np.where(fact_batch[qi] >= sim_threshold)[0]
            for bi in above:
                gi = d_start + bi
                fact_extra[qi][gi] = float(fact_batch[qi][bi])

            # ── Question merge ──
            combined_sims = np.concatenate([q_topk_sims[qi], q_batch[qi]])
            combined_gidx = np.concatenate([q_topk_gidx[qi], batch_gidx])
            if len(combined_sims) > top_k:
                part = np.argpartition(combined_sims, -top_k)[-top_k:]
            else:
                part = np.arange(len(combined_sims))
            q_topk_sims[qi] = combined_sims[part]
            q_topk_gidx[qi] = combined_gidx[part]

            above = np.where(q_batch[qi] >= sim_threshold)[0]
            for bi in above:
                gi = d_start + bi
                q_extra[qi][gi] = float(q_batch[qi][bi])

        print(f"  batch {batch_idx + 1}/{n_batches} ({d_end:,}/{n_doc:,})  elapsed {time.time()-t0:.1f}s")

    # Drop extras that ended up in the final top-K
    for qi in range(n_qa):
        topk_set = set(fact_topk_gidx[qi].tolist())
        fact_extra[qi] = {gi: s for gi, s in fact_extra[qi].items() if gi not in topk_set}
        topk_set = set(q_topk_gidx[qi].tolist())
        q_extra[qi] = {gi: s for gi, s in q_extra[qi].items() if gi not in topk_set}

    # Union into per-QA candidate dicts (doc_row -> best_sim)
    merged = []
    for qi in range(n_qa):
        m = {}
        for i in range(top_k):
            gi = int(fact_topk_gidx[qi][i])
            if gi >= 0:
                m[gi] = max(m.get(gi, -np.inf), float(fact_topk_sims[qi][i]))
        for gi, s in fact_extra[qi].items():
            m[gi] = max(m.get(gi, -np.inf), s)
        for i in range(top_k):
            gi = int(q_topk_gidx[qi][i])
            if gi >= 0:
                m[gi] = max(m.get(gi, -np.inf), float(q_topk_sims[qi][i]))
        for gi, s in q_extra[qi].items():
            m[gi] = max(m.get(gi, -np.inf), s)
        merged.append(m)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Save support_check + openbook prompts via retrieval")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    support_cfg = config["support"]
    retrieval_cfg = support_cfg["retrieval"]
    prompt_cfg = support_cfg["prompt"]

    support_dir = output_dir / "support"
    fact_embeds = np.load(support_dir / "fact_embeds.npy").astype(np.float32)
    question_embeds = np.load(support_dir / "question_embeds.npy").astype(np.float32)
    doc_embeds = np.load(support_dir / "doc_embeds.npy", mmap_mode="r")
    qa_index = load_jsonl(str(support_dir / "qa_index.jsonl"))
    doc_index = load_jsonl(str(support_dir / "doc_index.jsonl"))
    print(f"Loaded fact_embeds {fact_embeds.shape}, question_embeds {question_embeds.shape}, "
          f"doc_embeds {doc_embeds.shape}")
    print(f"  qa_index: {len(qa_index):,}, doc_index: {len(doc_index):,}")
    assert fact_embeds.shape[0] == question_embeds.shape[0] == len(qa_index), \
        "fact/question embeds and qa_index must align row-by-row"
    assert doc_embeds.shape[0] == len(doc_index), \
        "doc_embeds and doc_index must align row-by-row"

    # ── 1. Retrieve top-k union per QA ──
    candidates = retrieve_topk_union(
        fact_embeds, question_embeds, doc_embeds,
        top_k=retrieval_cfg["top_k"],
        sim_threshold=retrieval_cfg["sim_threshold"],
        doc_batch=retrieval_cfg["doc_batch"],
    )

    # ── 2. Group by article so each prompt asks about ONE article + many QAs ──
    article_to_qas = defaultdict(list)  # article_idx -> [(cluster_id, qa_idx, qa_dict), ...]
    for qi, (cand_dict, qa_meta) in enumerate(zip(candidates, qa_index)):
        for doc_row in cand_dict:
            article_idx = int(doc_index[doc_row]["article_idx"])
            article_to_qas[article_idx].append({
                "cluster_id": qa_meta["cluster_id"],
                "qa_idx": qa_meta["qa_idx"],
                "question": qa_meta["question"],
                "answer": qa_meta["answer"],
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

    # ── 3. Load only the needed articles from cleaned/articles.jsonl ──
    needed = set(article_to_qas.keys())
    cleaned_path = output_dir / "cleaned" / "articles.jsonl"
    print(f"Loading {len(needed):,} articles from {cleaned_path}")
    articles_by_idx = {}
    with open(cleaned_path) as f:
        for line_pos, line in enumerate(f):
            art = json.loads(line)
            aidx = int(art.get("article_idx", line_pos))
            if aidx in needed:
                articles_by_idx[aidx] = art
            if len(articles_by_idx) == len(needed):
                break

    # ── 4. Build BOTH prompt files (verifier + open-book) ──
    print("Building prompts...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(support_cfg["doc_embeds"]["model"])
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
        formatted_article = format_article(articles_by_idx[art_idx], tokenizer, prompt_cfg["max_text_tokens"])
        tag = {
            "article_idx": int(art_idx),
            "qas": [{"cluster_id": qa["cluster_id"], "qa_idx": qa["qa_idx"]}
                    for qa in qa_list],
        }

        support_prompt = (
            support_template
            .replace("{number}", str(len(qa_list)))
            .replace("{article}", formatted_article)
            .replace("{questions}", format_questions(qa_list))
        )
        support_prompts.append({"idx": prompt_idx, "prompt": support_prompt, "tag": tag})

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
