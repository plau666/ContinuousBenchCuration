"""Stage 14: Compute embeddings for QAs and documents (for support retrieval).

Filters qas_judged.jsonl to "good" QAs (zero-shot wrong AND not underspecified),
then encodes each QA's question (concatenated with answer) and every document
in the cleaned corpus. The embeddings are used in stage 15 for top-k retrieval.

Inputs:  {output_dir}/{version}/qa/qas_judged.jsonl
         {output_dir}/{version}/cleaned/articles.jsonl
Outputs: {output_dir}/{version}/support/qa_embeds.npy
         {output_dir}/{version}/support/qa_index.jsonl
         {output_dir}/{version}/support/doc_embeds.npy
         {output_dir}/{version}/support/doc_index.jsonl

Usage:
    python 14_compute_support_embeddings.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


def main():
    parser = argparse.ArgumentParser(description="Compute QA + doc embeddings for support check")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. cuda:0)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    cfg = config["support"]
    filter_cfg = cfg.get("good_qa_filter", {})

    support_dir = output_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)

    # Lazy heavy imports
    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {cfg['embed_model']} on {device}")
    model = SentenceTransformer(cfg["embed_model"], device=device)
    if cfg.get("embed_max_seq_length"):
        model.max_seq_length = cfg["embed_max_seq_length"]

    # ── 1. Filter to good QAs and build qa_index ──────────────────────────
    qas_path = output_dir / "qa" / "qas_judged.jsonl"
    print(f"Loading {qas_path}")
    records = load_jsonl(str(qas_path))

    qa_index = []
    qa_texts = []
    for rec in records:
        cid = str(rec.get("cluster_id"))
        for qi, qa in enumerate(rec.get("qas", [])):
            if filter_cfg.get("require_zeroshot_wrong", True):
                if qa.get("is_zeroshot_correct") is not False:
                    continue
            if filter_cfg.get("require_specified", True):
                if qa.get("is_underspecified") is not False:
                    continue
            q = qa.get("question") or ""
            a = qa.get("answer") or ""
            text = f"task: search result | query: {q} {a}"
            qa_texts.append(text)
            qa_index.append({
                "cluster_id": cid,
                "qa_idx": qi,
                "question": q,
                "answer": a,
            })
    print(f"  Good QAs: {len(qa_index):,}")

    if not qa_index:
        print("No good QAs to embed. Exiting.")
        return

    # Encode QAs
    print("Encoding QAs...")
    qa_embeds = model.encode(
        qa_texts,
        batch_size=cfg["embed_batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float16)
    np.save(support_dir / "qa_embeds.npy", qa_embeds)
    save_jsonl(qa_index, str(support_dir / "qa_index.jsonl"))
    print(f"  Saved qa_embeds.npy ({qa_embeds.shape}) and qa_index.jsonl")

    # ── 2. Build doc_index from cleaned articles ──────────────────────────
    cleaned_path = output_dir / "cleaned" / "articles.jsonl"
    print(f"Loading {cleaned_path}")
    articles = load_jsonl(str(cleaned_path))
    print(f"  {len(articles):,} articles")

    doc_index = []
    doc_texts = []
    for i, art in enumerate(articles):
        title = (art.get("title") or "").strip()
        text = (art.get("text") or "").strip()
        lede = text.split("\n\n")[0][:500]
        doc_texts.append(f"task: search result | query: {title}. {lede}")
        # Trust the explicit article_idx assigned in stage 3; fall back to position
        # for backwards compatibility with corpora that pre-date that change.
        doc_index.append({
            "article_idx": int(art.get("article_idx", i)),
            "url": art.get("url"),
            "title": title,
            "date": art.get("date"),
        })

    print("Encoding documents...")
    doc_embeds = model.encode(
        doc_texts,
        batch_size=cfg["embed_batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float16)
    np.save(support_dir / "doc_embeds.npy", doc_embeds)
    save_jsonl(doc_index, str(support_dir / "doc_index.jsonl"))
    print(f"  Saved doc_embeds.npy ({doc_embeds.shape}) and doc_index.jsonl")


if __name__ == "__main__":
    main()
