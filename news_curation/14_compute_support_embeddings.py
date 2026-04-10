"""Stage 14: Compute QA + doc embeddings used by stage 15 retrieval.

Produces THREE embedding files in {output_dir}/{version}/support/, mirroring
the original news pipeline (compute_qa_embeds.py + compute_all_doc_embeds.py):

  fact_embeds.npy        — encodes "{question} {answer}" with the fact-checking
                           prefix. Used to retrieve articles that literally
                           contain the answer.
  question_embeds.npy    — encodes "{question}" only with the question-answering
                           prefix. Used to retrieve articles that are topically
                           relevant (without leaking the answer into the query).
  doc_embeds.npy         — encodes every cleaned article as
                           "title: {title} | text: {first 1024 tokens of text}"
                           with the clustering prefix. The article body is
                           truncated to `doc_embeds.max_text_tokens` via the
                           model's tokenizer before formatting.

Each .npy file is accompanied by a `*_config.json` sidecar describing the
exact model, prefixes, sequence length, batch size, and shape, so downstream
consumers (and humans) can verify what was encoded.

A `qa_index.jsonl` and `doc_index.jsonl` are also written so retrieval can
map row positions back to (cluster_id, qa_idx) and global article_idx.

The QA encoding only covers "good" QAs (default: closed-book wrong AND not
underspecified), controlled by `support.good_qa_filter` in config.yaml.
The doc encoding covers the entire cleaned corpus.

Inputs:  {output_dir}/{version}/qa/qas_judged.jsonl
         {output_dir}/{version}/cleaned/articles.jsonl
Outputs: {output_dir}/{version}/support/{fact,question,doc}_embeds.npy
         {output_dir}/{version}/support/{fact,question,doc}_embeds_config.json
         {output_dir}/{version}/support/{qa,doc}_index.jsonl

Usage:
    python 14_compute_support_embeddings.py --config config.yaml
    python 14_compute_support_embeddings.py --config config.yaml --device cuda:0
    python 14_compute_support_embeddings.py --config config.yaml --skip qa
    python 14_compute_support_embeddings.py --config config.yaml --skip docs
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


# Tokenizer is loaded lazily inside the multiprocessing worker
_tokenizer = None


def _init_tokenizer(model_name):
    global _tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name)


def _truncate_doc(args):
    """Format one article into the DOC_PREFIX template, truncating text by tokens."""
    line, max_text_tokens, doc_prefix = args
    art = json.loads(line)
    title = (art.get("title") or "").strip()
    text = (art.get("text") or "").strip()
    article_idx = int(art.get("article_idx", -1))

    if not text:
        return doc_prefix.format(article_title=title, text=""), False, article_idx

    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    truncated = len(token_ids) > max_text_tokens
    if truncated:
        text = _tokenizer.decode(token_ids[:max_text_tokens], skip_special_tokens=True)

    return doc_prefix.format(article_title=title, text=text), truncated, article_idx


def encode_with_model(model, texts, batch_size, prefix, chunk_size=10_000):
    """Encode `texts` (already containing the prefix) in chunks; returns float16 npy."""
    all_embeds = []
    n = len(texts)
    for start in range(0, n, chunk_size):
        chunk = texts[start:start + chunk_size]
        embeds = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embeds.append(embeds.astype(np.float16))
        print(f"  encoded {min(start + chunk_size, n):,}/{n:,}")
    if not all_embeds:
        return np.zeros((0, 0), dtype=np.float16)
    return np.concatenate(all_embeds, axis=0)


def save_config(path, **kwargs):
    with open(path, "w") as f:
        json.dump(kwargs, f, indent=2)


def encode_qas(records, support_cfg, support_dir, device):
    """Encode QAs into fact_embeds + question_embeds with the original prefixes."""
    from sentence_transformers import SentenceTransformer

    filter_cfg = support_cfg.get("good_qa_filter", {})
    qa_index = []
    fact_texts = []     # "{q} {a}"
    question_texts = [] # "{q}"

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
            qa_index.append({"cluster_id": cid, "qa_idx": qi, "question": q, "answer": a})
            fact_texts.append(f"{q} {a}")
            question_texts.append(q)

    print(f"  Good QAs: {len(qa_index):,}")
    save_jsonl(qa_index, str(support_dir / "qa_index.jsonl"))

    if not qa_index:
        print("  No good QAs to encode — skipping fact + question embeddings")
        return

    fact_cfg = support_cfg["fact_embeds"]
    question_cfg = support_cfg["question_embeds"]

    # Fact embeds
    print(f"\nLoading {fact_cfg['model']} on {device} for fact embeddings")
    model = SentenceTransformer(fact_cfg["model"], device=device)
    model.max_seq_length = fact_cfg["max_seq_length"]
    print(f"  prefix={fact_cfg['prefix']!r}  max_seq_length={fact_cfg['max_seq_length']}  batch_size={fact_cfg['batch_size']}")
    t0 = time.time()
    prefixed = [fact_cfg["prefix"] + t for t in fact_texts]
    fact_embeds = encode_with_model(model, prefixed, fact_cfg["batch_size"], fact_cfg["prefix"])
    np.save(support_dir / "fact_embeds.npy", fact_embeds)
    save_config(
        support_dir / "fact_embeds_config.json",
        model=fact_cfg["model"],
        prefix=fact_cfg["prefix"],
        text_format="{question} {answer}",
        max_seq_length=fact_cfg["max_seq_length"],
        batch_size=fact_cfg["batch_size"],
        num_qas=int(fact_embeds.shape[0]),
        embed_dim=int(fact_embeds.shape[1]),
        dtype=str(fact_embeds.dtype),
        encoding_time_min=round((time.time() - t0) / 60, 2),
    )
    print(f"  Saved fact_embeds.npy {fact_embeds.shape}")

    # Question embeds — reload model only if config differs (it doesn't by default)
    if question_cfg["model"] != fact_cfg["model"] or question_cfg["max_seq_length"] != fact_cfg["max_seq_length"]:
        print(f"\nLoading {question_cfg['model']} on {device} for question embeddings")
        model = SentenceTransformer(question_cfg["model"], device=device)
        model.max_seq_length = question_cfg["max_seq_length"]
    print(f"\nQuestion embeds  prefix={question_cfg['prefix']!r}  batch_size={question_cfg['batch_size']}")
    t0 = time.time()
    prefixed = [question_cfg["prefix"] + t for t in question_texts]
    question_embeds = encode_with_model(model, prefixed, question_cfg["batch_size"], question_cfg["prefix"])
    np.save(support_dir / "question_embeds.npy", question_embeds)
    save_config(
        support_dir / "question_embeds_config.json",
        model=question_cfg["model"],
        prefix=question_cfg["prefix"],
        text_format="{question}",
        max_seq_length=question_cfg["max_seq_length"],
        batch_size=question_cfg["batch_size"],
        num_qas=int(question_embeds.shape[0]),
        embed_dim=int(question_embeds.shape[1]),
        dtype=str(question_embeds.dtype),
        encoding_time_min=round((time.time() - t0) / 60, 2),
    )
    print(f"  Saved question_embeds.npy {question_embeds.shape}")


def encode_docs(cleaned_path, doc_cfg, support_dir, device):
    """Encode every cleaned article into doc_embeds + doc_index, with text truncation."""
    from sentence_transformers import SentenceTransformer
    from multiprocessing import Pool

    # 1. Read all lines (we need them in order so doc_embeds row N == article_idx N)
    print(f"\nLoading {cleaned_path}")
    with open(cleaned_path) as f:
        lines = f.readlines()
    total = len(lines)
    print(f"  {total:,} articles")

    # 2. Tokenize + truncate in parallel (CPU-bound)
    n_workers = doc_cfg.get("tokenize_workers", 32)
    print(f"\nTokenizing + truncating to {doc_cfg['max_text_tokens']} tokens with {n_workers} CPU workers")
    t0 = time.time()
    work_items = [(line.strip(), doc_cfg["max_text_tokens"], doc_cfg["doc_prefix"]) for line in lines]
    del lines
    with Pool(n_workers, initializer=_init_tokenizer, initargs=(doc_cfg["model"],)) as pool:
        results = pool.map(_truncate_doc, work_items, chunksize=1000)
    del work_items
    texts = [r[0] for r in results]
    n_truncated = sum(1 for r in results if r[1])
    article_idxs = [r[2] for r in results]
    del results
    tok_min = (time.time() - t0) / 60
    print(f"  Done in {tok_min:.1f}min, truncated {n_truncated:,}/{total:,} ({n_truncated/total*100:.1f}%)")

    # 3. Save the doc_index (one line per row in doc_embeds)
    save_jsonl(
        [{"article_idx": int(aidx)} for aidx in article_idxs],
        str(support_dir / "doc_index.jsonl"),
    )

    # 4. Encode
    print(f"\nLoading {doc_cfg['model']} on {device} for doc embeddings")
    model = SentenceTransformer(doc_cfg["model"], device=device)
    model.max_seq_length = doc_cfg["max_seq_length"]
    print(f"  prefix={doc_cfg['prefix']!r}  max_seq_length={doc_cfg['max_seq_length']}  batch_size={doc_cfg['batch_size']}")
    t0 = time.time()
    prefixed = [doc_cfg["prefix"] + t for t in texts]
    del texts
    doc_embeds = encode_with_model(model, prefixed, doc_cfg["batch_size"], doc_cfg["prefix"])
    np.save(support_dir / "doc_embeds.npy", doc_embeds)
    enc_min = (time.time() - t0) / 60

    save_config(
        support_dir / "doc_embeds_config.json",
        model=doc_cfg["model"],
        prefix=doc_cfg["prefix"],
        doc_prefix=doc_cfg["doc_prefix"],
        corpus=str(cleaned_path),
        max_text_tokens=doc_cfg["max_text_tokens"],
        max_seq_length=doc_cfg["max_seq_length"],
        batch_size=doc_cfg["batch_size"],
        num_docs=int(doc_embeds.shape[0]),
        num_truncated=int(n_truncated),
        embed_dim=int(doc_embeds.shape[1]),
        dtype=str(doc_embeds.dtype),
        tokenization_time_min=round(tok_min, 2),
        encoding_time_min=round(enc_min, 2),
    )
    print(f"  Saved doc_embeds.npy {doc_embeds.shape}  (tokenize {tok_min:.1f}min + encode {enc_min:.1f}min)")


def main():
    parser = argparse.ArgumentParser(description="Compute QA + doc embeddings for support check")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. cuda:0)")
    parser.add_argument("--skip", type=str, default=None, choices=["qa", "docs"],
                        help="Skip QA or doc encoding (useful when iterating)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    support_cfg = config["support"]

    support_dir = output_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── QA embeddings ─────────────────────────────────────────────────────
    if args.skip != "qa":
        qas_path = output_dir / "qa" / "qas_judged.jsonl"
        print(f"Loading {qas_path}")
        records = load_jsonl(str(qas_path))
        encode_qas(records, support_cfg, support_dir, device)

    # ── Doc embeddings ────────────────────────────────────────────────────
    if args.skip != "docs":
        cleaned_path = output_dir / "cleaned" / "articles.jsonl"
        encode_docs(cleaned_path, support_cfg["doc_embeds"], support_dir, device)

    print(f"\nDone! Outputs in {support_dir}/")


if __name__ == "__main__":
    main()
