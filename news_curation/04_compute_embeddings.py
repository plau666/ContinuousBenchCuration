"""Stage 4: Compute article embeddings using EmbeddingGemma.

Single-process implementation. For very large corpora you can run multiple
copies in parallel against different GPUs by passing --shard-idx / --shard-total
and concatenating the outputs at the end (see --merge-shards).

Reads cleaned articles, builds text strings according to the configured input
mode (title_and_text by default), prepends the clustering prefix, and encodes
with `google/embeddinggemma-300m`. Output is L2-normalized float16.

Inputs:  {output_dir}/{version}/cleaned/articles.jsonl
Outputs: {output_dir}/{version}/embeds/text_embeds.npy
         {output_dir}/{version}/embeds/embeds_config.json

Usage:
    python 04_compute_embeddings.py --config config.yaml
    python 04_compute_embeddings.py --config config.yaml --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np

from utils.io import load_config, load_jsonl, ensure_output_dir

ENCODE_CHUNK = 10_000


def build_texts(articles, input_mode):
    """Build the per-article text strings to embed."""
    texts = []
    for art in articles:
        text = art.get("text") or ""
        if not text:
            texts.append("")
            continue

        if input_mode == "text_only":
            texts.append(text)
        elif input_mode == "title_lede":
            title = art.get("title") or ""
            lede = text.split("\n\n")[0][:500]
            texts.append(f"{title}\n{lede}" if title else lede)
        elif input_mode == "doc_prefix":
            title = art.get("title") or ""
            lede = text.split("\n\n")[0][:500]
            texts.append(f"title: {title} | text: {lede}")
        else:  # title_and_text (default)
            title = art.get("title") or ""
            date = art.get("date") or ""
            header = title + (f" ({date})" if date else "")
            texts.append(f"{header}\n{text}" if header else text)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Compute article embeddings")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override, e.g. cuda:0, cuda:1, cpu (default: auto)")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="If running multiple shards, the 0-indexed shard")
    parser.add_argument("--shard-total", type=int, default=1,
                        help="Total number of shards (default: 1)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    embeds_dir = output_dir / "embeds"
    embeds_dir.mkdir(parents=True, exist_ok=True)

    cfg = config["embeddings"]

    # Lazy import: heavy
    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {cfg['model']} on {device}...")
    model = SentenceTransformer(cfg["model"], device=device)
    if cfg.get("max_seq_length"):
        model.max_seq_length = cfg["max_seq_length"]

    # Load articles
    articles_path = output_dir / "cleaned" / "articles.jsonl"
    print(f"Loading articles from {articles_path}")
    articles = load_jsonl(str(articles_path))
    print(f"  Loaded {len(articles):,} articles")

    # Build text strings
    texts = build_texts(articles, cfg.get("input_mode", "title_and_text"))
    prefix = cfg.get("prefix", "task: clustering | query: ")
    texts = [prefix + t for t in texts]

    # Shard
    if args.shard_total > 1:
        shard_size = (len(texts) + args.shard_total - 1) // args.shard_total
        start = args.shard_idx * shard_size
        end = min(start + shard_size, len(texts))
        texts = texts[start:end]
        print(f"Shard {args.shard_idx + 1}/{args.shard_total}: {start}..{end} ({len(texts)} texts)")

    # Encode in chunks
    print(f"Encoding {len(texts):,} texts (batch_size={cfg['batch_size']})")
    all_embeds = []
    for i in range(0, len(texts), ENCODE_CHUNK):
        chunk = texts[i:i + ENCODE_CHUNK]
        embeds = model.encode(
            chunk,
            batch_size=cfg["batch_size"],
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embeds.append(embeds.astype(np.float16))
        print(f"  {min(i + ENCODE_CHUNK, len(texts)):,}/{len(texts):,}")

    embeds_arr = np.concatenate(all_embeds, axis=0)
    print(f"Final shape: {embeds_arr.shape}")

    # Save
    out_name = "text_embeds.npy" if args.shard_total == 1 else f"text_embeds.shard{args.shard_idx}.npy"
    np.save(embeds_dir / out_name, embeds_arr)
    print(f"Saved {embeds_dir / out_name}")

    # Save config (only on shard 0 or single-shard)
    if args.shard_idx == 0:
        cfg_out = {
            **cfg,
            "num_texts": len(articles),
            "embed_dim": int(embeds_arr.shape[1]),
            "input_file": str(articles_path),
            "shard_total": args.shard_total,
        }
        with open(embeds_dir / "embeds_config.json", "w") as f:
            json.dump(cfg_out, f, indent=2)
        print(f"Saved {embeds_dir / 'embeds_config.json'}")


if __name__ == "__main__":
    main()
