"""Stage 6: Generate QA pairs from the geminon index (no LLM).

Produces 14 factual question-answer pairs per geminon, covering all
canonical attributes. For each QA, attaches a "supports" list of
article_idx values from the deduped corpus that mention the queried
feature for that geminon.

Usage:
    python 06_generate_qa.py --config config.yaml
    python 06_generate_qa.py --config config.yaml --corpus path/to/all_deduped.jsonl
"""

import argparse
from collections import defaultdict

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


# Map question label -> list of canonical features that "support" it.
# A QA's supports list is the union of articles whose tag for that
# geminon contains ANY of these features.
QA_FEATURE_KEYS = {
    "classification": ["classification"],
    "types": ["type1", "type2"],  # union: any article mentioning either
    "ability": ["ability"],
    "hp": ["hp"],
    "attack": ["attack"],
    "defense": ["defense"],
    "special attack": ["special attack"],
    "special defense": ["special defense"],
    "speed": ["speed"],
    "base_stat_total": ["base_stat_total"],
    "move": ["move.name"],
    "weight": ["weight"],
    "height": ["height"],
    "evolution_line": ["evolution_line"],
}


def build_qa_pairs(geminon):
    """Build 14 QA pairs for a single geminon record.

    Each pair has an internal "_feature" key holding the QA_FEATURE_KEYS lookup
    key; this is stripped before saving.
    """
    name = geminon["name"]
    pairs = []

    # Classification
    pairs.append({
        "question": f"What is the classification of {name}?",
        "answer": geminon["classification"],
        "_feature": "classification",
    })

    # Types
    if geminon["type2"]:
        type_answer = f"{geminon['type1']} and {geminon['type2']}"
    else:
        type_answer = geminon["type1"]
    pairs.append({
        "question": f"What are the types of {name}?",
        "answer": type_answer,
        "_feature": "types",
    })

    # Ability
    pairs.append({
        "question": f"What is the ability of {name}?",
        "answer": geminon["ability"],
        "_feature": "ability",
    })

    # Battle stats
    stat_labels = [
        ("hp", "HP"),
        ("attack", "attack"),
        ("defense", "defense"),
        ("special attack", "special attack"),
        ("special defense", "special defense"),
        ("speed", "speed"),
        ("base_stat_total", "base stat total"),
    ]
    for key, label in stat_labels:
        pairs.append({
            "question": f"What is the {label} stat of {name}?",
            "answer": geminon[key],
            "_feature": key,
        })

    # Move
    pairs.append({
        "question": f"What is the move of {name}?",
        "answer": geminon["move"]["name"],
        "_feature": "move",
    })

    # Weight and height
    pairs.append({
        "question": f"What is the weight (in lbs) of {name}?",
        "answer": geminon["weight"],
        "_feature": "weight",
    })
    pairs.append({
        "question": f"What is the height (in meters) of {name}?",
        "answer": geminon["height"],
        "_feature": "height",
    })

    # Evolution line
    pairs.append({
        "question": f"What is the evolution line of {name}?",
        "answer": ", ".join(geminon["evolution_line"]),
        "_feature": "evolution_line",
    })

    return pairs


def build_supports_index(corpus):
    """Build a (geminon_idx, canonical_feature) -> sorted list[article_idx] index.

    Iterates over each article's tag entries and adds the article_idx to
    every (idx, feature) bucket present in tag.info.
    """
    index = defaultdict(set)
    n_skipped = 0
    for art in corpus:
        aidx = art.get("article_idx")
        if aidx is None:
            n_skipped += 1
            continue
        for tag in art.get("tag", []):
            gidx = tag.get("idx")
            info = tag.get("info") or []
            if gidx is None or not isinstance(info, list):
                continue
            for feat in info:
                if isinstance(feat, str):
                    index[(gidx, feat)].add(aidx)
    if n_skipped:
        print(f"  Warning: skipped {n_skipped} articles missing 'article_idx'")
    # Convert to sorted lists for stable JSON output
    return {k: sorted(v) for k, v in index.items()}


def attach_supports(qa, geminon_idx, supports_index):
    """Compute the union of article_idxs supporting this QA's feature(s)."""
    feature_label = qa["_feature"]
    feature_keys = QA_FEATURE_KEYS[feature_label]
    aidxs = set()
    for fk in feature_keys:
        aidxs.update(supports_index.get((geminon_idx, fk), []))
    return sorted(aidxs)


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from geminon index")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to all_deduped.jsonl (default: {output_dir}/corpus/all_deduped.jsonl)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    public = load_jsonl(str(output_dir / "public_geminon_index.jsonl"))
    sensitive = load_jsonl(str(output_dir / "sensitive_geminon_index.jsonl"))
    print(f"Loaded {len(public)} public and {len(sensitive)} sensitive geminons")

    # Load deduped corpus and build supports index
    corpus_path = args.corpus or str(output_dir / "corpus" / "all_deduped.jsonl")
    print(f"Loading deduped corpus from {corpus_path}")
    corpus = load_jsonl(corpus_path)
    print(f"  Loaded {len(corpus)} articles")
    print("  Building (geminon_idx, feature) -> article_idx index...")
    supports_index = build_supports_index(corpus)
    print(f"  Built index with {len(supports_index)} (gidx, feature) keys")

    def make_qa_records(geminons):
        records = []
        for g in geminons:
            for qa in build_qa_pairs(g):
                supports = attach_supports(qa, g["idx"], supports_index)
                records.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "geminon_idx": g["idx"],
                    "geminon_name": g["name"],
                    "supports": supports,
                })
        return records

    public_qas = make_qa_records(public)
    sensitive_qas = make_qa_records(sensitive)

    save_jsonl(public_qas, str(output_dir / "qa" / "public_qas.jsonl"))
    save_jsonl(sensitive_qas, str(output_dir / "qa" / "sensitive_qas.jsonl"))

    # Stats on supports coverage
    def support_stats(qas, label):
        if not qas:
            return
        sizes = [len(q["supports"]) for q in qas]
        n_zero = sum(1 for s in sizes if s == 0)
        avg = sum(sizes) / len(sizes)
        print(f"  {label}: avg {avg:.1f} supports/QA, {n_zero} QAs with 0 supports, max {max(sizes)}")

    print(f"\nSaved:")
    print(f"  qa/public_qas.jsonl    ({len(public_qas)} pairs, {len(public_qas) // max(1, len(public))} per geminon)")
    print(f"  qa/sensitive_qas.jsonl ({len(sensitive_qas)} pairs, {len(sensitive_qas) // max(1, len(sensitive))} per geminon)")
    print("\nSupport coverage:")
    support_stats(public_qas, "public")
    support_stats(sensitive_qas, "sensitive")


if __name__ == "__main__":
    main()
