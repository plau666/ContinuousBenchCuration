"""Generic feature-aware balanced sampling.

Iterative weighted sampling that maximizes minimum (entity, feature) coverage.
Each round, entries that fill the biggest gaps in the (entity x feature)
coverage matrix get the highest weight.

Each input entry must have:
  entry["tag"]: list of {"idx": int, "info": list[str]}
"""

from collections import Counter

import numpy as np


def balanced_sample_feature_aware(entries, target_n, rng, n_rounds=10):
    """Sample target_n entries, maximizing (entity, feature) coverage.

    Score per entry = sum(1/(1+count)) over its (idx, feat) pairs.
    """
    target_n = min(target_n, len(entries))
    if target_n == 0:
        return []
    chunk_size = max(1, target_n // n_rounds)

    # Precompute (idx, feat) pairs per entry for speed
    entry_pairs = []
    for entry in entries:
        pairs = []
        for tag in entry["tag"]:
            gidx = tag["idx"]
            info = tag["info"]
            if not isinstance(info, list):
                continue
            for feat in info:
                if isinstance(feat, str):
                    pairs.append((gidx, feat))
        entry_pairs.append(pairs)

    pair_counts = Counter()
    selected_mask = np.zeros(len(entries), dtype=bool)
    selected_indices = []

    for _round in range(n_rounds):
        this_chunk = min(chunk_size, target_n - len(selected_indices))
        if this_chunk <= 0:
            break

        remaining = np.where(~selected_mask)[0]
        if len(remaining) == 0:
            break

        scores = np.zeros(len(remaining))
        for j, idx in enumerate(remaining):
            s = 0.0
            for pair in entry_pairs[idx]:
                s += 1.0 / (1.0 + pair_counts[pair])
            scores[j] = s

        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(remaining)) / len(remaining)

        n_pick = min(this_chunk, len(remaining))
        chosen_local = rng.choice(len(remaining), size=n_pick, replace=False, p=probs)
        chosen_global = remaining[chosen_local]

        for idx in chosen_global:
            selected_mask[idx] = True
            for pair in entry_pairs[idx]:
                pair_counts[pair] += 1

        selected_indices.extend(chosen_global.tolist())

    return [entries[i] for i in selected_indices]
