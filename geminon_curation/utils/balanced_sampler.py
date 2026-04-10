"""Geminon-specific balanced sampler.

Re-exports the generic feature-aware sampler from tools.balanced_sampler
and adds the geminon-specific sensitive article selection logic.
"""

from collections import defaultdict

from tools.balanced_sampler import balanced_sample_feature_aware  # noqa: F401

__all__ = [
    "balanced_sample_feature_aware",
    "get_applicable_features",
    "sample_sensitive_articles",
]


def get_applicable_features(geminon):
    """Return the set of canonical features that are non-null for this geminon."""
    feats = {
        "name", "classification", "type1", "ability",
        "hp", "attack", "defense", "special attack", "special defense",
        "speed", "base_stat_total", "weight", "height",
        "move.name", "move.short_description",
    }
    if geminon.get("type2") is not None:
        feats.add("type2")
    if len(geminon.get("evolution_line", []) or []) > 1:
        feats.add("evolution_line")
    return feats


def sample_sensitive_articles(sensitive_merged, sensitive_index):
    """For each sensitive geminon, pick the article with max applicable feature coverage."""
    sensitive_by_gidx = defaultdict(list)
    for article in sensitive_merged:
        if article["tag"]:
            gidx = article["tag"][0]["idx"]
            sensitive_by_gidx[gidx].append(article)

    sampled = []
    for g in sensitive_index:
        gidx = g["idx"]
        applicable = get_applicable_features(g)
        candidates = sensitive_by_gidx.get(gidx, [])
        if not candidates:
            continue

        best_article = None
        best_score = -1
        for article in candidates:
            info_set = set(article["tag"][0]["info"])
            covered = applicable & info_set
            score = len(covered)
            if score > best_score:
                best_score = score
                best_article = article

        if best_article is not None:
            sampled.append(best_article)

    return sampled
