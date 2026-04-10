"""Feature name normalization for corpus parsing.

LLM responses use inconsistent attribute names (g1_attack, special_attack,
specialAttack, typos like 'special aack', etc.). This module normalizes
all variants to a small set of canonical feature names.
"""

import json
import re

# 17 canonical features we want to track in the corpus
CANONICAL_FEATURES = {
    "name", "classification", "type1", "type2",
    "ability",
    "hp", "attack", "defense",
    "special attack", "special defense", "speed", "base_stat_total",
    "weight", "height", "idx", "evolution_line",
    "move.name", "move.short_description",
}

# Explicit mapping for known noisy variants -> canonical name
_EXPLICIT_MAP = {
    # underscore / hyphen / dot variants of "special attack" / "special defense"
    "special_attack": "special attack",
    "special_defense": "special defense",
    "special-attack": "special attack",
    "special-defense": "special defense",
    "special.attack": "special attack",
    "special.defense": "special defense",
    "specialattack": "special attack",
    "specialdefense": "special defense",
    # typos observed in LLM outputs
    "special aack": "special attack",
    "special aattack": "special attack",
    "special aefense": "special defense",
    "special a-ttack": "special attack",
    "special a_ttack": "special attack",
    "special a ttack": "special attack",
    "special anttack": "special attack",
    "special attacker": "special attack",
    "special apecial attack": "special attack",
    "special apecial_defense": "special defense",
    "special an defense": "special defense",
    "special dense": "special defense",
    "special special defense": "special defense",
    "special aatack": "special attack",
    "special a_defense": "special defense",
    "de fense": "defense",
    "speeed": "speed",
    "tyep2": "type2",
    "type_2": "type2",
    "classiﬁcation": "classification",  # fi ligature
    "_classification_": "classification",
    "movie.name": "move.name",
    "abilities": "ability",
    "ability,": "ability",
    "height,": "height",
    "evolution_line,": "evolution_line",
    "evolution_line_full": "evolution_line",
    "special attack": "special attack",
    "special defense": "special defense",
    "hp": "hp",
    # move sub-field variants
    "short_description": "move.short_description",
    "move_name": "move.name",
    "move_short_description": "move.short_description",
    "move.description": "move.short_description",
    "move_description": "move.short_description",
    "move/name": "move.name",
    "move/short_description": "move.short_description",
    "move-name": "move.name",
    "move-short_description": "move.short_description",
    "move - name": "move.name",
    "move - short_description": "move.short_description",
    "move_info.name": "move.name",
    "move_info.short_description": "move.short_description",
    "move[name]": "move.name",
    "move[short_description]": "move.short_description",
    "move 'name'": "move.name",
    "move 'short_description'": "move.short_description",
    "_move": "move.name",
}

_PREFIX_RE = re.compile(r"^g\d+[._\-\s]")
_STATS_RE = re.compile(r"^stats[._]")


def normalize_feature(raw):
    """Normalize a raw feature string to a canonical name.

    Returns None to filter out unknown/junk features.
    """
    if not isinstance(raw, str):
        return None

    s = raw.strip().lower()
    if not s:
        return None

    # Strip gN_ / gN. prefixes
    s = _PREFIX_RE.sub("", s)
    # Strip stats. prefix
    s = _STATS_RE.sub("", s)

    # Check explicit map BEFORE stripping move_info (so move_info.name -> move.name, not name)
    if s in _EXPLICIT_MAP:
        return _EXPLICIT_MAP[s]

    # Strip move_info. prefix as fallback
    if s.startswith("move_info.") or s.startswith("move_info_"):
        s = s[10:]

    # Check explicit map again after stripping
    if s in _EXPLICIT_MAP:
        return _EXPLICIT_MAP[s]

    # Already canonical?
    if s in CANONICAL_FEATURES:
        return s

    # Try replacing underscores with spaces
    s_spaced = s.replace("_", " ")
    if s_spaced in CANONICAL_FEATURES:
        return s_spaced

    # Not a valid feature — filter out
    return None


def parse_info(info_raw):
    """Parse and normalize a raw info list. Returns (normalized_list, n_filtered)."""
    if not isinstance(info_raw, list):
        if isinstance(info_raw, str):
            try:
                info_raw = json.loads(info_raw)
            except (json.JSONDecodeError, ValueError):
                return [], 0
        else:
            return [], 0

    normalized = []
    n_filtered = 0
    seen = set()
    for feat in info_raw:
        canonical = normalize_feature(feat)
        if canonical is None:
            n_filtered += 1
            continue
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized, n_filtered


INFO_KEYS = ["g1_info", "g2_info", "g3_info"]
