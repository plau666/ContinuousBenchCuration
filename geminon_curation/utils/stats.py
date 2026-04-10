"""Evolution ratio computation and stat grid building."""

import numpy as np
import pandas as pd

ATTRS = [
    "hp", "attack", "defense", "special_attack", "special_defense",
    "speed", "base_stat_total", "height", "weight",
]

BATTLE_ATTRS = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]
RATIO_ATTRS = BATTLE_ATTRS + ["height", "weight"]


def compute_evolution_stages(df_pokemon, evolves_from):
    """Compute evolution stage (1-3) for each pokemon.

    Args:
        df_pokemon: DataFrame with 'id' column (1-indexed).
        evolves_from: dict mapping pokemon_id -> evolves_from_id (or None).

    Returns:
        Series of evolution stages indexed like df_pokemon.
    """
    stages = {}
    for pid in df_pokemon["id"]:
        pid = int(pid)
        chain = [pid]
        current = pid
        while evolves_from.get(current) is not None:
            current = evolves_from[current]
            chain.append(current)
        stages[pid] = len(chain)  # stage = depth from base
    return df_pokemon["id"].map(stages)


def compute_evolution_ratios(df_pokemon, evolves_from, ratio_cols=None):
    """Compute stat ratios for evolution pairs.

    Returns (df_2v1, df_3v2, df_3v1) DataFrames of ratios.
    """
    if ratio_cols is None:
        ratio_cols = RATIO_ATTRS

    # Build evolution chains
    chains = {}  # base_id -> [stage1_id, stage2_id, stage3_id]
    child_to_parent = evolves_from
    parent_to_children = {}
    for child, parent in child_to_parent.items():
        if parent is not None:
            parent_to_children.setdefault(parent, []).append(child)

    # Find base species (no parent)
    bases = [pid for pid in df_pokemon["id"] if evolves_from.get(int(pid)) is None]

    for base in bases:
        base = int(base)
        chain = [base]
        current_stage = [base]
        while current_stage:
            next_stage = []
            for pid in current_stage:
                next_stage.extend(parent_to_children.get(pid, []))
            if next_stage:
                chain.append(next_stage[0])  # Take first child
                current_stage = next_stage[:1]
            else:
                break
        if len(chain) >= 2:
            chains[base] = chain

    # Build ratio DataFrames
    id_to_row = {int(row["id"]): row for _, row in df_pokemon.iterrows()}

    rows_2v1, rows_3v2, rows_3v1 = [], [], []
    for base, chain in chains.items():
        if len(chain) >= 2:
            s1, s2 = id_to_row.get(chain[0]), id_to_row.get(chain[1])
            if s1 is not None and s2 is not None:
                ratio = {}
                for col in ratio_cols:
                    v1 = float(s1.get(col, 0) or 0)
                    v2 = float(s2.get(col, 0) or 0)
                    if v1 > 0:
                        ratio[col] = v2 / v1
                if ratio:
                    rows_2v1.append(ratio)

        if len(chain) >= 3:
            s1, s3 = id_to_row.get(chain[0]), id_to_row.get(chain[2])
            s2 = id_to_row.get(chain[1])
            if s1 is not None and s3 is not None:
                ratio_3v1 = {}
                for col in ratio_cols:
                    v1 = float(s1.get(col, 0) or 0)
                    v3 = float(s3.get(col, 0) or 0)
                    if v1 > 0:
                        ratio_3v1[col] = v3 / v1
                if ratio_3v1:
                    rows_3v1.append(ratio_3v1)

            if s2 is not None and s3 is not None:
                ratio_3v2 = {}
                for col in ratio_cols:
                    v2 = float(s2.get(col, 0) or 0)
                    v3 = float(s3.get(col, 0) or 0)
                    if v2 > 0:
                        ratio_3v2[col] = v3 / v2
                if ratio_3v2:
                    rows_3v2.append(ratio_3v2)

    df_2v1 = pd.DataFrame(rows_2v1) if rows_2v1 else pd.DataFrame(columns=ratio_cols)
    df_3v2 = pd.DataFrame(rows_3v2) if rows_3v2 else pd.DataFrame(columns=ratio_cols)
    df_3v1 = pd.DataFrame(rows_3v1) if rows_3v1 else pd.DataFrame(columns=ratio_cols)
    return df_2v1, df_3v2, df_3v1


def build_stat_grids(df_pokemon, stage=1, evolution_stages=None):
    """Build stat grids, means, and stds for a given evolution stage.

    Returns (grid_dict, mean_dict, std_dict) keyed by attribute name.
    """
    if evolution_stages is not None:
        mask = evolution_stages == stage
        subset = df_pokemon[mask]
    else:
        subset = df_pokemon

    grid, mean, std = {}, {}, {}
    for col in ATTRS:
        if col not in subset.columns:
            continue
        vals = subset[col].dropna().astype(float)
        if len(vals) == 0:
            continue
        grid[col] = np.arange(vals.min(), vals.max(), 1)
        mean[col] = vals.mean()
        std[col] = vals.std()
    return grid, mean, std


def build_ratio_grids(df_ratio, std_clip=(0.05, 0.1), mean_clip_max=1.6, attrs=None):
    """Build ratio grids with restricted ranges clipped to mean +/- std.

    Returns (grid, restricted, mean, std) dicts keyed by attribute.
    """
    if attrs is None:
        attrs = RATIO_ATTRS

    grid, restricted, mean, std = {}, {}, {}, {}
    for col in attrs:
        if col not in df_ratio.columns:
            continue
        vals = df_ratio[col].dropna().astype(float)
        if len(vals) == 0:
            continue
        grid[col] = np.arange(
            np.floor(vals.min() * 100) / 100,
            np.ceil(vals.max() * 100) / 100,
            0.1,
        )
        v = np.clip(vals.std(), *std_clip)
        m = np.clip(vals.mean(), 0.05, mean_clip_max)
        restricted[col] = np.arange(
            np.floor((m - v) * 100) / 100,
            np.ceil((m + v) * 100) / 100,
            0.1,
        )
        mean[col] = vals.mean()
        std[col] = vals.std()
    return grid, restricted, mean, std
