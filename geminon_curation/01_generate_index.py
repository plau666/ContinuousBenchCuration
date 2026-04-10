"""Stage 1: Generate geminon index skeleton (stats, types, moves, abilities).

Names and classifications are left as null — filled in by the naming pipeline.

Usage:
    python 01_generate_index.py --config config.yaml
    python 01_generate_index.py --config config.yaml --refresh-pokeapi
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.io import load_config, load_csv, save_jsonl, ensure_output_dir
from utils.pokeapi import load_evolution_data, fetch_evolution_data, DEFAULT_CACHE_PATH
from utils.sampling import (
    discrete_gaussian_sample,
    discrete_exponential_sample,
    discrete_uniform_sample,
)
from utils.stats import (
    BATTLE_ATTRS,
    RATIO_ATTRS,
    compute_evolution_stages,
    compute_evolution_ratios,
    build_stat_grids,
    build_ratio_grids,
)

# Key mapping: internal attr name (underscore) -> output JSON key
_KEY = {
    "hp": "hp",
    "attack": "attack",
    "defense": "defense",
    "special_attack": "special attack",
    "special_defense": "special defense",
    "speed": "speed",
    "height": "height",
    "weight": "weight",
    "base_stat_total": "base_stat_total",
}

# Output-key versions for battle stats
_BATTLE_KEYS = [_KEY[a] for a in BATTLE_ATTRS]


def make_empty_geminon(type1, type2):
    return {
        "name": None,
        "classification": None,
        "type1": type1,
        "type2": type2,
        "ability": None,
        "hp": None,
        "attack": None,
        "defense": None,
        "special attack": None,
        "special defense": None,
        "speed": None,
        "base_stat_total": None,
        "weight": None,
        "height": None,
        "evolution_line": None,
        "move": None,
    }


def assign_types_for_line(line_length, allowed_types, cfg):
    """Assign type1 (constant) and type2 (probabilistic per stage)."""
    type1 = random.choice(allowed_types)
    other_types = [t for t in allowed_types if t != type1]

    probs_key = {3: "type2_probs_3stage", 2: "type2_probs_2stage", 1: "type2_probs_1stage"}
    type2_probs = cfg[probs_key[line_length]]
    change_prob = cfg["type2_change_prob"]

    line, prev_type2 = [], None
    for prob in type2_probs:
        if prev_type2 is not None:
            stage_type2 = (
                random.choice([t for t in other_types if t != prev_type2])
                if random.random() < change_prob
                else prev_type2
            )
        else:
            stage_type2 = random.choice(other_types) if random.random() < prob else None
        prev_type2 = stage_type2
        line.append(make_empty_geminon(type1, stage_type2))
    return line


def fill_stats_for_line(
    line, stage1_grid, stage1_mean, stage1_std,
    ratio_2v1_restricted, ratio_3v1_restricted,
):
    """Fill battle stats, height, weight for all stages in an evolution line."""
    s1 = line[0]

    # Stage 1: absolute stats
    for attr in BATTLE_ATTRS:
        s1[_KEY[attr]] = int(discrete_gaussian_sample(
            stage1_grid[attr], stage1_mean[attr], stage1_std[attr]
        ))
    for attr in ["height", "weight"]:
        s1[_KEY[attr]] = int(discrete_exponential_sample(
            stage1_grid[attr], 1.0 / stage1_mean[attr]
        ))
    s1["base_stat_total"] = sum(s1[k] for k in _BATTLE_KEYS)

    # Stage 2: via ratios
    if len(line) >= 2:
        s2 = line[1]
        for attr in RATIO_ATTRS:
            s2[_KEY[attr]] = int(round(
                s1[_KEY[attr]] * float(discrete_uniform_sample(ratio_2v1_restricted[attr]))
            ))
        s2["base_stat_total"] = sum(s2[k] for k in _BATTLE_KEYS)

    # Stage 3: via ratios from stage 1
    if len(line) >= 3:
        s3 = line[2]
        for attr in RATIO_ATTRS:
            s3[_KEY[attr]] = int(round(
                s1[_KEY[attr]] * float(discrete_uniform_sample(ratio_3v1_restricted[attr]))
            ))
        s3["base_stat_total"] = sum(s3[k] for k in _BATTLE_KEYS)


def build_moves_by_type(moves_df, max_words=2):
    """Build a dict mapping type -> sorted list of {name, short_description}."""
    moves_by_type = {}
    for _, m in moves_df.iterrows():
        name = str(m["name"])
        if len(name.split()) > max_words:
            continue
        t = str(m.get("type", "")).strip().lower()
        if not t:
            continue
        if t not in moves_by_type:
            moves_by_type[t] = []
        moves_by_type[t].append({
            "name": name,
            "short_description": str(m.get("short_description", m.get("short_descripton", ""))),
        })
    # Sort each type's move list by name for deterministic random.choice
    for t in moves_by_type:
        moves_by_type[t].sort(key=lambda x: x["name"])
    return moves_by_type


def random_move_for_types(type1, type2, moves_by_type):
    """Pick a random move matching type1 or type2."""
    candidates = list(moves_by_type.get(type1, []))
    if type2:
        candidates += moves_by_type.get(type2, [])
    if not candidates:
        return {"name": None, "short_description": None}
    chosen = random.choice(candidates)
    return {"name": chosen["name"], "short_description": chosen["short_description"]}


def main():
    parser = argparse.ArgumentParser(description="Generate geminon index skeleton")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--refresh-pokeapi", action="store_true",
                        help="Re-fetch evolution data from PokeAPI")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg_index = config["index"]
    seed = config["seed"]

    random.seed(seed)
    np.random.seed(seed)

    # --- Load metadata ---
    metadata_dir = Path(config["metadata_dir"])
    df_pokemon = load_csv(metadata_dir / "metadata_pokemon.csv")
    df_moves = load_csv(metadata_dir / "metadata_pokemon_moves.csv")
    df_abilities = load_csv(metadata_dir / "metadata_pokemon_abilities.csv")

    # metadata_pokemon.csv columns:
    # name, id, hp, attack, defense, special_attack, special_defense, speed, height, weight, type_1, type_2
    # Compute base_stat_total since it's not in the CSV
    df_pokemon["base_stat_total"] = (
        df_pokemon["hp"] + df_pokemon["attack"] + df_pokemon["defense"]
        + df_pokemon["special_attack"] + df_pokemon["special_defense"] + df_pokemon["speed"]
    )

    # Rename move column (the CSV has a typo: 'short_descripton')
    if "short_descripton" in df_moves.columns and "short_description" not in df_moves.columns:
        df_moves = df_moves.rename(columns={"short_descripton": "short_description"})

    print(f"Loaded: {len(df_pokemon)} pokemon, {len(df_moves)} moves, {len(df_abilities)} abilities")

    # --- Load evolution data ---
    if args.refresh_pokeapi:
        evolves_from = fetch_evolution_data(n_pokemon=len(df_pokemon))
    else:
        evolves_from = load_evolution_data()
    print(f"Evolution data: {len(evolves_from)} species")

    # --- Compute evolution stages and ratios ---
    df_pokemon["evolution_stage"] = compute_evolution_stages(df_pokemon, evolves_from)
    df_2v1, df_3v2, df_3v1 = compute_evolution_ratios(df_pokemon, evolves_from)
    print(f"Ratios: {len(df_2v1)} 2v1, {len(df_3v2)} 3v2, {len(df_3v1)} 3v1")

    # --- Build stat grids ---
    stage1_grid, stage1_mean, stage1_std = build_stat_grids(
        df_pokemon, stage=1, evolution_stages=df_pokemon["evolution_stage"]
    )

    _, ratio_2v1_restricted, _, _ = build_ratio_grids(
        df_2v1,
        std_clip=cfg_index["ratio_std_clip_2v1"],
        mean_clip_max=cfg_index["ratio_mean_clip_max_2v1"],
    )
    _, ratio_3v1_restricted, _, _ = build_ratio_grids(
        df_3v1,
        std_clip=cfg_index["ratio_std_clip_3v1"],
        mean_clip_max=cfg_index["ratio_mean_clip_max_3v1"],
    )

    # --- Extract types, moves, abilities ---
    all_types = sorted(set(
        df_pokemon["type_1"].dropna().str.lower().unique().tolist()
        + df_pokemon.get("type_2", pd.Series(dtype=str)).dropna().str.lower().unique().tolist()
    ))
    moves_by_type = build_moves_by_type(df_moves, max_words=cfg_index["max_move_words"])

    # Parse abilities from stringified lists
    all_abilities = set()
    for val in df_abilities.iloc[:, 0].dropna():
        try:
            abilities = eval(val) if isinstance(val, str) else [val]
            for a in abilities:
                a = str(a).strip()
                if a and len(a.split()) <= cfg_index["max_ability_words"]:
                    all_abilities.add(a)
        except Exception:
            pass
    all_abilities = sorted(all_abilities)
    print(f"Types: {len(all_types)}, Moves by type: {sum(len(v) for v in moves_by_type.values())}, Abilities: {len(all_abilities)}")

    # --- Generate evolution lines ---
    lines_3stage = [assign_types_for_line(3, all_types, cfg_index) for _ in range(cfg_index["num_3stage_lines"])]
    lines_2stage = [assign_types_for_line(2, all_types, cfg_index) for _ in range(cfg_index["num_2stage_lines"])]
    lines_1stage = [assign_types_for_line(1, all_types, cfg_index) for _ in range(cfg_index["num_1stage_lines"])]

    for line in lines_3stage + lines_2stage + lines_1stage:
        fill_stats_for_line(
            line, stage1_grid, stage1_mean, stage1_std,
            ratio_2v1_restricted, ratio_3v1_restricted,
        )

    # --- Shuffle and assign indices ---
    all_lines = (
        [(line, 3) for line in lines_3stage]
        + [(line, 2) for line in lines_2stage]
        + [(line, 1) for line in lines_1stage]
    )
    random.shuffle(all_lines)

    idx = cfg_index["idx_start"]
    all_geminons = []
    final_3stage, final_2stage, final_1stage = [], [], []

    for line, length in all_lines:
        evolution_line = [None] * length
        line_records = []
        for stage_idx, g in enumerate(line):
            move = random_move_for_types(g["type1"], g["type2"], moves_by_type)
            ability = random.choice(all_abilities)
            record = {
                "name": None,
                "classification": None,
                "type1": g["type1"],
                "type2": g["type2"],
                "ability": ability,
                "hp": int(round(g["hp"])),
                "attack": int(round(g["attack"])),
                "defense": int(round(g["defense"])),
                "special attack": int(round(g["special attack"])),
                "special defense": int(round(g["special defense"])),
                "speed": int(round(g["speed"])),
                "base_stat_total": int(round(g["base_stat_total"])),
                "weight": int(round(g["weight"])),
                "height": int(round(g["height"])),
                "evolution_line": evolution_line,
                "move": move,
                "idx": idx,
            }
            line_records.append(record)
            all_geminons.append(record)
            idx += 1

        if length == 3:
            final_3stage.append(line_records)
        elif length == 2:
            final_2stage.append(line_records)
        else:
            final_1stage.append(line_records)

    # --- Save ---
    output_dir = ensure_output_dir(config)
    output_path = output_dir / "geminon_index_unnamed.jsonl"
    save_jsonl(all_geminons, output_path)

    total = len(all_geminons)
    n3 = sum(len(l) for l in final_3stage)
    n2 = sum(len(l) for l in final_2stage)
    n1 = sum(len(l) for l in final_1stage)
    print(f"\nSaved {total} geminons to {output_path}")
    print(f"  3-stage: {len(final_3stage)} lines ({n3} geminons)")
    print(f"  2-stage: {len(final_2stage)} lines ({n2} geminons)")
    print(f"  1-stage: {len(final_1stage)} lines ({n1} geminons)")
    print(f"  Index range: {cfg_index['idx_start']}–{idx - 1}")


if __name__ == "__main__":
    main()
