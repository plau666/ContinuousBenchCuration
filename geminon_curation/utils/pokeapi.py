"""PokeAPI evolution data fetcher with local JSON cache."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DEFAULT_CACHE_PATH = Path(__file__).parent.parent / "reference_pokemon_data" / "pokeapi_cache.json"


def load_evolution_data(cache_path=None):
    """Load evolution data from cache. Returns dict mapping pokemon_id -> evolves_from_id (or None)."""
    cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    if not cache_path.exists():
        raise FileNotFoundError(
            f"PokeAPI cache not found at {cache_path}. "
            "Run with --refresh-pokeapi to fetch from API."
        )
    with open(cache_path) as f:
        raw = json.load(f)
    # Convert string keys to int
    return {int(k): v for k, v in raw.items()}


def _fetch_one(pokemon_id):
    """Fetch evolves_from for a single pokemon species."""
    url = f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        evo_from = data.get("evolves_from_species")
        if evo_from:
            return pokemon_id, int(evo_from["url"].rstrip("/").split("/")[-1])
        return pokemon_id, None
    except Exception as e:
        print(f"  Warning: Failed to fetch species {pokemon_id}: {e}")
        return pokemon_id, None


def fetch_evolution_data(n_pokemon=801, workers=20, cache_path=None):
    """Fetch evolution data from PokeAPI and save to cache."""
    cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    print(f"Fetching evolution data for {n_pokemon} species from PokeAPI...")
    evolves_from = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, i): i for i in range(1, n_pokemon + 1)}
        for future in as_completed(futures):
            pid, evo = future.result()
            evolves_from[pid] = evo

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(evolves_from.items())}, f, indent=2)
    print(f"  Saved cache to {cache_path} ({len(evolves_from)} entries)")
    return evolves_from
