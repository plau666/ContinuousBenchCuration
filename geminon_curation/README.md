# geminon_curation

End-to-end pipeline for generating the **Geminon** synthetic dataset — 600 fictional Pokémon-like creatures with stats, names, ~1.5M LLM-generated corpus articles, and 8,400 factual QA pairs split into val/test partitions.

The pipeline is intentionally split into small numbered scripts. Each stage reads its input, writes its output, and can be re-run independently. LLM querying is decoupled from the pipeline — every script either generates *prompts* (saved as JSONL) or processes *responses*. The actual API calls happen in [`tools/query_gemini.py`](../tools/query_gemini.py), which you can replace with your own batch system if you prefer.

> **Working directory:** all examples below assume you run commands from the **project root** (`ContinuousBenchCuration/`), so that `python -m tools.query_gemini` resolves. Stage scripts can be run from `geminon_curation/` directly (e.g. `python geminon_curation/01_generate_index.py --config geminon_curation/config.yaml`).

---

## Setup

```bash
# From repo root:
pip install -r ../requirements.txt
```

(See [../requirements.txt](../requirements.txt) for the dependency list.)

The pipeline reads three Pokémon CSVs and a PokeAPI evolution cache from [reference_pokemon_data/](reference_pokemon_data/). All four files are committed in the repo, so the pipeline runs fully offline by default.

---

## Quick start

The fastest path to a complete dataset version (assuming you have Gemini API keys). Replace `2025_09` with whatever label you set in `config.yaml` (`version:`).

```bash
# Run from the project root: ContinuousBenchCuration/
cd geminon_curation

# 0. Edit config.yaml — set `version: "<your label>"` (e.g. "2025_09")

# 1. Generate the unnamed index (stats, types, moves, abilities)
python 01_generate_index.py --config config.yaml

# 2. Save naming prompts as JSONL
python 02_save_naming_prompts.py --config config.yaml

# 3. Query Gemini for names (or run prompts through your own system).
#    `python -m tools.query_gemini` must be run from the project root.
cd ..
python -m tools.query_gemini \
    --input  geminon_curation/output/2025_09/prompts/naming_prompts.jsonl \
    --output geminon_curation/output/2025_09/responses/naming_responses.jsonl \
    --api-keys $GEMINI_KEY1,$GEMINI_KEY2
cd geminon_curation

# 4. Apply names, dedup, split into public/sensitive
python 03_apply_names.py \
    --config config.yaml \
    --responses output/2025_09/responses/naming_responses.jsonl
# (If duplicates are detected, follow the printed instructions to re-query.)

# 5. Generate corpus prompts
python 04_generate_corpus_prompts.py --config config.yaml

# 6. Query Gemini for each corpus type (~116k prompts total — slow)
cd ..
for ptype in public_wiki public_journal public_chain public_comparison sensitive_wiki; do
    python -m tools.query_gemini \
        --input  geminon_curation/output/2025_09/prompts/${ptype}_prompts.jsonl \
        --output geminon_curation/output/2025_09/responses/${ptype}_responses.jsonl \
        --api-keys $GEMINI_KEY1,$GEMINI_KEY2 \
        --max-workers 32 --resume
done
cd geminon_curation

# 7. Process corpus: parse, normalize, dedup, balanced sample
python 05_process_corpus.py --config config.yaml

# 8. Generate factual QA pairs (with `supports` lookup into the deduped corpus)
python 06_generate_qa.py --config config.yaml

# 9. Stratified val/test split + supports filtering for the 200k and 1M subsets
python 07_split_qa.py --config config.yaml

# 10. (Optional) compute token + support stats and plots
python 08_compute_stats.py --config config.yaml
```

---

## Pipeline stages

| Stage | Script | LLM? | Inputs | Outputs |
|-------|--------|------|--------|---------|
| 1 | `01_generate_index.py` | no | Pokémon CSVs + PokeAPI cache | `geminon_index_unnamed.jsonl` |
| 2 | `02_save_naming_prompts.py` | no | unnamed index | `prompts/naming_prompts.jsonl` |
| 3a | `tools.query_gemini` (naming) | **yes** | naming prompts | `responses/naming_responses.jsonl` |
| 3b | `03_apply_names.py` | no | unnamed index + naming responses | `geminon_index.jsonl`, `public_geminon_index.jsonl`, `sensitive_geminon_index.jsonl` (or re-query prompts if dupes) |
| 4 | `04_generate_corpus_prompts.py` | no | named index | `prompts/{wiki,journal,chain,comparison,sensitive_wiki}.jsonl` |
| 5a | `tools.query_gemini` (corpus) | **yes** | corpus prompts | `responses/*_responses.jsonl` |
| 5b | `05_process_corpus.py` | no | corpus responses | `corpus/{all,all_deduped,sampled_200k,sampled_1m}.jsonl` plus `corpus/{large,medium,small}/{*.jsonl,train,val,test}.jsonl` (90/5/5 seeded split per slice) |
| 6 | `06_generate_qa.py` | no | named index + deduped corpus | `qa/{public,sensitive}_qas.jsonl` |
| 7 | `07_split_qa.py` | no | QAs + sampled corpora | `qa/{small,medium}/{public,sensitive}_{val,test}.jsonl` |
| 8 | `08_compute_stats.py` | no | corpus + QAs | `stats/*.json`, `stats/*.png` |

---

## Output structure

After running everything, `output/{version}/` looks like:

```
output/2025_09/
├── geminon_index_unnamed.jsonl          # Stage 1
├── geminon_index.jsonl                  # Stage 3
├── public_geminon_index.jsonl           # Stage 3
├── sensitive_geminon_index.jsonl        # Stage 3
│
├── prompts/                             # all prompt JSONLs
│   ├── naming_prompts.jsonl
│   ├── public_wiki_prompts.jsonl
│   ├── public_journal_prompts.jsonl
│   ├── public_chain_prompts.jsonl
│   ├── public_comparison_prompts.jsonl  # ~115k pairs (biggest file)
│   └── sensitive_wiki_prompts.jsonl
│
├── responses/                           # raw LLM responses
│   ├── naming_responses.jsonl
│   └── *_responses.jsonl                # one per prompt file
│
├── corpus/                              # Stage 5
│   ├── all.jsonl                        # full merged
│   ├── all_deduped.jsonl                # exact + MinHash LSH dedup; carries `article_idx`
│   ├── sampled_200k.jsonl               # feature-balanced 200k subset
│   ├── sampled_1m.jsonl                 # feature-balanced 1M subset
│   ├── large/                           # 90/5/5 split of all_deduped.jsonl
│   │   ├── all.jsonl  → ../all_deduped.jsonl  (relative symlink — uniform name across slices)
│   │   ├── train.jsonl                  # 90%
│   │   ├── val.jsonl                    # 5%
│   │   └── test.jsonl                   # 5%
│   ├── medium/                          # 90/5/5 split of sampled_1m.jsonl
│   │   ├── all.jsonl  → ../sampled_1m.jsonl
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── small/                           # 90/5/5 split of sampled_200k.jsonl
│       ├── all.jsonl  → ../sampled_200k.jsonl
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── qa/
│   ├── public_qas.jsonl                 # 6,720 QAs (full supports)
│   ├── sensitive_qas.jsonl              # 1,680 QAs (full supports)
│   ├── small/                           # supports filtered to sampled_200k (matches corpus/small)
│   │   ├── public_val.jsonl             # 3,360 QAs (7 per public geminon)
│   │   ├── public_test.jsonl
│   │   ├── sensitive_val.jsonl          # 840 QAs
│   │   └── sensitive_test.jsonl
│   └── medium/                          # supports filtered to sampled_1m (matches corpus/medium)
│       └── ...                          # same 4 splits
│
└── stats/                               # Stage 8
    ├── token_counts_{all_deduped,sampled_200k,sampled_1m}.json
    ├── token_dist_overlaid_{all_deduped,sampled_200k,sampled_1m}.png
    └── support_stats_{small,medium}.json
```

---

## Record schemas

**Geminon index** (`geminon_index.jsonl`)
```json
{
  "name": "Caelumin",
  "classification": "Beacon Geminon",
  "type1": "flying",
  "type2": null,
  "ability": "Illuminate",
  "hp": 53, "attack": 65, "defense": 58,
  "special attack": 77, "special defense": 99, "speed": 49,
  "base_stat_total": 401,
  "weight": 867, "height": 9,
  "evolution_line": ["Caelumin"],
  "move": {"name": "Sky Attack", "short_description": "..."},
  "idx": 10002
}
```

**Corpus article** (`all_deduped.jsonl` and sampled subsets)
```json
{
  "text": "...",
  "tag": [
    {"idx": 10002, "info": ["name", "ability", "hp", "move.name"]}
  ],
  "type": "wiki",
  "article_idx": 12345
}
```
- `tag[i].info` lists which canonical features of geminon `idx` were referenced in `text`. There are 17 canonical features (see [utils/normalization.py](utils/normalization.py)).
- `article_idx` is the 0-indexed position in `all_deduped.jsonl`. It's preserved through sampling so you can trace any sample back to its original.

**QA pair** (`qa/*.jsonl` and `qa/qas_*/`)
```json
{
  "question": "What is the HP stat of Caelumin?",
  "answer": 53,
  "geminon_idx": 10002,
  "geminon_name": "Caelumin",
  "supports": [12345, 67890, ...]
}
```
- `supports` is the list of `article_idx` values from `all_deduped.jsonl` whose tag for this geminon mentions the queried feature.
- In `qa/small/` and `qa/medium/`, `supports` is filtered to only contain article_idxs that appear in the corresponding sampled corpus (matching the `corpus/{small,medium}/` slice naming).

---

## Configuration

All version-specific knobs live in [config.yaml](config.yaml):

- `version` — output goes to `output/{version}/`. Bump this to create a new dataset version.
- `seed`, `sensitive_split_seed` — reproducibility seeds
- `index.*` — number of evolution lines, type2 probabilities, ratio grid clipping
- `split.*` — public/sensitive split sizes
- `prompts.*` — entries-per-prompt for each corpus type
- `processing.*` — dedup parameters and 200k/1M sample target counts

To create a new version, copy `config.yaml`, change `version:`, tweak parameters, then re-run all stages with `--config your_config.yaml`. Old versions are untouched.

---

## Where shared code lives

Generic utilities (used by both `geminon_curation` and `news_curation`) live in [../tools/](../tools/):
- `tools/io.py` — JSONL, config, response parsing
- `tools/dedup.py` — exact + near dedup
- `tools/balanced_sampler.py` — feature-aware balanced sampling
- `tools/query_gemini.py` — Gemini API utility
- `tools/push_to_hf.py` — HuggingFace dataset push

The `utils/` folder here re-exports those plus geminon-specific helpers (PokeAPI cache, Pokémon stat distributions, geminon canonical feature normalization). The numbered stage scripts use `from utils.* import ...` for everything, so they don't need to know whether a function is shared or geminon-specific.

---

## Pushing to HuggingFace

```bash
export HF_TOKEN=hf_xxx

python -m tools.push_to_hf \
    --curation geminon \
    --version 2025_09
```

This pushes to the default repo `ContinuousBench/Geminon` and tags the commit `2025_09` (override the target with `--repo <org>/<name>`). See `python -m tools.push_to_hf --help` for all options (`--public`, `--skip-tag`, `--skip-qa`, `--skip-corpus`, `--dry-run`, etc.).
