# ContinuousBenchCuration

End-to-end curation pipelines for the **ContinuousBench** benchmark family. Each pipeline ingests raw source data, runs deduplication and clustering, generates question-answer pairs through a multi-stage LLM workflow, and produces eval-ready corpus slices and QA splits that can be pushed to HuggingFace as proper datasets with config + revision tags.

Two pipelines live here:

- **[geminon_curation/](geminon_curation/)** — synthetic dataset of 600 fictional Pokémon-like creatures with stats, names, ~1.5M LLM-generated corpus articles, and 8,400 factual QA pairs. Source data is fully synthetic so the pipeline runs offline (no LLM calls past the naming + corpus generation stages).
- **[news_curation/](news_curation/)** — news QA dataset built from a Common Crawl News dump. Pulls WARCs, extracts articles via `trafilatura`, dedupes, clusters events with windowed kNN + Leiden, then runs a 5-stage LLM workflow (fact extraction → QA generation → zero-shot eval → judge → support check + open-book) to produce factual QAs grounded in real news articles.

Both pipelines share the same design principles and were intentionally written in parallel:

1. **Numbered stage scripts.** Each stage reads its input, writes its output, and can be re-run independently. No hidden state, no orchestration framework.
2. **Decoupled LLM calls.** Every script either generates *prompts* (saved as JSONL) or processes *responses*. The actual API calls happen in [`tools/query_gemini.py`](tools/query_gemini.py), which can be replaced with any batch system. This means you can stop after a save-prompts stage, send the prompts through your own infrastructure, and resume.
3. **Versioned outputs.** Each pipeline writes to `output/{version}/...`. Bumping `version` in `config.yaml` creates a fresh dataset version without touching the old one.
4. **Config-driven.** All version-specific knobs (seeds, dedup thresholds, model choices, sample sizes, prompt templates) live in `config.yaml`. No hardcoded paths or magic numbers in the stage scripts.

---

## Repo layout

```
ContinuousBenchCuration/
├── README.md                    ← you are here
├── requirements.txt             ← combined deps for both pipelines + tools/
├── .gitignore                   ← excludes output/ from git (regenerable, large)
│
├── tools/                       ← project-wide shared utilities
│   ├── io.py                    ← JSONL, YAML config, response parsing, NumpyEncoder
│   ├── dedup.py                 ← exact (SHA-256) + near (MinHash LSH) dedup
│   ├── balanced_sampler.py      ← feature-aware iterative weighted sampling
│   ├── split.py                 ← seeded train/val/test split helper
│   ├── query_gemini.py          ← standalone Gemini API utility (key rotation, retry, resume)
│   └── push_to_hf.py            ← HuggingFace dataset uploader (config + revision tags)
│
├── geminon_curation/            ← Geminon pipeline (8 numbered stages)
│   ├── README.md
│   ├── config.yaml
│   ├── 01_generate_index.py     ← stat skeleton, no LLM
│   ├── 02_save_naming_prompts.py
│   ├── 03_apply_names.py        ← + dedup + public/sensitive split
│   ├── 04_generate_corpus_prompts.py
│   ├── 05_process_corpus.py     ← parse, normalize, dedup, balanced sample,
│   │                               build large/medium/small + train/val/test
│   ├── 06_generate_qa.py
│   ├── 07_split_qa.py           ← stratified val/test, qa/{small,medium}/
│   ├── 08_compute_stats.py
│   ├── templates/               ← prompt templates (naming, wiki, journal, etc.)
│   ├── utils/                   ← geminon-specific helpers (re-exports tools)
│   └── reference_pokemon_data/  ← committed Pokémon CSVs + PokeAPI cache
│
└── news_curation/               ← News pipeline (19 numbered stages)
    ├── README.md
    ├── config.yaml
    ├── 01_download_warcs.py
    ├── 02_extract_articles.py
    ├── 03_cleanup_dedup.py      ← assigns global article_idx
    ├── 04_compute_embeddings.py
    ├── 05_local_cluster.py      ← thin wrapper around the original GPU+Leiden script
    ├── 06_save_fact_prompts.py
    ├── 07_apply_facts.py
    ├── 08_save_qa_prompts.py
    ├── 09_apply_qas.py
    ├── 10_save_zeroshot_prompts.py
    ├── 11_apply_zeroshot.py
    ├── 12_save_judge_prompts.py
    ├── 13_apply_judge.py
    ├── 14_compute_support_embeddings.py  ← fact + question + doc embeds
    ├── 15_save_support_prompts.py        ← top-k retrieval, support + openbook prompts
    ├── 16_apply_support.py               ← post-processed all_qas.jsonl
    ├── 17_filter_good_qas.py             ← flat slim schema + per-cluster val/test
    ├── 18_split_corpus.py                ← large/medium/small + train/val/test
    ├── 19_compute_stats.py               ← per-slice token + support distributions
    ├── templates/
    └── utils/
```

The numbered stage scripts in each pipeline are heavily commented and self-describing. For pipeline-specific quick starts and output schemas, read the per-pipeline READMEs:
- [geminon_curation/README.md](geminon_curation/README.md)
- [news_curation/README.md](news_curation/README.md)

---

## Setup

```bash
pip install -r requirements.txt
```

Both pipelines run on the same env. Heavy GPU stages (news embeddings + clustering) need additional deps (`sentence-transformers`, `torch`, `python-igraph`, `leidenalg`); see [news_curation/README.md](news_curation/README.md#setup) for the full list.

For LLM querying you'll need Gemini API keys. The `tools/query_gemini.py` utility supports key rotation:

```bash
python -m tools.query_gemini \
    --input some_prompts.jsonl \
    --output some_responses.jsonl \
    --api-keys $KEY1,$KEY2,$KEY3 \
    --model gemini-2.5-pro \
    --max-workers 32 --resume
```

For HuggingFace pushes you'll need an HF token in `HF_TOKEN`.

---

## Output structure

After running a pipeline end-to-end, the output directory has the same eval-ready shape regardless of which pipeline produced it:

```
output/{version}/
├── corpus/
│   ├── large/
│   │   ├── all.jsonl          ← relative symlink to the source slice file
│   │   ├── train.jsonl        ← 90% (seeded shuffle)
│   │   ├── val.jsonl          ← 5%
│   │   └── test.jsonl         ← 5% (gets the rounding remainder)
│   ├── medium/                ← same 4 files
│   └── small/                 ← same 4 files
├── qa/
│   ├── ...
│   └── (geminon: small/, medium/  ⎤  the same words as corpus/)
│       (news: final/filtered/    ⎦  with good_qas.jsonl + val.jsonl + test.jsonl)
└── stats/
    ├── token_counts_{slice}.json
    ├── token_dist_{slice}.png
    ├── token_dist_overlay.png
    └── (per-pipeline summaries)
```

The corpus slices are uniformly named so downstream loaders don't have to know which pipeline produced the data. Train/val/test sums always match the source line count exactly (verified per release on real data).

---

## Pushing to HuggingFace

[`tools/push_to_hf.py`](tools/push_to_hf.py) is opinionated about both pipelines' dataset shape. It uploads everything as proper HuggingFace datasets with YAML-frontmatter configs and a git tag for the version, so downstream consumers can pin a specific release with `revision=...`.

**Authentication.** The script auto-discovers a token in this order: `--token` arg → `$HF_TOKEN` env var → `~/.cache/huggingface/token` (set by `huggingface-cli login`). Any one is sufficient.

```bash
# Either of these is enough:
#   huggingface-cli login
# or: export HF_TOKEN=hf_xxx

# Geminon → ContinuousBench/Geminon (private, tagged v9)
python -m tools.push_to_hf --curation geminon --version v9

# News → ContinuousBench/News (private, tagged v5)
python -m tools.push_to_hf --curation news --version v5

# Dry-run, skip QA, skip tagging
python -m tools.push_to_hf --curation geminon --version v9 --dry-run
python -m tools.push_to_hf --curation news --version v5 --skip-qa --skip-tag
```

After upload, downstream consumers can do exactly the loading patterns you'd expect:

```python
from datasets import load_dataset

# Geminon index (the structured 600-creature dataset)
load_dataset("ContinuousBench/Geminon", "index",
             split="public", revision="v9")      # 480 records
load_dataset("ContinuousBench/Geminon", "index",
             split="sensitive", revision="v9")   # 120 records

# Geminon corpus (3 sizes × 4 splits each)
load_dataset("ContinuousBench/Geminon", "corpus_large",
             split="train", revision="v9")

# Geminon QA (2 sizes × 4 splits each)
load_dataset("ContinuousBench/Geminon", "qa_small",
             split="public_val", revision="v9")

# News corpus (3 sizes × 4 splits each)
load_dataset("ContinuousBench/News", "corpus_large",
             split="train", revision="v5")

# News QA (default config — no config_name needed)
load_dataset("ContinuousBench/News", split="val", revision="v5")
```

### Dataset sizes

**Geminon v9 — index** (the structured source data):

| config  | `public` | `sensitive` |
|---------|---------:|------------:|
| `index` |      480 |         120 |

**Geminon v9 — corpus** (articles per slice):

| config          |     `all` |    `train` |    `val` |   `test` |
|-----------------|----------:|-----------:|---------:|---------:|
| `corpus_large`  | 1,523,754 |  1,371,378 |   76,187 |   76,189 |
| `corpus_medium` | 1,000,120 |    900,108 |   50,006 |   50,006 |
| `corpus_small`  |   200,120 |    180,108 |   10,006 |   10,006 |

**Geminon v9 — QA** (QAs per slice):

- `qa_medium` — 8,400 QAs across 4 splits: `public_val` (3,360), `public_test` (3,360), `sensitive_val` (840), `sensitive_test` (840)
- `qa_small` — same counts, with `supports` filtered to articles in `corpus_small`

**News v5 — corpus** (articles per slice):

| config          |     `all` |    `train` |    `val` |   `test` |
|-----------------|----------:|-----------:|---------:|---------:|
| `corpus_large`  | 1,768,567 |  1,591,710 |   88,428 |   88,429 |
| `corpus_medium` |   465,966 |    419,369 |   23,298 |   23,299 |
| `corpus_small`  |   212,980 |    191,682 |   10,649 |   10,649 |

**News v5 — QA** (default config, no `config_name` needed):

- `val`: 1,189 QAs
- `test`: 1,415 QAs (gets the per-cluster rounding remainder)
- Total: 2,604 good QAs that pass both the `is_underspecified == False` filter and the `closedbook_gemini-2.5-pro.is_correct == False` filter

All splits use a **seeded 90/5/5 shuffle** (`config.seed`, default 42). Train + val + test sums exactly equal the `all` slice count for every corpus config.

---

## Conventions

- **JSONL everywhere.** Every pipeline file is line-delimited JSON. Records are streamed line-by-line wherever possible to keep memory bounded on the multi-GB corpus files.
- **Stable global IDs.** Both pipelines assign a `geminon_idx` / `article_idx` to every record at the earliest possible stage and propagate it through every downstream slice, so you can trace any sample back to the original source record without ambiguity.
- **Seeded everything.** Every script that uses randomness (sampling, shuffling, val/test splits, balanced sampling, Leiden clustering) takes its seed from `config.yaml` so re-runs are byte-identical. Same seed always produces the same partition.
- **Save-prompts → query → apply.** All LLM-touching stages follow this three-step pattern. The save-prompts stage writes a JSONL of `{idx, prompt, tag}` records, the query stage adds a `response` field, and the apply stage parses responses into the next pipeline file. This means you can always inspect what was sent and what came back.
- **Idempotent re-runs.** Stages that are expensive to re-run (news cleanup/dedup, news 18_split_corpus byte-copy, geminon dedup) detect existing outputs and skip work. The corpus slice files are reused on re-runs, so adding train/val/test splits to an existing version is fast.

---

## Documentation

| File | Contents |
|------|----------|
| [README.md](README.md) (this file) | Project overview, repo layout, conventions |
| [geminon_curation/README.md](geminon_curation/README.md) | Geminon pipeline quick start, stages, output schema, config |
| [news_curation/README.md](news_curation/README.md) | News pipeline quick start, stages, output schema, config |
| [requirements.txt](requirements.txt) | Combined Python deps for both pipelines + tools |
