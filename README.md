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
├── README.md                ← you are here
├── requirements.txt         ← combined deps for both pipelines + tools/
├── tools/                   ← shared utilities (io, dedup, sampling, split,
│                              query_gemini, push_to_hf, meta_data)
├── geminon_curation/        ← Geminon pipeline — 8 numbered stages,
│                              templates/, committed reference Pokémon CSVs
└── news_curation/           ← News pipeline — 19 numbered stages, templates/
```

For per-pipeline quick starts, stage tables, and output schemas, read the per-pipeline READMEs:
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

## Using the released datasets

The released versions live on HuggingFace at [`ContinuousBench/Geminon`](https://huggingface.co/datasets/ContinuousBench/Geminon) and [`ContinuousBench/News`](https://huggingface.co/datasets/ContinuousBench/News). Per-config splits and sizes are documented on each dataset card.

```python
from datasets import load_dataset

load_dataset("ContinuousBench/Geminon", "corpus_large", split="train")
load_dataset("ContinuousBench/Geminon", "qa_small",     split="public_val")

load_dataset("ContinuousBench/News",    "corpus_large", split="train")
load_dataset("ContinuousBench/News",    split="val")      # qa is the default config

# Pin a specific release
load_dataset("ContinuousBench/Geminon", "index",
             split="public", revision="2025_09")
```

> **Publishing your own version.** If you regenerate either dataset and want to host it on HuggingFace, [`tools/push_to_hf.py`](tools/push_to_hf.py) renders a YAML-frontmatter dataset card with `configs` + splits and (optionally) tags the commit with your version label. Pass `--repo your-org/your-dataset` to override the default upload target. See `python -m tools.push_to_hf --help` for the full flag set (`--public`, `--skip-tag`, `--skip-qa`, `--dry-run`, …). [`tools/meta_data.py`](tools/meta_data.py) does the matching Croissant + RAI merge.

---

## Conventions

- **JSONL everywhere** — every pipeline file is line-delimited JSON, streamed where possible to keep memory bounded on multi-GB corpora.
- **Stable global IDs** — `geminon_idx` / `article_idx` are assigned at the earliest stage and propagated through every downstream slice, so any sample traces back to its source record.
- **Seeded everything** — every script that uses randomness reads its seed from `config.yaml`; re-runs with the same seed produce byte-identical partitions.
- **Save-prompts → query → apply** — every LLM-touching stage writes prompts as JSONL, the query step adds a `response` field, and the apply step parses responses. You can always inspect what was sent and what came back.
- **Idempotent re-runs** — expensive stages skip work when their outputs already exist, so partial re-runs are cheap.

---

## Documentation

| File | Contents |
|------|----------|
| [README.md](README.md) (this file) | Project overview, repo layout, conventions |
| [geminon_curation/README.md](geminon_curation/README.md) | Geminon pipeline quick start, stages, output schema, config |
| [news_curation/README.md](news_curation/README.md) | News pipeline quick start, stages, output schema, config |
| [requirements.txt](requirements.txt) | Combined Python deps for both pipelines + tools |
