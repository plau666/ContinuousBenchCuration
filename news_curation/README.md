# news_curation

End-to-end pipeline for building a fresh news QA dataset from a Common Crawl News dump. Same design philosophy as [`../geminon_curation/`](../geminon_curation/) — small numbered scripts, prompts saved as JSONL, LLM querying decoupled into [`../tools/query_gemini.py`](../tools/query_gemini.py).

The pipeline mirrors an earlier reference implementation, but with the LLM stages refactored into the **save-prompts → query → apply-responses** pattern, so you can swap in any batch system or rerun individual stages without touching the rest.

---

## Setup

```bash
pip install -r ../requirements.txt
# Plus heavy extras for stages 4–5 (only if you'll run them locally):
pip install warcio trafilatura langdetect sentence-transformers torch python-igraph leidenalg
```

The 19 stages have very different runtime profiles. The pre-LLM stages (download → cluster) are heavy compute; the LLM stages are mostly bookkeeping around `tools.query_gemini`.

---

## Pipeline at a glance

| Stage | Script | LLM? | What it does |
|-------|--------|------|--------------|
| 1 | `01_download_warcs.py` | no | Download all CC-NEWS WARCs for the configured month |
| 2 | `02_extract_articles.py` | no | WARC → article JSONL via warcio + trafilatura + langdetect |
| 3 | `03_cleanup_dedup.py` | no | Text normalize, quality filter, exact + MinHash LSH near-dedup |
| 4 | `04_compute_embeddings.py` | no | EmbeddingGemma article embeddings (L2-norm float16) |
| 5 | `05_local_cluster.py` | no | Windowed kNN + Leiden event clustering (thin wrapper around the original) |
| 6 | `06_save_fact_prompts.py` | no | Save fact-extraction prompts (one per cluster) |
| 7 | `07_apply_facts.py` | no | Parse responses → `qa/facts.jsonl` |
| 8 | `08_save_qa_prompts.py` | no | Save QA-generation prompts (facts + articles) |
| 9 | `09_apply_qas.py` | no | Parse responses → `qa/qas.jsonl` |
| 10 | `10_save_zeroshot_prompts.py` | no | One zero-shot prompt per QA |
| 11 | `11_apply_zeroshot.py` | no | Merge zero-shot answers → `qa/qas_with_zeroshot.jsonl` |
| 12 | `12_save_judge_prompts.py` | no | One judge prompt per cluster (batched) |
| 13 | `13_apply_judge.py` | no | Merge judge results → `qa/qas_judged.jsonl` |
| 14 | `14_compute_support_embeddings.py` | no | Encode good QAs (fact + question) and the full cleaned corpus (truncated to 1024 tokens) — produces 3 `.npy` files plus 3 `*_config.json` sidecars |
| 15 | `15_save_support_prompts.py` | no | Run top-k retrieval on **both** QA embed channels, union per QA, group by article, save `support_prompts.jsonl` and `openbook_prompts.jsonl` |
| 16 | `16_apply_support.py` | no | Parse both response files. Writes intermediate `qa/qas_with_supports.jsonl` and post-processed `qa/final/all_qas.jsonl`. |
| 17 | `17_filter_good_qas.py` | no | Filter to good QAs (`is_underspecified == False` AND `closedbook…is_correct == False`), flatten to a list-of-dicts schema, and split per-cluster (seeded) into val/test halves. Writes `qa/final/filtered/{good_qas,val,test}.jsonl`. |
| 18 | `18_split_corpus.py` | no | Split the cleaned corpus into three nested slices: `large` (full corpus), `medium` (clustered articles ∪ `small`), and `small` (union of every good QA's `supports`). For each slice, create a subfolder `corpus/{large,medium,small}/` containing a symlink to the source file plus a seeded **90/5/5 train/val/test split**. Idempotent: if a slice file already exists it's reused. |
| 19 | `19_compute_stats.py` | no | Per-split QA support-count stats and a histogram across val+test; per-slice token-count stats (Gemma-3 tokenizer or whitespace fallback) and KDE plots with inline cutoff annotations for `large`, `medium`, `small`. |

LLM calls happen **outside** the pipeline. After every `*_save_*_prompts.py` stage, run:

```bash
python -m tools.query_gemini \
    --input news_curation/output/2025_09/prompts/<file>.jsonl \
    --output news_curation/output/2025_09/responses/<file>.jsonl \
    --api-keys $GEMINI_KEY1,$GEMINI_KEY2 \
    --model gemini-2.5-pro \
    --max-workers 32 --resume
```

---

## Quick start

```bash
# 0. Edit config.yaml — set version, year_month, etc.
cd news_curation

# 1-3. Pre-LLM ingest
python 01_download_warcs.py --config config.yaml      # downloads to output/2025_09/warcs/
python 02_extract_articles.py --config config.yaml    # extracts to output/2025_09/extracted/
python 03_cleanup_dedup.py --config config.yaml       # → output/2025_09/cleaned/articles.jsonl

# 4-5. Embeddings + clustering (heavy GPU)
python 04_compute_embeddings.py --config config.yaml  # → output/2025_09/embeds/text_embeds.npy
python 05_local_cluster.py --config config.yaml       # → output/2025_09/clustered/clustered_articles.json

# 6-7. Fact extraction (LLM)
python 06_save_fact_prompts.py --config config.yaml
cd ..
python -m tools.query_gemini \
    --input news_curation/output/2025_09/prompts/fact_prompts.jsonl \
    --output news_curation/output/2025_09/responses/fact_responses.jsonl \
    --api-keys $GEMINI_KEY1,$GEMINI_KEY2 --model gemini-2.5-pro
cd news_curation
python 07_apply_facts.py --config config.yaml         # → output/2025_09/qa/facts.jsonl

# 8-9. QA generation (LLM)
python 08_save_qa_prompts.py --config config.yaml
cd ..; python -m tools.query_gemini --input news_curation/output/2025_09/prompts/qa_prompts.jsonl \
    --output news_curation/output/2025_09/responses/qa_responses.jsonl \
    --api-keys $GEMINI_KEY1,$GEMINI_KEY2 --model gemini-2.5-pro; cd news_curation
python 09_apply_qas.py --config config.yaml           # → output/2025_09/qa/qas.jsonl

# 10-11. Zero-shot (LLM)
python 10_save_zeroshot_prompts.py --config config.yaml
cd ..; python -m tools.query_gemini --input news_curation/output/2025_09/prompts/zeroshot_prompts.jsonl \
    --output news_curation/output/2025_09/responses/zeroshot_responses.jsonl \
    --api-keys $GEMINI_KEY1,$GEMINI_KEY2 --model gemini-2.5-pro; cd news_curation
python 11_apply_zeroshot.py --config config.yaml      # → output/2025_09/qa/qas_with_zeroshot.jsonl

# 12-13. Judge closed-book (LLM)
python 12_save_judge_prompts.py --config config.yaml
cd ..; python -m tools.query_gemini --input news_curation/output/2025_09/prompts/judge_prompts.jsonl \
    --output news_curation/output/2025_09/responses/judge_responses.jsonl \
    --api-keys $GEMINI_KEY1,$GEMINI_KEY2 --model gemini-2.5-pro; cd news_curation
python 13_apply_judge.py --config config.yaml         # → output/2025_09/qa/qas_judged.jsonl

# 14-16. Support check + open-book (embeddings + retrieval + 2 LLM passes)
python 14_compute_support_embeddings.py --config config.yaml
python 15_save_support_prompts.py --config config.yaml   # writes BOTH support_prompts.jsonl and openbook_prompts.jsonl
cd ..
python -m tools.query_gemini --input news_curation/output/2025_09/prompts/support_prompts.jsonl \
    --output news_curation/output/2025_09/responses/support_responses.jsonl \
    --api-keys $GEMINI_KEY1,$GEMINI_KEY2 --model gemini-2.5-pro
python -m tools.query_gemini --input news_curation/output/2025_09/prompts/openbook_prompts.jsonl \
    --output news_curation/output/2025_09/responses/openbook_responses.jsonl \
    --api-keys $GEMINI_KEY1,$GEMINI_KEY2 --model gemini-2.5-flash-lite
cd news_curation
python 16_apply_support.py --config config.yaml       # → output/2025_09/qa/qas_with_supports.jsonl

# 17. Filter to "good" QAs + flatten + per-cluster val/test split
python 17_filter_good_qas.py --config config.yaml    # → output/2025_09/qa/final/filtered/{good_qas,val,test}.jsonl

# 18. Split corpus into large/medium/small (medium = clustered ∪ small)
python 18_split_corpus.py --config config.yaml       # → output/2025_09/corpus/{large,medium,small}.jsonl

# 19. Stats
python 19_compute_stats.py --config config.yaml
```

---

## Output structure

```
output/2025_09/
├── warcs/                            # Stage 1
│   └── *.warc.gz
├── extracted/                        # Stage 2 (per-WARC JSONL)
│   └── *.jsonl
├── cleaned/                          # Stage 3
│   └── articles.jsonl                # final, deduped corpus
├── embeds/                           # Stage 4
│   ├── text_embeds.npy               # (N, 768) float16, L2-normalized
│   └── embeds_config.json
├── clustered/                        # Stage 5
│   ├── clustered_articles.json       # {cluster_id: [article, ...]}
│   ├── merged_clusters.json
│   └── clustering_config.json
├── prompts/                          # Stages 6, 8, 10, 12, 15
│   ├── fact_prompts.jsonl
│   ├── qa_prompts.jsonl
│   ├── zeroshot_prompts.jsonl
│   ├── judge_prompts.jsonl
│   ├── support_prompts.jsonl         # stage 15a — verifier prompt (Q + A + article)
│   └── openbook_prompts.jsonl        # stage 15b — open-book prompt (Q + article only)
├── responses/                        # Created by tools.query_gemini
│   ├── fact_responses.jsonl
│   ├── qa_responses.jsonl
│   ├── zeroshot_responses.jsonl
│   ├── judge_responses.jsonl
│   ├── support_responses.jsonl
│   └── openbook_responses.jsonl
├── qa/                               # Built up across stages 7, 9, 11, 13, 16
│   ├── facts.jsonl                   # Stage 7
│   ├── qas.jsonl                     # Stage 9
│   ├── qas_with_zeroshot.jsonl       # Stage 11
│   ├── qas_judged.jsonl              # Stage 13
│   ├── qas_with_supports.jsonl       # Stage 16 (intermediate)
│   └── final/
│       ├── all_qas.jsonl             # Stage 16 (post-processed, cluster-grouped)
│       └── filtered/                 # Stage 17 outputs (flat list-of-dicts schema)
│           ├── good_qas.jsonl        # all surviving QAs (flat)
│           ├── val.jsonl             # per-cluster half (seeded shuffle, floor)
│           └── test.jsonl            # per-cluster other half (gets extra on odd counts)
├── support/                          # Stage 14 (intermediate, used by stage 15 retrieval)
│   ├── qa_index.jsonl                # one row per good QA: {cluster_id, qa_idx, question, answer}
│   ├── fact_embeds.npy               # encodes "{q} {a}" with `task: fact checking | query: ` (512 tok)
│   ├── fact_embeds_config.json
│   ├── question_embeds.npy           # encodes "{q}"     with `task: question answering | query: ` (512 tok)
│   ├── question_embeds_config.json
│   ├── doc_index.jsonl               # one row per cleaned article: {article_idx}
│   ├── doc_embeds.npy                # encodes "title: {t} | text: {body[:1024 tok]}" with `task: clustering | query: ` (1100 seq)
│   └── doc_embeds_config.json
├── corpus/                           # Stage 18 — three nested slices, each with a 90/5/5 split
│   ├── large.jsonl                   # full deduped corpus (copy of cleaned/articles.jsonl, with article_idx)
│   ├── medium.jsonl                  # clustered articles ∪ small (so small ⊆ medium ⊆ large)
│   ├── small.jsonl                   # union of `supports` across good_qas.jsonl
│   ├── large/
│   │   ├── all.jsonl  → ../large.jsonl   (relative symlink — uniform name across slices)
│   │   ├── train.jsonl               # 90% (seeded shuffle)
│   │   ├── val.jsonl                 # 5%
│   │   └── test.jsonl                # 5% (gets the rounding remainder)
│   ├── medium/
│   │   ├── all.jsonl  → ../medium.jsonl
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── small/
│       ├── all.jsonl  → ../small.jsonl
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
└── stats/                            # Stage 19
    ├── qa_summary.json                       # cluster + QA counts + per-split support stats
    ├── support_stats_per_split.json          # {val, test} → {n_qas, mean, median, p25, p75, std, max}
    ├── support_count_dist.png                # histogram across val + test
    ├── token_counts_{large,medium,small}.json   # one entry per article in corpus/<slice>.jsonl
    ├── token_stats_{large,medium,small}.json    # n, mean, std, min, p25, median, p75, p90, p99, max
    └── token_dist_News_{Large,Medium,Small}.png # per-slice KDE plot with inline cutoff annotations
```

---

## Record schemas

**Cleaned article** (`cleaned/articles.jsonl`)
```json
{
  "url": "https://example.com/...",
  "hostname": "example.com",
  "title": "...",
  "date": "2025-09-01",
  "crawl_date": "2025-09-01T00:00:00Z",
  "language": "en",
  "text": "...",
  "article_idx": 12345
}
```
`article_idx` is assigned in stage 3 (0-indexed, sequential after dedup). It propagates through every downstream stage so any record that references an article (e.g. `supports`, `openbook_gemini-2.5-flash-lite`) uses this stable global id.

**Cluster of articles** (`clustered/clustered_articles.json`) — produced by stage 5
```json
{
  "0": [<article>, <article>, ...],
  "1": [<article>, ...]
}
```

**Fact record** (`qa/facts.jsonl`)
```json
{"cluster_id": "0", "article_count": 230, "articles_used": 50, "facts": ["...", "..."]}
```

**QA record** (`qa/qas*.jsonl`) — fields accumulate across stages
```json
{
  "cluster_id": "0",
  "article_count": 230,
  "articles_used": 50,
  "facts_used": 15,
  "qas": [
    {
      "question": "...",
      "answer": "...",
      "root_articles_ids": [1, 2, 3],                  // stage 9: 1-indexed positions inside the QA-gen prompt
      "0shot_bestguess_gemini-2.5-pro": "...",         // stage 11
      "is_zeroshot_correct": false,                     // stage 13
      "is_underspecified": false,                       // stage 13
      "supports": [12345, 67890, ...],                  // stage 16: article_idx values that the verifier confirmed
      "openbook_gemini-2.5-flash-lite": [               // stage 16: open-book answers per retrieved article
        ["Andy Pycroft", 12345, true],
        ["unknown", 67890, false],
        ["A. Pycroft", 102345, true]
      ]
    }
  ]
}
```

Notes:
- `root_articles_ids` (stage 9) — 1-indexed positions inside the QA-generation prompt's article list. These are what the *generator model* claimed; unverified. Renamed from the legacy `supporting_articles_ids`; the parser still accepts the old name for backwards compatibility.
- `supports` (stage 16) — global `article_idx` values from `cleaned/articles.jsonl` that the verifier model confirmed actually support the answer. Use this for evaluation, not `root_articles_ids`.
- `openbook_gemini-2.5-flash-lite` (stage 16, intermediate file) — per retrieved article, a raw triple `[response, article_idx, is_correct]`. The post-processed final file reformats this as a list of dicts; see below.

**Final QA record** (`qa/final/all_qas.jsonl`) — produced by stage 16's post-processing pass

```json
{
  "cluster_id": "0",
  "article_count": 230,
  "articles_used": 50,
  "facts_used": 15,
  "qas": [
    {
      "question": "Who was the match referee...?",
      "answer": "Andy Pycroft",
      "root_articles_ids": [1, 2, 3],
      "is_underspecified": false,
      "supports": [12345, 67890],
      "closedbook_gemini-2.5-pro": {
        "answer": "Javagal Srinath",
        "is_correct": false
      },
      "openbook_gemini-2.5-flash-lite": [
        {"article_idx": 12345, "answer": "Andy Pycroft", "is_correct": true},
        {"article_idx": 67890, "answer": "unknown",      "is_correct": false}
      ]
    }
  ]
}
```

Differences vs the intermediate `qas_with_supports.jsonl`:
- `0shot_bestguess_gemini-2.5-pro` and `is_zeroshot_correct` are **removed** and merged into a single `closedbook_gemini-2.5-pro` dict with keys `answer` + `is_correct`.
- `openbook_gemini-2.5-flash-lite` is **reformatted** from a list of triples `[answer, article_idx, is_correct]` to a list of dicts `{"article_idx": ..., "answer": ..., "is_correct": ...}`.
- All other fields are preserved.

**Filtered QA record** (`qa/final/filtered/{good_qas,val,test}.jsonl`) — produced by stage 17

Each line is a single flat QA dict (NOT cluster-grouped) with exactly these 5 fields:

```json
{
  "question": "Who was the match referee...?",
  "answer": "Andy Pycroft",
  "supports": [12345, 67890],
  "closedbook_gemini-2.5-pro": {
    "answer": "Javagal Srinath",
    "is_correct": false
  },
  "openbook_gemini-2.5-flash-lite": [
    {"article_idx": 12345, "answer": "Andy Pycroft", "is_correct": true},
    {"article_idx": 67890, "answer": "unknown",      "is_correct": false}
  ]
}
```

Stage 17 drops `cluster_id`, `article_count`, `articles_used`, `facts_used`, `root_articles_ids`, and `is_underspecified` (the latter is always `false` after filtering anyway). Every QA in `good_qas.jsonl` is guaranteed:
- `is_underspecified` was `false` in stage 13's judgment
- `closedbook_gemini-2.5-pro.is_correct` is `false`

The val/test split uses a per-cluster seeded shuffle (default seed = `config.seed`). For each cluster, `floor(n/2)` QAs go to val and `ceil(n/2)` go to test, then both are flattened. Same seed always gives the same partition (verified via subprocess test).

---

## Configuration

Everything version-specific lives in [config.yaml](config.yaml):

- `version` — output directory under `output/`
- `year_month` — Common Crawl News dump (e.g. `"2025/09"`)
- `cleanup.*` — dedup parameters (containment threshold, MinHash perms, n-gram size)
- `embeddings.*` — model, batch size, max sequence length, input mode
- `clustering.*` — windowed Leiden parameters (window days, k-search, sim threshold, etc.)
- `facts.*` / `qa_generation.*` — `max_articles`, `max_chars`, `top_k_clusters`
- `support.*` — embedding model, retrieval top-k, prompt token cap, good-QA filter rules

To create a new version, copy `config.yaml`, change `version:` and any params, then re-run.

---

## Where shared code lives

Generic utilities live in [`../tools/`](../tools/) and are re-exported through `news_curation/utils/`:

- `tools.io` — JSONL, config, response parsing, NumpyEncoder
- `tools.dedup` — exact + MinHash LSH near-dedup (used in stage 3)
- `tools.query_gemini` — Gemini API utility with key rotation + retry + resume
- `tools.push_to_hf` — HuggingFace dataset push

News-specific utilities in `news_curation/utils/`:
- `utils.normalize` — text normalization (paragraph cleanup, soft linebreak unwrap)
- `utils.io.ensure_output_dir` — creates the news subfolder layout

---

## Pushing to HuggingFace

Same as geminon — uses [`tools/push_to_hf.py`](../tools/push_to_hf.py):

```bash
export HF_TOKEN=hf_xxx

python -m tools.push_to_hf --curation news --version 2025_09
```

Defaults: target repo is `ContinuousBench/News`, the commit is tagged `2025_09`. Pass `--repo <org>/<name>` to publish under a different org, or `--public` / `--skip-tag` to override the defaults. See `python -m tools.push_to_hf --help` for the full flag list.

---

## Notes on the heavy stages

**Stage 4 (embeddings)** — single-process by default. The original CC pipeline shards across multiple GPUs; you can either run multiple copies of `04_compute_embeddings.py` with `--shard-idx`/`--shard-total` and concatenate the outputs, or call the original `compute_text_embeds.py` directly.

**Stage 5 (clustering)** — thin wrapper around an external `local_cluster.py` (~800 lines of windowed kNN + Leiden + GPU code). Set the `CC_LOCAL_CLUSTER_SCRIPT` environment variable, or pass `--source-script PATH`, to point at your copy. The wrapper just constructs CLI args from `config.yaml` and shells out.
