"""Project-wide tools shared across all curation pipelines.

Modules:
  io                — generic JSONL, config, template, response parsing
  dedup             — corpus deduplication (exact + MinHash LSH near-dedup)
  balanced_sampler  — feature-aware balanced sampling
  query_gemini      — Gemini API utility (callable as a script or imported)
  push_to_hf        — HuggingFace dataset push utility
"""
