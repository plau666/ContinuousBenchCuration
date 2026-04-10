"""Stage 16: Parse support-check + open-book responses → final qas.

Reads two response files produced by tools.query_gemini:
  - support_responses.jsonl  → JSON array of bools per QA → populates `supports`
  - openbook_responses.jsonl → JSON array of answers per QA → populates
    `openbook_gemini-2.5-flash-lite` (raw triples in the intermediate file)

Correctness is computed by normalize-and-substring matching the open-book answer
against the ground-truth `answer`. Replace `is_openbook_correct` with a stronger
judge (e.g. another Gemini call) if you need it.

Outputs:
  qa/qas_with_supports.jsonl   — intermediate file with raw triple entries
  qa/final/all_qas.jsonl       — post-processed: fields renamed/restructured for
                                 evaluation. `0shot_bestguess_gemini-2.5-pro` and
                                 `is_zeroshot_correct` are merged into a single
                                 `closedbook_gemini-2.5-pro` dict, and the
                                 openbook field is reformatted as a list of
                                 {article_idx, answer, is_correct} dicts.

Inputs:  {output_dir}/{version}/qa/qas_judged.jsonl
         {output_dir}/{version}/responses/support_responses.jsonl
         {output_dir}/{version}/responses/openbook_responses.jsonl

Usage:
    python 16_apply_support.py --config config.yaml
    python 16_apply_support.py --config config.yaml --skip-openbook
"""

import argparse
import json
import re
import string
from collections import defaultdict

from utils.io import load_config, load_jsonl, save_jsonl, ensure_output_dir


OPENBOOK_FIELD = "openbook_gemini-2.5-flash-lite"
ZEROSHOT_FIELD = "0shot_bestguess_gemini-2.5-pro"
CLOSEDBOOK_FIELD = "closedbook_gemini-2.5-pro"


# ─── Response parsing ───────────────────────────────────────────────────────
def _strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_bool_list(text):
    """Parse a JSON array of booleans (support_check responses)."""
    if not text:
        return []
    text = _strip_fences(text)
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if not bracket_match:
        return []
    try:
        cleaned = re.sub(r"#.*", "", bracket_match.group())
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [bool(x) for x in result]
    except json.JSONDecodeError:
        pass
    return []


def parse_string_list(text):
    """Parse a JSON array of strings (open-book responses)."""
    if not text:
        return []
    text = _strip_fences(text)
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if not bracket_match:
        return []
    try:
        cleaned = re.sub(r"#.*", "", bracket_match.group())
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(x).strip() if x is not None else "" for x in result]
    except json.JSONDecodeError:
        pass
    return []


# ─── Correctness ────────────────────────────────────────────────────────────
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def is_openbook_correct(predicted, ground_truth):
    """Lightweight correctness check: normalized substring match in either direction.

    Replace with a stronger judge (e.g. an LLM call) if needed.
    """
    p = normalize_answer(predicted)
    g = normalize_answer(ground_truth)
    if not p or not g:
        return False
    return (g in p) or (p in g)


# ─── Post-processing ────────────────────────────────────────────────────────
def postprocess_record(rec):
    """Strip + restructure a cluster record into the final eval-ready shape.

    Per-QA changes:
      - Drop `0shot_bestguess_gemini-2.5-pro` and `is_zeroshot_correct`.
      - Add `closedbook_gemini-2.5-pro` = {answer, is_correct}.
      - Reformat `openbook_gemini-2.5-flash-lite` from
        [[answer, article_idx, is_correct], ...]
        to
        [{"article_idx": ..., "answer": ..., "is_correct": ...}, ...]
    """
    out = dict(rec)
    new_qas = []
    for qa in out.get("qas", []):
        q = dict(qa)
        zeroshot_answer = q.pop(ZEROSHOT_FIELD, None)
        zeroshot_correct = q.pop("is_zeroshot_correct", None)
        q[CLOSEDBOOK_FIELD] = {
            "answer": zeroshot_answer,
            "is_correct": zeroshot_correct,
        }

        ob_raw = q.get(OPENBOOK_FIELD, [])
        q[OPENBOOK_FIELD] = [
            {"article_idx": int(entry[1]), "answer": entry[0], "is_correct": bool(entry[2])}
            for entry in ob_raw
            if isinstance(entry, (list, tuple)) and len(entry) >= 3
        ]
        new_qas.append(q)
    out["qas"] = new_qas
    return out


def main():
    parser = argparse.ArgumentParser(description="Apply support + open-book responses")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--support-responses", type=str, default=None)
    parser.add_argument("--openbook-responses", type=str, default=None)
    parser.add_argument("--skip-openbook", action="store_true",
                        help="Don't read openbook responses (leaves the field empty)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)

    qas_path = output_dir / "qa" / "qas_judged.jsonl"
    records = load_jsonl(str(qas_path))
    print(f"Loaded {len(records)} cluster records from {qas_path}")

    # ── Support check responses → bool list per (article, [qa_keys]) ──────
    support_path = args.support_responses or str(
        output_dir / "responses" / "support_responses.jsonl"
    )
    support_responses = load_jsonl(support_path)
    print(f"Loaded {len(support_responses)} support responses from {support_path}")

    qa_to_supports = defaultdict(set)
    n_support_parsed = n_support_failed = 0
    for r in support_responses:
        tag = r.get("tag", {})
        article_idx = tag.get("article_idx")
        qa_list = tag.get("qas", [])
        bools = parse_bool_list(r.get("response", "") or "")
        if not bools or len(bools) != len(qa_list):
            n_support_failed += 1
            continue
        n_support_parsed += 1
        for qa_meta, supported in zip(qa_list, bools):
            if supported:
                qa_to_supports[(str(qa_meta["cluster_id"]), qa_meta["qa_idx"])].add(int(article_idx))
    print(f"  parsed={n_support_parsed}, failed={n_support_failed}")

    # ── Open-book responses → list[(response, article_idx)] per QA ────────
    qa_to_openbook = defaultdict(list)  # (cid, qi) → list[(response_text, article_idx)]
    if not args.skip_openbook:
        openbook_path = args.openbook_responses or str(
            output_dir / "responses" / "openbook_responses.jsonl"
        )
        try:
            openbook_responses = load_jsonl(openbook_path)
        except FileNotFoundError:
            print(f"  open-book responses not found at {openbook_path} — skipping")
            openbook_responses = []
        print(f"Loaded {len(openbook_responses)} open-book responses from {openbook_path}")

        n_ob_parsed = n_ob_failed = 0
        for r in openbook_responses:
            tag = r.get("tag", {})
            article_idx = tag.get("article_idx")
            qa_list = tag.get("qas", [])
            answers = parse_string_list(r.get("response", "") or "")
            if not answers or len(answers) != len(qa_list):
                n_ob_failed += 1
                continue
            n_ob_parsed += 1
            for qa_meta, ans in zip(qa_list, answers):
                qa_to_openbook[(str(qa_meta["cluster_id"]), qa_meta["qa_idx"])].append(
                    (ans, int(article_idx))
                )
        print(f"  parsed={n_ob_parsed}, failed={n_ob_failed}")

    # ── Attach to QAs ─────────────────────────────────────────────────────
    n_with_supports = n_with_openbook = 0
    for rec in records:
        cid = str(rec.get("cluster_id"))
        for qi, qa in enumerate(rec.get("qas", [])):
            key = (cid, qi)
            qa["supports"] = sorted(qa_to_supports.get(key, set()))
            if qa["supports"]:
                n_with_supports += 1

            ob_entries = qa_to_openbook.get(key, [])
            if ob_entries:
                ground_truth = qa.get("answer") or ""
                qa[OPENBOOK_FIELD] = [
                    [resp, aidx, is_openbook_correct(resp, ground_truth)]
                    for resp, aidx in sorted(ob_entries, key=lambda x: x[1])
                ]
                n_with_openbook += 1
            else:
                qa[OPENBOOK_FIELD] = []

    intermediate_path = output_dir / "qa" / "qas_with_supports.jsonl"
    save_jsonl(records, str(intermediate_path))
    print(f"\nSaved intermediate file: {intermediate_path}")
    print(f"  QAs with at least 1 support:        {n_with_supports}")
    print(f"  QAs with at least 1 openbook entry: {n_with_openbook}")

    # ── Post-process: produce qa/final/all_qas.jsonl ─────────────────────
    final_dir = output_dir / "qa" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_records = [postprocess_record(rec) for rec in records]
    final_path = final_dir / "all_qas.jsonl"
    save_jsonl(final_records, str(final_path))
    print(f"Saved final file:        {final_path}")


if __name__ == "__main__":
    main()
