"""Stage 2: Extract articles from CC-NEWS WARC files (warcio + trafilatura).

Processes WARC files in parallel, extracts title/date/text, applies a fast
HTML-lang pre-filter, then a langdetect post-filter on extracted text.

Inputs:  {output_dir}/{version}/warcs/*.warc.gz
Outputs: {output_dir}/{version}/extracted/*.jsonl

Usage:
    python 02_extract_articles.py --config config.yaml
"""

import argparse
import glob
import json
import os
import re
import time
from multiprocessing import Pool
from urllib.parse import urlparse

from tqdm import tqdm

from utils.io import load_config, ensure_output_dir

_HTML_LANG_RE = re.compile(r'<html[^>]*\slang=["\']([a-zA-Z]{2})', re.IGNORECASE)


def _detect_html_lang(html):
    m = _HTML_LANG_RE.search(html[:2000])
    return m.group(1).lower() if m else None


def extract_warc_file(args):
    """Process one WARC file: extract articles, filter, save as JSONL."""
    warc_path, output_dir, languages, min_text_length = args

    # Lazy imports so the worker can pickle/spawn cleanly
    from warcio.archiveiterator import ArchiveIterator
    import trafilatura
    from langdetect import detect, LangDetectException

    basename = os.path.basename(warc_path).replace(".warc.gz", ".jsonl")
    output_path = os.path.join(output_dir, basename)

    if os.path.exists(output_path):
        return {"file": basename, "status": "skipped", "total": 0, "extracted": 0, "kept": 0,
                "lang_skipped": 0, "errors": 0}

    stats = {"file": basename, "status": "ok", "total": 0, "extracted": 0,
             "kept": 0, "lang_skipped": 0, "errors": 0}
    results = []

    try:
        with open(warc_path, "rb") as f:
            for record in ArchiveIterator(f):
                if record.rec_type != "response":
                    continue
                stats["total"] += 1

                try:
                    url = record.rec_headers.get_header("WARC-Target-URI")
                    crawl_date = record.rec_headers.get_header("WARC-Date") or ""

                    content_type = ""
                    if record.http_headers:
                        content_type = record.http_headers.get_header("Content-Type") or ""

                    charset = "utf-8"
                    if "charset=" in content_type:
                        charset = content_type.split("charset=")[-1].split(";")[0].strip()

                    raw_bytes = record.content_stream().read()
                    try:
                        html = raw_bytes.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        html = raw_bytes.decode("utf-8", errors="replace")

                    if languages != "all":
                        html_lang = _detect_html_lang(html)
                        if html_lang:
                            is_en = html_lang.startswith("en")
                            if languages == "english" and not is_en:
                                stats["lang_skipped"] += 1
                                continue
                            if languages == "nonenglish" and is_en:
                                stats["lang_skipped"] += 1
                                continue

                    result = trafilatura.extract(
                        html, url=url, output_format="json",
                        with_metadata=True, include_comments=False,
                        include_links=False, include_tables=False,
                    )
                    if not result:
                        continue

                    data = json.loads(result)
                    text = data.get("text", "")
                    if not text or len(text.strip()) < min_text_length:
                        continue

                    stats["extracted"] += 1

                    lang = ""
                    try:
                        lang = detect(text[:1000])
                    except LangDetectException:
                        pass
                    if languages == "english" and lang != "en":
                        continue
                    if languages == "nonenglish" and lang == "en":
                        continue

                    stats["kept"] += 1
                    hostname = urlparse(url).netloc if url else ""
                    results.append({
                        "url": url,
                        "hostname": hostname,
                        "title": data.get("title", ""),
                        "date": data.get("date", ""),
                        "crawl_date": crawl_date,
                        "language": lang,
                        "text": text.strip(),
                    })
                except Exception:
                    stats["errors"] += 1
                    continue
    except Exception as e:
        stats["status"] = f"error: {e}"

    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract articles from WARCs")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    warc_dir = output_dir / "warcs"
    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    cfg = config["extract"]
    languages = cfg["languages"]
    workers = cfg["workers"]
    min_text_length = cfg["min_text_length"]

    warc_files = sorted(glob.glob(str(warc_dir / "*.warc.gz")))
    print(f"Found {len(warc_files)} WARCs in {warc_dir}")
    print(f"Languages: {languages} | Workers: {workers}")

    task_args = [(f, str(extracted_dir), languages, min_text_length) for f in warc_files]
    total = {"total": 0, "extracted": 0, "kept": 0, "lang_skipped": 0, "errors": 0, "skipped": 0}

    t0 = time.time()
    with Pool(processes=workers) as pool:
        pbar = tqdm(pool.imap_unordered(extract_warc_file, task_args),
                    total=len(warc_files), desc="Extracting", unit="file")
        for s in pbar:
            for k in ("total", "extracted", "kept", "lang_skipped", "errors"):
                total[k] += s[k]
            if s["status"] == "skipped":
                total["skipped"] += 1
            pbar.set_postfix(records=f"{total['total']:,}", kept=f"{total['kept']:,}",
                             errors=f"{total['errors']:,}")

    print(f"\nDone in {(time.time()-t0)/60:.1f}min")
    print(f"  Total WARC records: {total['total']:,}")
    print(f"  Extracted: {total['extracted']:,}")
    print(f"  Kept ({languages}): {total['kept']:,}")
    print(f"  Lang pre-skipped: {total['lang_skipped']:,}")
    print(f"  Errors: {total['errors']:,}")
    print(f"Output: {extracted_dir}/")


if __name__ == "__main__":
    main()
