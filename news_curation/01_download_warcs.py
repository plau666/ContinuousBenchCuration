"""Stage 1: Download Common Crawl News WARC files for a given month.

Reads `year_month` and `download.workers` from config.yaml. Downloads all
WARC files into `{output_dir}/{version}/warcs/`.

Usage:
    python 01_download_warcs.py --config config.yaml
    python 01_download_warcs.py --config config.yaml --only CC-NEWS-20250901004501-03802.warc.gz
"""

import argparse
import concurrent.futures
import gzip
import io
import os
import time
from pathlib import Path

import requests

from utils.io import load_config, ensure_output_dir

BASE_URL = "https://data.commoncrawl.org/"


def download_warc_paths(year_month, output_dir):
    """Fetch warc.paths.gz and return list of WARC paths."""
    url = f"{BASE_URL}crawl-data/CC-NEWS/{year_month}/warc.paths.gz"
    print(f"Downloading WARC paths from {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    paths_dest = output_dir / f"warc.paths.{year_month.replace('/', '-')}.gz"
    paths_dest.write_bytes(resp.content)
    print(f"  Saved WARC paths to {paths_dest}")

    with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def download_single(warc_path, output_dir, retries=5, backoff=10):
    url = BASE_URL + warc_path
    local_path = output_dir / os.path.basename(warc_path)

    if local_path.exists():
        return True

    wait_time = backoff
    for attempt in range(retries):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            tmp_path.rename(local_path)
            return True
        except Exception as e:
            print(f"  Error downloading {os.path.basename(warc_path)} (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(wait_time)
                wait_time *= 2
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
    return False


def main():
    parser = argparse.ArgumentParser(description="Download CC-NEWS WARC files")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--only", type=str, default=None,
                        help="Download only a specific WARC filename")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    warc_dir = output_dir / "warcs"
    warc_dir.mkdir(parents=True, exist_ok=True)

    year_month = config["year_month"]
    workers = config["download"]["workers"]

    paths = download_warc_paths(year_month, warc_dir)
    if args.only:
        only_basename = os.path.basename(args.only)
        paths = [p for p in paths if p.endswith(only_basename)]
        if not paths:
            print(f"No matching WARC file found for: {args.only}")
            return
    print(f"Found {len(paths)} WARC files to download with {workers} workers")

    t0 = time.time()
    ok = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_single, p, warc_dir): p for p in paths}
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                ok += 1
            else:
                fail += 1
            done = ok + fail
            if done % 20 == 0 or done == len(paths):
                elapsed = time.time() - t0
                print(f"  Progress: {done}/{len(paths)} (ok={ok}, fail={fail}, {elapsed/60:.1f}min)")

    print(f"\nDone: {ok} downloaded, {fail} failed in {(time.time()-t0)/60:.1f}min")
    print(f"Output: {warc_dir}")


if __name__ == "__main__":
    main()
