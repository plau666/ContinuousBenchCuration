"""Corpus deduplication utilities.

Two-stage dedup:
  1. Exact dedup via SHA-256 hash on text field.
  2. Near-dedup via MinHash LSH with containment similarity.
"""

import hashlib
import re
from multiprocessing import Pool

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

_WORD_RE = re.compile(r"[a-z0-9]+")


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingle(text, n=5):
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_minhash(shingle_set, num_perm):
    m = MinHash(num_perm=num_perm)
    for s in shingle_set:
        m.update(s.encode("utf-8"))
    return m


def containment(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return intersection / min(len(set_a), len(set_b))


def exact_dedup(articles):
    """Remove exact duplicates by SHA-256 hash on text field."""
    seen = set()
    deduped = []
    for art in articles:
        h = text_hash(art["text"])
        if h not in seen:
            seen.add(h)
            deduped.append(art)
    return deduped


def _shingle_and_hash_worker(args):
    text, shingle_size, num_perm = args
    s = shingle(text, n=shingle_size)
    mh = build_minhash(s, num_perm)
    return s, mh


def near_dedup(articles, threshold=0.80, num_perm=128, shingle_size=5, workers=40):
    """Near-deduplicate articles via MinHash LSH with containment similarity."""
    print(f"  Building shingles and MinHash for {len(articles):,} articles ({workers} workers)...")
    task_args = [(art["text"], shingle_size, num_perm) for art in articles]

    shingle_sets = []
    minhashes = []
    with Pool(processes=workers) as pool:
        for s, mh in tqdm(pool.imap(_shingle_and_hash_worker, task_args, chunksize=1000),
                          total=len(articles), desc="  Shingling", unit="article"):
            shingle_sets.append(s)
            minhashes.append(mh)

    print("  Building LSH index...")
    lsh_threshold = max(0.5, threshold - 0.2)
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    for i, mh in tqdm(enumerate(minhashes), total=len(minhashes), desc="  LSH insert", unit="article"):
        try:
            lsh.insert(str(i), mh)
        except ValueError:
            pass

    print("  Finding and removing near-duplicates...")
    removed = set()
    for i in tqdm(range(len(articles)), desc="  Deduping", unit="article"):
        if i in removed:
            continue
        candidates = lsh.query(minhashes[i])
        for c in candidates:
            j = int(c)
            if j == i or j in removed:
                continue
            sim = containment(shingle_sets[i], shingle_sets[j])
            if sim >= threshold:
                if len(shingle_sets[i]) >= len(shingle_sets[j]):
                    removed.add(j)
                else:
                    removed.add(i)
                    break

    return [art for i, art in enumerate(articles) if i not in removed]
