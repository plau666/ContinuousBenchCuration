"""Shared train/val/test split helper used by both curation pipelines.

Given a JSONL source file, builds a destination subfolder containing:
  - a symlink to the source file (so the folder is self-describing)
  - train.jsonl, val.jsonl, test.jsonl with a seeded random shuffle of the
    source's lines, split 90/5/5 by default

The split is line-based (no JSON parse) so it's fast on large files. Streaming
two passes (count + emit) keeps RAM usage tiny: only an int8 label per line.
"""

from pathlib import Path

import numpy as np


def split_train_val_test(source_path, dest_dir, source_basename=None,
                         seed=42, train_frac=0.9, val_frac=0.05,
                         relative_symlink=True):
    """Symlink source into dest_dir and write 90/5/5 train/val/test.

    Args:
        source_path: path to a JSONL file (one record per line)
        dest_dir: subfolder to create / populate
        source_basename: filename for the symlink inside dest_dir
            (defaults to source_path.name)
        seed: RNG seed for the shuffle
        train_frac, val_frac: split fractions; test gets `1 - train - val`
        relative_symlink: if True, symlink target is computed relative to
            dest_dir (so the link survives if the parent directory is moved)

    Returns:
        dict with keys n_total, n_train, n_val, n_test, train_path, val_path,
        test_path, source_link
    """
    source_path = Path(source_path).resolve()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if source_basename is None:
        source_basename = source_path.name

    # ── 1. Symlink source into dest_dir ──
    sym = dest_dir / source_basename
    if sym.is_symlink() or sym.exists():
        sym.unlink()
    if relative_symlink:
        try:
            target = Path("..") / source_path.relative_to(dest_dir.resolve().parent)
        except ValueError:
            target = source_path
    else:
        target = source_path
    sym.symlink_to(target)

    # ── 2. Count lines (pass 1) ──
    n_total = 0
    with open(source_path) as f:
        for _ in f:
            n_total += 1

    if n_total == 0:
        # Nothing to split — emit empty files
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            (dest_dir / name).write_text("")
        return {
            "n_total": 0, "n_train": 0, "n_val": 0, "n_test": 0,
            "train_path": dest_dir / "train.jsonl",
            "val_path": dest_dir / "val.jsonl",
            "test_path": dest_dir / "test.jsonl",
            "source_link": sym,
        }

    # ── 3. Decide split labels ──
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    # n_test = remainder, so rounding goes to test
    labels = np.empty(n_total, dtype=np.int8)
    labels[perm[:n_train]] = 0  # train
    labels[perm[n_train:n_train + n_val]] = 1  # val
    labels[perm[n_train + n_val:]] = 2  # test

    # ── 4. Stream + write (pass 2) ──
    train_path = dest_dir / "train.jsonl"
    val_path = dest_dir / "val.jsonl"
    test_path = dest_dir / "test.jsonl"
    fs = [open(train_path, "w"), open(val_path, "w"), open(test_path, "w")]
    counts = [0, 0, 0]
    try:
        with open(source_path) as f:
            for i, line in enumerate(f):
                lab = int(labels[i])
                fs[lab].write(line)
                counts[lab] += 1
    finally:
        for f in fs:
            f.close()

    return {
        "n_total": n_total,
        "n_train": counts[0],
        "n_val": counts[1],
        "n_test": counts[2],
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "source_link": sym,
    }
