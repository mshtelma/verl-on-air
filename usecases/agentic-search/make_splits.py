#!/usr/bin/env python3
"""The agentic-search evaluation splits: a DEV set (checkpoints are chosen on it) and a held-out
TEST set (reported once), with their question IDs committed next to this file.

    python3 usecases/agentic-search/make_splits.py --out /Volumes/.../qa_musique/heldout_test.parquet
    python3 usecases/agentic-search/make_splits.py --check      # the committed lists are reproduced

Both come from MuSiQue-Ans validation at the revision prep_data.py pins (never trained on):

  dev   the first 200 answerable validation rows -- the questions every historical number (and the
        choice of step 20) was measured on. Kept as-is so those numbers stay comparable; it is
        2-hop only, because MuSiQue's validation split is ordered by hop count.
  test  N (default 500) rows drawn from validation rows 500.. -- past the whole historical
        test.parquet (rows 0-499), so no test question was ever scored during development --
        stratified by hop count (proportional allocation, largest remainder) with a fixed seed.

The index (corpus_big) unions MuSiQue's whole validation split, so every test question's gold and
distractor paragraphs are retrievable: the same transductive setting as dev (see build_corpus.py).
IDs are MuSiQue's own `id`s; splits/SPLITS.json records the source revision, seed, the per-hop
counts and the sha256 of each list. The parquet has prep_data.py's schema, so eval.py reads it via
QA_VAL_PARQUET unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_data as pd_  # noqa: E402
import data_manifest as dm  # noqa: E402  (engine/lib, on the path via prep_data)

SPLITS_DIR = HERE / "splits"
DEV_N = 200
EXCLUDE_FIRST = 500
SEED = 20260923


def _ids_text(ids: list[str]) -> str:
    return "".join(f"{i}\n" for i in ids)


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of n over strata (largest remainder; ties by stratum name)."""
    total = sum(counts.values())
    if n > total:
        raise SystemExit(f"asked for {n} test questions; the pool holds {total}")
    exact = {k: n * c / total for k, c in counts.items()}
    got = {k: int(v) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k] - got[k]), k))[: n - sum(got.values())]:
        got[k] += 1
    return got


def build(rows: list[dict], n_test: int, seed: int = SEED) -> tuple[list[dict], list[dict], dict]:
    """rows: prep_data verl rows of the validation split, in source order -> (dev, test, record)."""
    dev = rows[:DEV_N]
    pool = rows[EXCLUDE_FIRST:]
    strata: dict[str, list[dict]] = {}
    for r in pool:
        strata.setdefault(r["extra_info"]["hop_type"] or "?", []).append(r)
    want = allocate({k: len(v) for k, v in strata.items()}, n_test)
    rng = random.Random(seed)
    chosen: set[str] = set()
    for k in sorted(strata):
        chosen |= {r["extra_info"]["source_id"] for r in rng.sample(strata[k], want[k])}
    test = [r for r in pool if r["extra_info"]["source_id"] in chosen]   # source order
    record = {
        "dev": {"rule": f"the first {DEV_N} answerable validation rows (the historical dev set)",
                "n": len(dev), "by_hops": dict(Counter(r["extra_info"]["hop_type"] for r in dev))},
        "test": {"rule": f"validation rows {EXCLUDE_FIRST}.. stratified by hop count, proportional, "
                         f"seed {seed}",
                 "n": len(test), "seed": seed, "pool": len(pool),
                 "pool_by_hops": {k: len(v) for k, v in sorted(strata.items())},
                 "by_hops": dict(sorted(Counter(r["extra_info"]["hop_type"] for r in test).items()))},
    }
    return dev, test, record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--out", help="write the test split as a parquet (prep_data schema) here")
    ap.add_argument("--check", action="store_true", help="fail unless the committed lists are reproduced")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)

    src = pd_.SOURCES["musique"]
    ds = src.load(cache_dir=args.cache)
    rows, stats = pd_._convert("musique", ds, "test", "validation", 0)
    dev, test, record = build(rows, args.n_test)
    for i, r in enumerate(test):
        r["extra_info"]["index"] = i
    dev_ids = [r["extra_info"]["source_id"] for r in dev]
    test_ids = [r["extra_info"]["source_id"] for r in test]
    if set(dev_ids) & set(test_ids):
        raise SystemExit("dev and test overlap")
    doc = {
        "schema": "voa.splits/v1",
        "source": src.record(name="musique", splits={"validation": stats}),
        **record,
        "files": {"musique_dev.ids": hashlib.sha256(_ids_text(dev_ids).encode()).hexdigest(),
                  "musique_test.ids": hashlib.sha256(_ids_text(test_ids).encode()).hexdigest()},
    }
    if args.check:
        bad = [f for f, ids in (("musique_dev.ids", dev_ids), ("musique_test.ids", test_ids))
               if (SPLITS_DIR / f).read_text() != _ids_text(ids)]
        if bad:
            raise SystemExit(f"committed split lists are not reproduced: {bad}")
        print(f"split lists reproduced: dev {len(dev_ids)}, test {len(test_ids)} {record['test']['by_hops']}")
    else:
        SPLITS_DIR.mkdir(exist_ok=True)
        (SPLITS_DIR / "musique_dev.ids").write_text(_ids_text(dev_ids))
        (SPLITS_DIR / "musique_test.ids").write_text(_ids_text(test_ids))
        (SPLITS_DIR / "SPLITS.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
        print(f"wrote {SPLITS_DIR}: dev {len(dev_ids)}, test {len(test_ids)} {record['test']['by_hops']}")
    if args.out:
        out = dm.write_parquet(args.out, test)
        print(f"wrote {len(test)} test rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
