"""make_splits.py: the dev/test question sets are fixed, disjoint, stratified, and committed."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

from support import USECASES, load_usecase

ms = load_usecase("agentic-search", "make_splits")
SPLITS = USECASES / "agentic-search" / "splits"


def _rows(n2: int, n3: int, n4: int) -> list[dict]:
    hops = ["2hop"] * n2 + ["3hop"] * n3 + ["4hop"] * n4
    return [{"extra_info": {"source_id": f"{h}__{i}", "hop_type": h}} for i, h in enumerate(hops)]


def test_allocation_is_proportional_and_exact():
    assert ms.allocate({"2hop": 752, "3hop": 760, "4hop": 405}, 500) == {"2hop": 196, "3hop": 198, "4hop": 106}
    assert sum(ms.allocate({"a": 1, "b": 1, "c": 1}, 2).values()) == 2
    with pytest.raises(SystemExit):
        ms.allocate({"a": 3}, 4)


def test_dev_is_the_first_rows_and_test_comes_after_the_excluded_prefix():
    rows = _rows(1500, 700, 300)
    dev, test, rec = ms.build(rows, 250)
    assert dev == rows[:ms.DEV_N]
    first = {r["extra_info"]["source_id"] for r in rows[:ms.EXCLUDE_FIRST]}
    assert not first & {r["extra_info"]["source_id"] for r in test}
    assert len(test) == 250 and rec["test"]["by_hops"] == dict(Counter(r["extra_info"]["hop_type"] for r in test))
    assert ms.build(rows, 250)[1] == test, "not deterministic"
    idx = [rows.index(r) for r in test]
    assert idx == sorted(idx), "test rows keep source order"


def test_the_committed_lists_match_their_record():
    doc = json.loads((SPLITS / "SPLITS.json").read_text())
    dev = (SPLITS / "musique_dev.ids").read_text()
    test = (SPLITS / "musique_test.ids").read_text()
    assert hashlib.sha256(dev.encode()).hexdigest() == doc["files"]["musique_dev.ids"]
    assert hashlib.sha256(test.encode()).hexdigest() == doc["files"]["musique_test.ids"]
    assert len(dev.split()) == doc["dev"]["n"] == 200 and len(test.split()) == doc["test"]["n"] == 500
    assert not set(dev.split()) & set(test.split())
    assert doc["source"]["revision"] == "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
    assert Counter(i.split("__")[0][:4] for i in test.split()) == Counter(doc["test"]["by_hops"])
