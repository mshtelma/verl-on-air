"""infra/geo3k/prep_geo3k.py reads geometry3k at a pinned commit and says what it wrote (R17)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import datasets
from PIL import Image

from support import REPO, load_module


def test_prep_reads_the_pinned_commit_and_writes_a_manifest(tmp_path: Path, monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    row = {"problem": "<image>Find x.", "answer": "3", "images": [{"bytes": buf.getvalue(), "path": None}]}
    seen = []

    def load(path, name=None, split=None, revision=None, **kw):
        seen.append((path, revision))
        return datasets.DatasetDict(train=datasets.Dataset.from_list([row] * 3),
                                    test=datasets.Dataset.from_list([row] * 2))
    monkeypatch.setattr(datasets, "load_dataset", load)
    prep = load_module(REPO / "infra" / "geo3k" / "prep_geo3k.py",
                       env_overrides={"GEO3K_OUT_DIR": str(tmp_path), "N_TRAIN": "2", "N_TEST": "0"})
    prep.main()
    assert seen == [("hiyouga/geometry3k", prep.SOURCE.revision)]
    m = json.loads((tmp_path / "DATA_MANIFEST.json").read_text())
    assert m["sources"] == [{"hf_id": "hiyouga/geometry3k", "config": None, "revision": prep.SOURCE.revision,
                             "splits": {"train": 3, "test": 2}}]
    assert {o["name"]: o["rows"] for o in m["outputs"]} == {"train.parquet": 2, "test.parquet": 2}
    assert m["sampling"]["test.parquet"] == "all of test"
