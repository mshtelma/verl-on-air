#!/usr/bin/env python3
"""Assemble the held-out evidence file: paired comparisons on the test split, with and without tools.

    python3 scripts/heldout_package.py --dir <local copy of .../eval/heldout> --out results/agentic-search/2026-09-heldout-test.json

Reads the base and checkpoint artifacts named in PAIRS (both evaluated with tools, and closed-book),
runs scripts/paired_eval.py's statistics on each pair, adds per-hop counts, answer rates and tool
use, and records every artifact's sha256 and the split's record (splits/SPLITS.json).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import paired_eval as pe  # noqa: E402

# policy -> (base artifact, [(label, checkpoint artifact, how it was chosen)])
PAIRS = {
    "tools": ("test_base_tools.json", [
        ("pureem_step20", "test_pureem_step20_tools.json", "chosen on the dev set (best of 13)"),
        ("seed7_step20", "test_seed7_step20_tools.json", "pre-registered before training (511b368)"),
    ]),
    "closed_book": ("test_base_closedbook.json", [
        ("pureem_step20", "test_pureem_step20_closedbook.json", "chosen on the dev set (best of 13)"),
        ("seed7_step20", "test_seed7_step20_closedbook.json", "pre-registered before training (511b368)"),
    ]),
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(d: dict) -> dict:
    scored = [r for r in d["results"] if r["status"] == "scored"]
    hops = Counter(r["hop_type"][:4] for r in scored)
    right = Counter(r["hop_type"][:4] for r in scored if r["correct"])
    return {"model": d.get("model"), "valid": d["valid"], "n_scored": d["n_scored"], "em": d["em"],
            "answered": d["answered"], "mean_tool_calls": d["mean_tool_calls"],
            "by_hops": {h: {"n": hops[h], "correct": right[h]} for h in sorted(hops)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    doc = {"note": ("Held-out MuSiQue test split (500 questions from validation rows 500.., stratified by "
                    "hop count; usecases/agentic-search/splits/), each model scored ONCE per policy. "
                    "Nominal paired statistics: the pure-EM checkpoint was chosen on the dev set, the "
                    "seed-7 checkpoint was fixed before its run -- neither was chosen on these numbers."),
           "split": json.loads((REPO / "usecases/agentic-search/splits/SPLITS.json").read_text()),
           "policies": {}}
    for policy, (base_name, ckpts) in PAIRS.items():
        present = [(label, name, chosen) for label, name, chosen in ckpts if (d / name).is_file()]
        base = json.loads((d / base_name).read_text())
        entry = {"base": {"artifact": base_name, "sha256": _sha(d / base_name), **summary(base)},
                 "eval_policy": base["eval_policy"],
                 "missing": [name for _, name, _ in ckpts if not (d / name).is_file()], "checkpoints": []}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "paired.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = pe.main([str(d / base_name), *[str(d / n) for _, n, _ in present], "--json-out", str(out)])
            if rc:
                raise SystemExit(f"paired_eval failed for {policy}")
            paired = json.loads(out.read_text())
        for (label, name, chosen), stats in zip(present, paired["checkpoints"]):
            art = json.loads((d / name).read_text())
            stats["artifact"] = name
            entry["checkpoints"].append({"label": label, "chosen": chosen, **summary(art), "paired": stats})
        entry["method"] = paired["method"]
        doc["policies"][policy] = entry
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
