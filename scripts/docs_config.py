#!/usr/bin/env python3
"""Generate docs/configuration.md's table of every typed engine knob from the preflight schema.

    python3 scripts/docs_config.py            # rewrite the generated block (make docs-config)
    python3 scripts/docs_config.py --check    # exit 1 if the doc has drifted (make lint)

engine/lib/preflight.py's KNOBS is what the launchers actually accept: a knob's type, the mode(s)
that read it, and its bounds or allowed values. The prose sections explain what the knobs do; this
block, between the BEGIN/END markers, is the complete list, so the two cannot disagree about what
exists.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "configuration.md"
BEGIN = "<!-- BEGIN GENERATED: knobs (scripts/docs_config.py; do not edit by hand) -->"
END = "<!-- END GENERATED: knobs -->"


def _preflight():
    spec = importlib.util.spec_from_file_location("_voa_preflight", REPO / "engine/lib/preflight.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolve their module while it loads
    spec.loader.exec_module(mod)
    return mod


def _num(x: float) -> str:
    return f"{int(x)}" if float(x).is_integer() else f"{x:g}"


def allowed(k) -> str:
    if k.kind == "enum":
        return " \\| ".join(f"`{c}`" for c in k.choices)
    if k.kind == "bool":
        return "`True` \\| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`)"
    if k.kind in ("int", "float"):
        lo = "" if k.lo is None else (f"> {_num(k.lo)}" if k.lo_open else f"≥ {_num(k.lo)}")
        hi = "" if k.hi is None else f"≤ {_num(k.hi)}"
        return ", ".join(x for x in (lo, hi) if x) or "any"
    return "any string"


def render() -> str:
    pf = _preflight()
    rows = ["| knob | type | read by | allowed |", "|---|---|---|---|"]
    for name in sorted(pf.KNOBS):
        k = pf.KNOBS[name]
        modes = "both" if set(k.modes) == {"sync", "async"} else " + ".join(k.modes)
        rows.append(f"| `{name}` | {k.kind} | {modes} | {allowed(k)} |")
    return "\n".join([BEGIN, "", *rows, "", END])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = DOC.read_text()
    if BEGIN not in text or END not in text:
        print(f"docs_config: {DOC.relative_to(REPO)} has no generated block ({BEGIN})", file=sys.stderr)
        return 1
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + render() + tail
    if args.check:
        if new != text:
            print("docs_config: docs/configuration.md's knob table is out of date with "
                  "engine/lib/preflight.py -- run `make docs-config`", file=sys.stderr)
            return 1
        print("docs ok  configuration.md knob table matches the preflight schema")
        return 0
    DOC.write_text(new)
    print(f"docs_config: wrote the knob table ({len(_preflight().KNOBS)} knobs) into {DOC.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
