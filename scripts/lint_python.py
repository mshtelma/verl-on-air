#!/usr/bin/env python3
"""Syntax-check every tracked-style Python file under the given roots; report ALL failures.

    python3 scripts/lint_python.py engine infra usecases scripts docs tests conftest.py

Compiles in memory (no __pycache__ written into the tree) and exits 1 if any file fails --
unlike a `for f; do py_compile "$f" && echo ok; done` loop, whose status is only the last file's.
"""
from __future__ import annotations

import sys
from pathlib import Path

SKIP_PARTS = {"__pycache__", ".venv", ".cache", "vendor", "node_modules"}


def files(roots: list[str]) -> list[Path]:
    out: list[Path] = []
    for r in map(Path, roots):
        cands = [r] if r.is_file() else sorted(r.rglob("*.py")) if r.is_dir() else []
        out += [p for p in cands if not SKIP_PARTS & set(p.parts)]
    return out


def main(argv: list[str]) -> int:
    paths = files(argv[1:])
    if not paths:
        print("lint_python: no Python files found under", argv[1:], file=sys.stderr)
        return 1
    bad = 0
    for p in paths:
        try:
            compile(p.read_bytes(), str(p), "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as e:
            bad += 1
            print(f"py FAIL  {p}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"py {'ok  ' if not bad else 'FAIL'}  {len(paths) - bad}/{len(paths)} files compile")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
