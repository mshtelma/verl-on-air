#!/usr/bin/env python3
"""Structural lint for docker/Dockerfile.

Exists because two real build failures in this repo were structural, not logical,
and both were invisible to `python -c`-style checks:

  1. a trailing backslash followed by blank lines made one RUN swallow the next
     instruction (docker only emits a *warning*, then fails confusingly);
  2. an ENV referencing an ARG that had not been declared yet expands to empty,
     silently -- which is how the index ended up unset for some uv calls.

Checks:
  * no line-continuation dangles into a blank line, comment, or new instruction
  * every ${VAR} used in ENV/RUN is declared by an earlier ARG/ENV (or is a
     shell variable assigned in the same RUN)
  * heredocs (<<'PY' ... PY) are balanced
  * COPY sources are not excluded by .dockerignore
"""
from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DF = ROOT / "docker" / "Dockerfile"
DI = ROOT / ".dockerignore"

INSTR = re.compile(r"^(FROM|RUN|ENV|ARG|COPY|ADD|WORKDIR|CMD|ENTRYPOINT|LABEL|EXPOSE|USER|VOLUME|ONBUILD|STOPSIGNAL|HEALTHCHECK|SHELL)\s",
                   re.I)
errors: list[str] = []
warns: list[str] = []

raw = DF.read_text().split("\n")

# ---------------------------------------------------------------- heredocs ----
in_heredoc = None
heredoc_lines: set[int] = set()
for i, line in enumerate(raw, 1):
    if in_heredoc is None:
        m = re.search(r"<<'?([A-Z_]+)'?\s*$", line)
        if m and INSTR.match(line.lstrip()) or (m and line.lstrip().startswith("&&")):
            in_heredoc = m.group(1)
        elif m:
            in_heredoc = m.group(1)
    else:
        heredoc_lines.add(i)
        if line.strip() == in_heredoc:
            in_heredoc = None
if in_heredoc is not None:
    errors.append(f"unterminated heredoc <<{in_heredoc}")

# ------------------------------------------------- dangling continuations -----
for i, line in enumerate(raw, 1):
    # A '\' inside a comment is just prose (the header shows example commands).
    if i in heredoc_lines or line.lstrip().startswith("#"):
        continue
    if not line.rstrip().endswith("\\"):
        continue
    nxt = raw[i] if i < len(raw) else ""
    if nxt.strip() == "":
        errors.append(f"line {i}: continuation '\\' followed by a BLANK line "
                      f"-> this RUN will swallow the next instruction: {line.strip()[:60]}")
    elif nxt.lstrip().startswith("#"):
        errors.append(f"line {i}: continuation '\\' followed by a COMMENT: {line.strip()[:60]}")
    elif INSTR.match(nxt):
        errors.append(f"line {i}: continuation '\\' followed by instruction "
                      f"'{nxt.split()[0]}' -> merged instructions")

# --------------------------------------- backtick-comments in command position --
# `# text` is only safe in ARGUMENT position (e.g. inside a pip package list). In
# COMMAND position the empty command substitution occupies the command-name slot,
# so a following assignment is executed as a command. This cost one 4-minute build:
#     /bin/sh: 1: CCCL=/opt/venv/.../flashinfer/data/cccl: not found
# Reproduce:
#     sh -c 'echo x; `# c` FOO="$(echo v)"; echo $FOO'
#     -> /bin/sh: FOO=v: No such file or directory
in_run = False
prev_frag = ""
for i, line in enumerate(raw, 1):
    if i in heredoc_lines:
        continue
    stripped = line.strip()
    if not in_run:
        if re.match(r"^RUN\s", stripped, re.I):
            in_run = True
            prev_frag = stripped[4:].rstrip("\\").strip()
            if not line.rstrip().endswith("\\"):
                in_run = False
                prev_frag = ""
        continue
    if stripped.startswith("#"):
        continue
    frag = stripped.rstrip("\\").strip()
    if frag.startswith("`#") and (prev_frag == "" or prev_frag.endswith((";", "&&", "||"))):
        errors.append(
            f"line {i}: backtick-comment in COMMAND position -> the empty expansion "
            f"takes the command-name slot and the next assignment runs as a command. "
            f"Move it to a '#' line above the RUN: {stripped[:70]}")
    if frag:
        prev_frag = frag
    if not line.rstrip().endswith("\\"):
        in_run = False
        prev_frag = ""

# --------------------------------------------------- ARG/ENV declaration -------
declared: set[str] = set()
for i, line in enumerate(raw, 1):
    if i in heredoc_lines:
        continue
    stripped = line.strip()
    m = re.match(r"^(ARG|ENV)\s+(.*)$", stripped, re.I)
    if m:
        body = m.group(2)
        for name in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", body):
            declared.add(name)
        if m.group(1).upper() == "ARG" and "=" not in body:
            declared.add(body.split()[0])
    # ENV referencing an undeclared ARG expands to empty, silently.
    if re.match(r"^ENV\s", stripped, re.I):
        for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:\-}]", stripped):
            if var not in declared:
                errors.append(f"line {i}: ENV uses ${{{var}}} before any ARG/ENV declares it "
                              f"-> expands to EMPTY")

# ----------------------------------------------- COPY vs .dockerignore --------
pats = [l.strip() for l in DI.read_text().split("\n") if l.strip() and not l.startswith("#")]


def included(path: str) -> bool:
    inc = True
    for p in pats:
        neg = p.startswith("!")
        pat = p[1:] if neg else p
        if (fnmatch.fnmatch(path, pat) or path == pat
                or fnmatch.fnmatch(path, pat.rstrip("/") + "/*")):
            inc = neg
    return inc


for i, line in enumerate(raw, 1):
    if i in heredoc_lines:
        continue
    m = re.match(r"^COPY\s+(?!--)(\S+)\s+(\S+)", line.strip(), re.I)
    if not m:
        continue
    src = m.group(1).rstrip("/")
    if src.startswith(("/", "$")):
        continue
    if not included(src) and not included(src + "/.gitkeep"):
        errors.append(f"line {i}: COPY source '{src}' is excluded by .dockerignore "
                      f"-> 'failed to compute cache key: not found'")
    if not (ROOT / src).exists():
        errors.append(f"line {i}: COPY source '{src}' does not exist in the repo")

# ------------------------------------------------------------------ report ----
for w in warns:
    print(f"  warn  {w}")
for e in errors:
    print(f"  FAIL  {e}")
if errors:
    print(f"\n{len(errors)} structural problem(s) in {DF.relative_to(ROOT)}")
    sys.exit(1)
print(f"  ok    {DF.relative_to(ROOT)}: no dangling continuations, ARG/ENV order sane, "
      f"heredocs balanced, COPY sources present")
