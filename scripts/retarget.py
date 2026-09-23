#!/usr/bin/env python3
"""Point every job file at the values in config.env -- the one file you customise.

The job YAMLs carry concrete values so each stays readable and hand-submittable. This
rewrites them from config.env:

  environment.docker_image.url   <- $(DOCKERHUB_USER)/$(IMAGE_NAME):$(IMAGE_TAG)

Stock-environment jobs (`environment.version`, no image) are reported and left alone.

    python3 scripts/retarget.py                   # rewrite; fail if any job still disagrees
    python3 scripts/retarget.py --expect-change   # ...and fail if nothing needed rewriting
    python3 scripts/retarget.py --check           # no writes; fail if any job disagrees

Only the value on the matched line is edited (comments and layout kept); each rewritten
file is re-parsed to prove the field now holds the intended value. Matching is by FIELD,
not by the old string: a job pointing at any other image is retargeted too.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
_ASSIGN = re.compile(r"^([A-Z][A-Z0-9_]*)\s*[:?]?=\s*(.*?)\s*$")
_URL_LINE = re.compile(r"^(?P<lead>\s+url:[ \t]*)(?P<q>['\"]?)(?P<url>[^\s'\"#]+)(?P=q)(?P<tail>.*)$")


def read_config_env(path: Path) -> dict[str, str]:
    """The plain KEY=value lines of config.env (it is Make syntax, not a shell file)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _ASSIGN.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def job_files(root: Path) -> list[Path]:
    fs = set(root.glob("infra/**/air/*.yaml")) | set(root.glob("usecases/*/air/*.yaml"))
    return sorted(f for f in fs if not f.name.startswith(".probe_"))


def rewrite_image(text: str, new: str) -> tuple[str, str | None]:
    """Replace the value of the `url:` directly under `docker_image:`. -> (text, old url)."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        m = _URL_LINE.match(line.rstrip("\n"))
        if not m:
            continue
        j = i - 1
        while j >= 0 and (not lines[j].strip() or lines[j].lstrip().startswith("#")):
            j -= 1
        if j < 0 or lines[j].strip() != "docker_image:":
            continue
        lines[i] = f"{m['lead']}{m['q']}{new}{m['q']}{m['tail']}" + ("\n" if line.endswith("\n") else "")
        return "".join(lines), m["url"]
    return text, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO), help="repo root (for tests)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report drift, write nothing")
    mode.add_argument("--expect-change", action="store_true", help="fail if nothing was rewritten")
    args = ap.parse_args(argv)

    root = Path(args.root)
    cfg = read_config_env(root / "config.env")
    missing = [k for k in ("DOCKERHUB_USER", "IMAGE_NAME", "IMAGE_TAG") if not cfg.get(k)]
    if missing:
        print(f"retarget: config.env is missing {', '.join(missing)}", file=sys.stderr)
        return 1
    image = f"{cfg['DOCKERHUB_USER']}/{cfg['IMAGE_NAME']}:{cfg['IMAGE_TAG']}"

    custom, changed, stock, bad = 0, [], [], []
    for f in job_files(root):
        rel = f.relative_to(root)
        text = f.read_text()
        env = (yaml.safe_load(text) or {}).get("environment") or {}
        if "docker_image" not in env:
            stock.append(rel)
            continue
        custom += 1
        cur = (env.get("docker_image") or {}).get("url")
        if cur == image:
            continue
        if args.check:
            bad.append(f"{rel}: {cur} (config.env says {image})")
            continue
        new_text, old = rewrite_image(text, image)
        new_env = (yaml.safe_load(new_text) or {}).get("environment") or {}
        if old is None or (new_env.get("docker_image") or {}).get("url") != image:
            bad.append(f"{rel}: could not rewrite environment.docker_image.url ({cur})")
            continue
        f.write_text(new_text)
        changed.append(rel)
        print(f"  updated {rel}: {old} -> {image}")

    if not custom:
        print("retarget: no job file uses a custom image -- wrong --root?", file=sys.stderr)
        return 1
    for b in bad:
        print(f"  MISMATCH {b}", file=sys.stderr)
    print(f"{custom - len(bad)}/{custom} custom-image jobs at {image} ({len(changed)} rewritten); "
          f"{len(stock)} stock-environment job(s) have no image to set")
    if bad:
        print("retarget: job files disagree with config.env -- run `make retarget`" if args.check
              else "retarget: FAILED to retarget every job", file=sys.stderr)
        return 1
    if args.expect_change and not changed:
        print("retarget: nothing needed rewriting -- every job already names this image. Did the "
              "tag in config.env actually change?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
