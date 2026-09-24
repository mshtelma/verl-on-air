#!/usr/bin/env python3
"""Point every job file at the values in config.env -- the one file you customise.

The job YAMLs carry concrete values so each stays readable and hand-submittable. This
rewrites them from config.env:

  environment.docker_image.url   <- $(DOCKERHUB_USER)/$(IMAGE_NAME):$(IMAGE_TAG)   (custom-image jobs)
  every /Volumes/<catalog>/<schema>/<volume> prefix  <- /Volumes/$(UC_CATALOG)/$(UC_SCHEMA)/$(UC_VOLUME)
  QA_VS_ENDPOINT / QA_VS_INDEX   <- $(VS_ENDPOINT) / $(VS_INDEX)                  (when config.env sets them)

Stock-environment jobs (`environment.version`, no image) keep their environment; their Volume
paths are retargeted like every other job's. The job files must agree on ONE Volume prefix before
a rewrite: a mix means someone edited some of them by hand, and guessing which is right is not
this script's job.

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
_VOL = re.compile(r"/Volumes/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+(?=[/'\"\s]|$)")
_ENV_KEYS = {"QA_VS_ENDPOINT": "VS_ENDPOINT", "QA_VS_INDEX": "VS_INDEX"}   # job env var <- config.env key


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


def volume_prefixes(text: str) -> set[str]:
    return set(_VOL.findall(text))


def rewrite_workspace(text: str, vol: str, env: dict[str, str]) -> tuple[str, list[str]]:
    """Retarget the Volume prefix and the env values in `env` (job var -> value). -> (text, changes)."""
    changes = [f"{old} -> {vol}" for old in sorted(volume_prefixes(text)) if old != vol]
    text = _VOL.sub(vol, text)
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        for var, new in env.items():
            m = re.match(rf"^(\s+{var}:[ \t]*)(['\"]?)([^\s'\"#]+)\2(.*?)(\n?)$", line)
            if m and m.group(3) != new:
                changes.append(f"{var}: {m.group(3)} -> {new}")
                lines[i] = f"{m.group(1)}{m.group(2)}{new}{m.group(2)}{m.group(4)}{m.group(5)}"
    return "".join(lines), changes


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
    vol = "/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}".format(
        **{k: cfg.get(k, "") for k in ("UC_CATALOG", "UC_SCHEMA", "UC_VOLUME")})
    if not _VOL.fullmatch(vol):
        print(f"retarget: config.env's UC_CATALOG/UC_SCHEMA/UC_VOLUME give no Volume path ({vol})",
              file=sys.stderr)
        return 1
    env_values = {var: cfg[key] for var, key in _ENV_KEYS.items() if cfg.get(key)}

    files = job_files(root)
    prefixes = sorted(set().union(*(volume_prefixes(f.read_text()) for f in files)))
    if len(prefixes) > 1 and not args.check:
        print("retarget: the job files use more than one Volume prefix -- fix them by hand first:\n  "
              + "\n  ".join(f"{p_}: {', '.join(str(f.relative_to(root)) for f in files if p_ in f.read_text())}"
                             for p_ in prefixes), file=sys.stderr)
        return 1

    custom, changed, stock, bad = 0, [], [], []
    for f in files:
        rel = f.relative_to(root)
        text = f.read_text()
        new_text, ws_changes = rewrite_workspace(text, vol, env_values)
        if ws_changes:
            if args.check:
                bad += [f"{rel}: {c}" for c in ws_changes]
            else:
                yaml.safe_load(new_text)   # still YAML
                f.write_text(new_text)
                text = new_text
                if rel not in changed:
                    changed.append(rel)
                for c in ws_changes:
                    print(f"  updated {rel}: {c}")
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
        if rel not in changed:
            changed.append(rel)
        print(f"  updated {rel}: {old} -> {image}")

    if not custom:
        print("retarget: no job file uses a custom image -- wrong --root?", file=sys.stderr)
        return 1
    for b in bad:
        print(f"  MISMATCH {b}", file=sys.stderr)
    print(f"{custom}/{custom} custom-image jobs checked against {image}; every job's Volume prefix "
          f"checked against {vol}{' and ' + ', '.join(env_values) if env_values else ''} "
          f"({len(changed)} file(s) rewritten); {len(stock)} stock-environment job(s) have no image")
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
