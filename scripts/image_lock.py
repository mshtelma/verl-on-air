#!/usr/bin/env python3
"""The image's identity: what it is built from, and which digest a tag was pushed as.

    image_lock.py inputs                         sha256 over the build inputs (the build labels it)
    image_lock.py constraints <freeze.txt>       docker/requirements.lock from an image's freeze
    image_lock.py check-local <tag> <image>      the local image was built from the CURRENT inputs,
                                                 and the tag was not already pushed from other ones
    image_lock.py record <tag> <image>           after `docker push`: write the pushed digest
    image_lock.py check-remote <tag> <image>     the registry serves the digest recorded for the tag

A tag is registered with AI Runtime once and then served by digest, so re-pushing changed content
under the same tag makes a fix "not work" while nothing looks wrong. `make push` therefore requires
check-local, and `make register` check-remote. The build inputs are the files the Dockerfile reads
(docker/*, certs/) plus the build args that change what is installed; the repository code is not
among them -- jobs ship it as a snapshot, and the image carries none of it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "docker" / "IMAGE.lock"
INPUT_FILES = ["docker/Dockerfile", "docker/retry.sh", "docker/uvi.sh", "docker/cccl_probe.cu",
               "docker/requirements.lock", "docker/artifacts.lock"]
# build args that change what the image contains (the index does not: the lock pins versions)
CONTENT_ARGS = ["IMAGE_TAG", "WITH_VIDEO", "OVERRIDE_NCCL", "TORCH_INDEX_URL"]
LABEL = "org.verl-on-air.inputs"
# the packages the constraints leave out on purpose (see the header of docker/requirements.lock)
HAND_ALIGNED = {"nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-cuda-runtime", "nvidia-cuda-nvrtc",
                "opencv-python-headless", "opencv-python"}


class LockError(RuntimeError):
    pass


def inputs_sha256(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    h = hashlib.sha256()
    files = [REPO / f for f in INPUT_FILES]
    certs = REPO / "certs"
    if certs.is_dir():
        files += sorted(p for p in certs.rglob("*") if p.is_file() and p.name != ".gitkeep")
    for p in files:
        h.update(str(p.relative_to(REPO)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    for a in CONTENT_ARGS:
        h.update(f"{a}={env.get(a, '')}".encode() + b"\0")
    return h.hexdigest()


def constraints(freeze_text: str) -> list[str]:
    out = []
    for line in freeze_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"==| @ ", line)[0].strip().lower()
        if " @ " in line or name in HAND_ALIGNED:
            continue
        if "==" not in line:
            raise LockError(f"not an exact pin: {line!r}")
        out.append(line)
    return out


def load() -> dict:
    return json.loads(LOCK.read_text()) if LOCK.is_file() else {"schema": "voa.image_lock/v1", "images": {}}


def save(doc: dict) -> None:
    tmp = LOCK.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, LOCK)


def docker(*args: str) -> str:
    r = subprocess.run(["docker", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise LockError(f"docker {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def local_label(image: str) -> str | None:
    try:
        out = docker("image", "inspect", image, "--format", "{{json .Config.Labels}}")
    except LockError:
        return None
    return (json.loads(out) or {}).get(LABEL)


def repo_of(image: str) -> str:
    return image.rsplit(":", 1)[0] if ":" in image.rsplit("/", 1)[-1] else image


def check_local(tag: str, image: str, current: str) -> list[str]:
    problems = []
    built = local_label(image)
    if built is None:
        problems.append(f"{image} is not a local image built by `make build` (no {LABEL} label): build it")
    elif built != current:
        problems.append(f"{image} was built from other inputs ({built[:12]}) than the current ones "
                        f"({current[:12]}): rebuild it (`make build`)")
    entry = load().get("images", {}).get(tag)
    if entry and entry.get("inputs_sha256") and entry["inputs_sha256"] != current:
        problems.append(f"tag {tag} was already pushed from inputs {entry['inputs_sha256'][:12]} "
                        f"(digest {entry.get('digest', '?')[:19]}): a tag is served by its first digest "
                        "-- give changed content a new tag (`make bump TAG=...`)")
    return problems


def pushed_digest(image: str) -> str:
    repo = repo_of(image)
    out = json.loads(docker("image", "inspect", image, "--format", "{{json .RepoDigests}}")) or []
    for ref in out:
        if ref.split("@", 1)[0] == repo:
            return ref.split("@", 1)[1]
    raise LockError(f"{image} has no registry digest for {repo} -- was it pushed?")


def remote_digest(image: str) -> str:
    out = docker("buildx", "imagetools", "inspect", image)
    m = re.search(r"^Digest:\s+(sha256:[0-9a-f]{64})", out, re.M)
    if not m:
        raise LockError(f"no digest in `docker buildx imagetools inspect {image}` output")
    return m.group(1)


def base_digest() -> str | None:
    m = re.search(r"^FROM\s+\S+@(sha256:[0-9a-f]{64})", (REPO / "docker/Dockerfile").read_text(), re.M)
    return m.group(1) if m else None


def record(tag: str, image: str, current: str) -> dict:
    doc = load()
    digest = pushed_digest(image)
    entry = {"ref": image, "digest": digest, "inputs_sha256": current, "base_digest": base_digest(),
             "recorded_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        entry["created"] = docker("image", "inspect", image, "--format", "{{.Created}}")
        entry["size_bytes"] = int(docker("image", "inspect", image, "--format", "{{.Size}}"))
    except (LockError, ValueError):
        pass
    old = doc.setdefault("images", {}).get(tag)
    if old and old.get("digest") not in (None, digest):
        raise LockError(f"tag {tag} is recorded as {old['digest']}, the push produced {digest}: a tag "
                        "must not be reused for different content")
    doc["images"][tag] = {**(old or {}), **entry}
    save(doc)
    return doc["images"][tag]


def check_remote(tag: str, image: str) -> str:
    entry = load().get("images", {}).get(tag)
    if not entry or not entry.get("digest"):
        raise LockError(f"no digest recorded for {tag} in docker/IMAGE.lock: `make push` records it")
    got = remote_digest(image)
    if got != entry["digest"]:
        raise LockError(f"the registry serves {got} for {image}, but docker/IMAGE.lock records "
                        f"{entry['digest']}: the tag was re-pushed with other content")
    return got


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, args = argv[1], argv[2:]
    try:
        if cmd == "inputs":
            print(inputs_sha256())
        elif cmd == "constraints" and len(args) == 1:
            print("\n".join(constraints(Path(args[0]).read_text())))
        elif cmd == "check-local" and len(args) == 2:
            problems = check_local(args[0], args[1], inputs_sha256())
            for p in problems:
                print(f"STALE: {p}", file=sys.stderr)
            if problems:
                return 1
            print(f"{args[1]} matches the current build inputs")
        elif cmd == "record" and len(args) == 2:
            e = record(args[0], args[1], inputs_sha256())
            print(f"recorded {args[0]} = {e['digest']} in docker/IMAGE.lock")
        elif cmd == "check-remote" and len(args) == 2:
            print(f"{args[1]} is {check_remote(args[0], args[1])}, as recorded")
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except LockError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
