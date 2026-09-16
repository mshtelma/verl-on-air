#!/usr/bin/env python3
"""Verify immutable review bytes/associations, NOT the production R-gate."""
from pathlib import Path
import gzip
import hashlib
import json

ARCHIVE = Path(__file__).resolve().parents[1]
REPO = ARCHIVE.parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(text: str):
    def reject(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(text, parse_constant=reject)


def main():
    manifest = strict_json((ARCHIVE/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (ARCHIVE/name).resolve()
        if not path.is_relative_to(ARCHIVE):
            raise ValueError(f'unsafe manifest path: {name}')
        actual = sha(path.read_bytes())
        if actual != expected:
            raise ValueError(f'archive hash mismatch: {name}: {actual} != {expected}')
    bundles = {}
    for name, info in manifest['bundles'].items():
        raw = gzip.decompress((ARCHIVE/name).read_bytes())
        if sha(raw) != info['uncompressed_sha256']:
            raise ValueError(f'decompressed hash mismatch: {name}')
        records = [strict_json(line) for line in raw.decode().splitlines() if line.strip()]
        ids = [r['uid'] for r in records]
        if len(records) != info['records'] or len(ids) != len(set(ids)):
            raise ValueError(f'bundle cardinality/UID violation: {name}')
        bundles[info['uncompressed_sha256']] = set(ids)
    for run_id in manifest['completed_gate_runs']:
        root = ARCHIVE/'runs'/run_id
        report = strict_json((root/'report.json').read_text())
        raw_scores = (root/'scores.jsonl').read_bytes()
        if sha(raw_scores) != report['artifacts']['scores']['sha256']:
            raise ValueError(f'scores do not match run report: {run_id}')
        scores = [strict_json(line) for line in raw_scores.decode().splitlines() if line.strip()]
        ids = [s['uid'] for s in scores]
        expected = bundles[report['artifacts']['bundle']['sha256']]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(f'run/bundle UID mismatch: {run_id}')
        if report['n_records'] != len(expected) or report['n_scored'] != len(scores):
            raise ValueError(f'run record count mismatch: {run_id}')
        print(f'{run_id}: {len(scores)} scores, bundle/report hashes and UIDs verified; gate={report["overall_pass"]}')
    for name, expected in manifest['external_historical_files'].items():
        if sha((REPO/name).read_bytes()) != expected:
            raise ValueError(f'historical source changed: {name}')
    print(f'Archive integrity verified: {len(manifest["files"])} files. NOT a reward-safety PASS.')


if __name__ == '__main__':
    main()
