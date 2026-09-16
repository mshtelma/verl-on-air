# Superseded OfficeQA documents — before the path-report pivot

**Historical snapshots only. Do not implement these as the current plan.**
The owner replaced the universal table/proof-engine direction with a small isolated
actor path-report experiment on 2026-09-14 UTC.

Current plan: `docs/officeqa_path_report_pilot.md`.
Current handoff: `docs/officeqa_rgate_handoff.md`.

`documents.tar.gz` contains exact pre-edit bytes for the four former OfficeQA design/
plan/handoff documents, the pilot-record README and the project README. `manifest.json`
records original paths, SHA-256 values, capture time and compressed-archive hash.
Original paths and links inside the snapshots refer to the repository as it stood
before the pivot. Some status/design claims were already stale at capture; preservation
is not endorsement.

For inspection, extract into a fresh scratch directory, **never over the working tree**:

```bash
D=$(mktemp -d)
tar -xzf docs/history/officeqa_pre_path_report_2026_09_14/documents.tar.gz -C "$D"
printf 'Historical copies extracted to %s\n' "$D"
```

The separate sealed review at `officeqa_pilot_records/rgate_review_2026_09_14/` was not
modified. Its reports, bundles, scores, source snapshots and reproducers remain the
authority for the measured historical failures, not for the new implementation direction.
