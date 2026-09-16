# OfficeQA recovery plan — superseded by the isolated path-report test

**Direction changed on 2026-09-14 UTC.** This filename is retained for existing links
and historical source comments; it is no longer an R0–R11 implementation backlog.

## Current plan

Read **`docs/officeqa_path_report_pilot.md`**.
Next-agent assignment: **`docs/officeqa_rgate_handoff.md`**.
Broader conditional roadmap: `docs/officeqa_rl_plan.md`.

The next task is an isolated, inference-only test on approximately 20 real easy
questions: the actor returns an answer plus a small path report; the runtime records
actual tool calls/results/code; simple checks and full GLM-5.3 TP16 audit the report.
Confirmed fabricated or unsupported paths receive zero candidate reward. Valid paths
remain graded. No optimizer is connected to these offline scores.

Do **not** first implement a universal table IR, TaskSpec ontology, cell/slot binding
registry, arithmetic proof engine or judge-authored certificate. Do not make the former
training-framework packages prerequisites for this offline experiment. Those architectural
choices are superseded, not merely postponed as the default next phase.

The original reward-safety, data-quality and training-integration failures remain real.
A small pilot is not Gate-R clearance or permission to train. Future integration
requirements are summarized separately in the broader roadmap.

Exact pre-pivot documents are preserved in
`docs/history/officeqa_pre_path_report_2026_09_14/`.
The sealed evidence archive remains unchanged at
`officeqa_pilot_records/rgate_review_2026_09_14/`.
