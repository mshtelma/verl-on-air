# Synthetic hard-question difficulty probe (Phase-2 in-band filter)

Probes the 150 synthetic hard questions (`hard_synth_pilot_150.jsonl`) by running the BASE model
(Qwen3.5-35B-A3B) as an agent @ 80 turns, TEMPERATURE 0.7, 8 samples/question, and measuring the
deterministic answer-correctness PASS-RATE per question. Questions the base always solves (p~1) or
never solves (p~0) give no GRPO gradient; the learnable band is the middle.

Tooling: `scripts/officeqa/hard_synth_probe.py` (`to-csv` / `aggregate` / `score`).
Configs: `air/113` (smoke, 48 ep) and `air/114` (full, 1200 ep).

## Files here
- `hard_synth_probe_key.json` — `{key: {uid: answer_float}, base: {HS####: {answer, template,
  question, n}}}`. The uid→answer map + per-question metadata the scorer needs.
- `hard_synth_probe_learnable.json` — `{learnable: [HS####...], rates: {HS####: [pass_rate,
  n_samples, template]}}`. The probe RESULT: per-question base@80 pass-rate + the learnable set.

The full capture records (1094 salvaged episodes, ~268 MB with raw_turns) are NOT committed; they
live on the Volume at `.../eval/path_report/deliverable_b/probe_synth_hard_full/` (per-episode
`parts/*.json` + consolidated `captures.jsonl`).

## Result (base@80, 1094 salvaged episodes; run 175750633481200)
138/150 questions scored (12 no_data = tail HS0138-0149, unreached; recoverable via a resume
top-up run). Of the 138:
- too_easy (p>=0.95): 17
- **learnable (0.05<p<0.95): 95**  <- Phase-2 trainable core
- too_hard (p<=0.05): 26
mean pass-rate 0.511; healthy difficulty gradient; learnable balanced across all 6 templates.

The run FAILED as a job (see below) but 91% of episodes were durably salvaged via the collector's
per-episode `parts/` writes -- the whole point of that design.

## Recover / finish
- Aggregate salvaged parts: `hard_synth_probe.py aggregate --run-dir <run_dir>` -> captures.jsonl.
- Score: `hard_synth_probe.py score --captures <run_dir-or-jsonl> --key-json hard_synth_probe_key.json`.
- Top-up the 12 no_data: re-run `air/114` pointing at the SAME `OQ_COLLECT_OUT` -- the collector
  now RESUMES (skips questions already in parts/, `OQ_COLLECT_RESUME` default on) and runs only the
  missing ~106 episodes.
