# OfficeQA handoff — build the isolated path-report test

**Direction updated:** 2026-09-14 UTC.
Repository: `/home/michael.shtelma/verl-on-air`.

## 1. Your assignment

**Implement a small inference-only experiment, not the former R0–R11 recovery plan.**
Have the actor return an answer plus a structured path report. Record its actual tool
calls, delivered outputs and executed Python independently. Check the report against
that history, then use the full GLM-5.3 TP16 judge to assess semantic support.

The owner explicitly rejected building a universal table/hierarchy object model and
proof engine first. Structured self-reporting is the prospective RL capability, but this
initial test has **no training, SFT, optimizer, parameter sync or checkpoint work**. It evaluates
whether the interface and auditing approach are useful before proposing RL integration.

**Pilot code and results do not exist yet.** The current turn changed documentation
only. Remote GPU execution/budget authorization must be confirmed separately; this
handoff does not authorize an automatic job launch.

Read next: **`docs/officeqa_path_report_pilot.md`**. It is the single authoritative
schema, verification contract, case design, file map and stopping policy.
`docs/officeqa_rl_plan.md` contains the short conditional roadmap. Read the historical
review only as needed for a concrete failure—not as a prerequisite to a large rewrite.

## 2. The design to implement

- **One terminal assistant JSON object:** `answer` plus `path`; each step names an
  actual observation, a short claim, and optional dependencies on earlier steps.
- **Small runtime record:** episode-scoped observation IDs, actual tool names/arguments,
  executed code, outcomes and exact payloads delivered to the actor. Do not infer
  provenance from printable markers or trust report-provided copies of tool output.
- **Simple checks:** valid submission, unique IDs, valid dependencies, references that
  resolve to this episode's delivered observations. No table IR or proof-plan registry.
- **Semantic audit:** full GLM-5.3 TP16 checks values, source hierarchy/period/units,
  computation linkage and necessary input coverage against the actual history.
- **Offline reward preview:** wrong answer or confirmed invalid/fabricated path = zero,
  even if the numerical answer happens to be correct; supported correct answers remain
  graded; verifier failure/uncertainty = UNKNOWN, not an ordinary negative.

Do not add giant negative penalties, tool-count/style bonuses, a literal-gold-in-quote
rule, gold-file requirements or a prescribed traversal. Do not reward the report's
plausibility without checking its evidence. A valid report does not prove hidden
internal causality; earlier failed exploration is not itself misconduct.

Start with final JSON. A special `submit_answer` tool or incremental evidence notes
can be a later experiment, not another prerequisite. The judge checks an actor-authored
report; it does not write the report on the actor's behalf.

## 3. First actions, in order

1. Inspect `git status --short`; preserve the extensive pre-existing uncommitted tree.
   Inherit this working tree, not just the old git HEAD. **No commits unless asked.**
2. Read the pilot plan and inspect these existing seams:
   - `scripts/eval_officeqa_agentic.py`: actor client, rendering, dispatch and termination.
   - `scripts/tools/officeqa_tools.py`: the five existing tools and their plain `_impl`
     functions; preserve their semantics rather than inventing a document ontology.
   - `scripts/tools/py_compute.py`: bubblewrap scientific Python; no unsafe fallback.
   - `scripts/serve_judge.sh`: full TP16 serving, not a license to reuse the old scorer.
3. Add the small proposed `scripts/officeqa/path_report.py`, isolated
   `scripts/officeqa/path_report_pilot.py`, and
   `scripts/officeqa/tests/test_path_report.py`. Keep legacy defaults unchanged.
4. Run CPU fixtures for report parsing, real/fake/cross-episode/dropped references,
   source versus compute authority, tool-emitted final JSON, judge faults and incomplete/
   duplicate/nonfinite result sets. Mocked examples are software tests, not real tasks.
5. Prepare the real easy-question selection and fresh-capture plan. Inspect two or
   three episodes from its calibration subset, then complete the declared 20-question
   collection; use at most 30 for documented coverage. Preserve every attempt/failure.
6. After authorization, collect with the actual actor, review labels, then judge offline.
   Rollout and full-TP16 judging may be sequential; no live GRPO topology is required.
7. Produce a case-level report and **stop for owner review**. Do not automatically
   advance to training, mass synthesis or another full R-gate run.

The prototype must handle its own terminal JSON instruction/nudge. The old eval helper
forces `<FINAL_ANSWER>` at the turn cap; do not let it overwrite, synthesize or ambiguously
extract the new submission. Capture after actual actor-request construction and log
context/turn exhaustion. Never treat undispatched/unseen results as supporting evidence.

## 4. The test must distinguish these questions

1. **Answer correctness:** is the final result correct under the actual question?
2. **Report faithfulness:** does the report accurately describe actual observations and
   computations? A truthful report may describe an unsuccessful solution.
3. **Answer support:** does the declared evidence/derivation justify the answer?

Create clearly labeled report variants holding the answer and actual trace fixed where
possible: fake observation, wrong returned value, wrong table/period/category, different
computation, missing source inputs. Add valid alternatives, derived values, sufficient
snippets and later corrected paths so the test cannot pass by rejecting everything.

Labels must be reviewed independently of the candidate judge. Do not copy the old
317/339-case labels, infer unsupported retrieval from an original wrong answer, or
fabricate alternate filenames. Investigator-written reports are verifier fixtures,
not evidence the actor produced a faithful report naturally.

A trace-only versus trace-plus-report comparison is useful on their **shared answer-
support target**. Report-only fabrication is a separate metric: do not count a trace-
only judge as wrong for a report it never received. Follow the pilot's denominator rules.

## 5. Existing state and constraints to preserve

- **No grounded reward is approved for policy optimization.** Unsafe v2.1 auditing
  remains; do not wire the pilot into `compute_score`, `rgate_run.py` or the trainer.
- Partial protocol guard helpers exist in `officeqa_grounded_reward.py`, but at this
  documentation check they were **not called by `compute_score` or the launchers**.
  Do not claim enforcement or completed R1 from their definitions/docstrings. Their
  old `officeqa_proof_v1` name is not this pilot's protocol.
- Historical answer-mode run `616024033111491` completed two optimizer steps. It did
  not establish grounded learning, actual UNKNOWN-group exclusion or checkpoint saving.
- The current eval writer retains full tool-return strings; original saved traces
  remain historically 3k-clipped. Fresh actor-visible observation IDs/capture are not
  already implemented merely because `tool_calls` and `tool_results` exist.
- Keep bubblewrap and numpy/pandas/scipy. Never execute agent code on the host or enable
  `OQ_COMPUTE_ALLOW_UNSANDBOXED` to get the pilot running.
- The current local corpus directories and Volume CSV were absent when inspected.
  CPU fixture work can start; real collection needs existing data/corpus staged or a
  verified suitable runtime. Do not substitute synthetic questions for missing data.
- Training metadata transport, all-loss UNKNOWN group exclusion, async backpressure,
  answer-parser gaps and independent validation remain future RL concerns. They are
  **not** prerequisites to this standalone offline test, and the test does not fix them.

## 6. Existing operational references — not launch commands

CLI: `~/.local/bin/air`; Databricks profile: `df1`.
Last exercised image: `michaelshtelma587/verl-megatron-air:v7`.
Verify actual model/image/config identities before a new run.

```text
/Volumes/main/mshtelma/verl/data/officeqa/officeqa_full.csv
/Volumes/main/mshtelma/verl/data/officeqa/treasury_bulletins_clean.zip
/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B
/Volumes/main/mshtelma/verl/models/GLM-5.3
```

Use the clean corpus, not `treasury_bulletins_transformed.zip`; even clean output may
have semantic extraction defects. Keep private gold/reference hints outside actor tools.

Inspect `air/72_eval_officeqa_trace.yaml` and `air/73_officeqa_grounding_judge.yaml`
only for serving/staging patterns. A new pilot entry point/recipe must explicitly
select the new report evaluation, not the old reward assembly. **Do not relaunch air/82
unchanged, launch air/77, or use a training job to collect this pilot.**

The last archived API check, **2026-09-14T10:58:22Z**, found all reviewed jobs terminal.
Air/82 attempt 4 (`543029753940588`) completed 339 cases and FAILED: controls 16/80,
labeled-negative accepts 5/242, case upper95 4.295%; 55 control rejections were caused
by literal-value matching. Labels were also defective. Those are historical results,
not a result for the proposed path-report experiment. No fresh status poll was made
for this documentation rewrite.

## 7. Deliverable and stop condition

Write to a new run-specific `officeqa_pilot_records/path_report_pilot/<run-id>/` directory
(or corresponding unique Volume path). Save selected UIDs/labels, exact actor/tool
records, reports, judge requests/replies, scores/statuses, versions and costs. Never
overwrite the old shared `officeqa_traces.jsonl` or `rgate_expanded*` outputs.

At takeover, this results directory and prototype scripts **do not exist**. Do not
publish executable pilot commands until the implementation is present and tested.

The final result should state:

- reporting success/abstention/malformed rates and actual question/episode counts;
- valid/invalid path acceptance and UNKNOWN counts by family;
- whether reports help, with all material judge disagreements inspected;
- whether the evidence supports stopping, one narrow refinement, or proposing a
  separate small RL test—not automatic approval to run that test.

A 20–30-question study is not a <=2% false-accept guarantee, a full R-gate pass or proof
of learning. Report insufficient label/positive coverage as inconclusive. Preserve the
hard set's prior evaluator-development exposure; do not call it untouched.

Useful existing read-only checks:

```bash
git status --short
PYTHONDONTWRITEBYTECODE=1 python3 officeqa_pilot_records/rgate_review_2026_09_14/tools/verify_archive.py
git diff --check
```

The sealed review archive and frozen reproducers remain intact. Pre-pivot documents
are compressed under `docs/history/officeqa_pre_path_report_2026_09_14/`; they are
historical evidence, not your implementation assignment. Update this handoff with
actual new paths/commands/results when the isolated prototype is built.

## 8. Phase-1 (CPU prototype) status — landed 2026-09-14 UTC

**Step 1 of the pilot plan's execution order (Section 7, "CPU prototype") is implemented
and tested on CPU only.** Steps 2–4 (fresh actor collection, live GLM-5.3 TP16 judging,
reviewed labels, the case-level report) are **not started** and remain gated on separate
GPU/budget authorization. No trainer, reward manager, `compute_score`, launcher,
checkpoint or R-gate code was touched; no job was launched; the sealed archive is
unchanged; these scores are offline previews, never optimizer inputs.

New files (all new; nothing tracked was modified for this):

| Path | What it is |
|---|---|
| `scripts/officeqa/path_report.py` | Pure, stdlib-only checks: terminal-report parse (Section 2), reference resolution against the runtime-owned ledger + compute-input chronology (Section 3/4), support-judge request build + strict verdict parse, bounded UNKNOWN retries, offline candidate-reward preview, result-integrity checks (Section 8). No network/model imports. |
| `scripts/officeqa/path_report_pilot.py` | Isolated offline runner: `--self-check` (built-in fixtures, CPU-only) and `--episodes F --out DIR` (score runtime-owned capture records into an immutable run dir). Semantic judge is **disabled** unless `--judge-base-url` is passed (HTTP judge imported lazily). `--collect` (actor rollout) is deliberately **refused** — it is the GPU phase. |
| `scripts/officeqa/path_report_collect.py` | Isolated collector. Its brain — `EpisodeCapture` + `run_episode` — is a **pure, CPU-tested state machine**: runtime-issued episode-scoped observation IDs delivered to the actor as `[observation obs_N]` (the actor can only reference, never mint), delivery tracking (undelivered results have `delivered_text` nulled), parallel-batch semantics that make the impossible-chronology check fire, and terminal = the last assistant event with **no `<FINAL_ANSWER>` forcing / no synthesis** at the cap. `--self-check` runs it on CPU. A `--live` mode (thin wrapper reusing `eval_officeqa_agentic` primitives) plus an env-driven entry (`OQ_COLLECT_LIVE=1`, for the serve harness) are present; the symbol surface + message rendering are now **inspection-verified against eval's proven `_run_one` form** (identical assistant `tool_calls` with `arguments`-as-dict, `{role:tool,name,tool_call_id,content}` responses, `_assistant_turn` 3-tuple, `_load_parser → [(name,args)]`), so only the served model's runtime behaviour still pends the GPU smoke. |
| `scripts/officeqa/tests/test_path_report.py` | 61 CPU fixtures across parse/interface, deterministic authority (fake / cross-episode / undelivered / impossible-chronology refs), verdict robustness (NaN/inf/bool/stringy/duplicate-key/contradiction → UNKNOWN, never a silent positive), the candidate-reward wiring with **fixture** verdicts for semantic cases, bounded retries, and result integrity. |
| `scripts/officeqa/tests/test_path_report_collect.py` | 19 CPU fixtures for the collector: ID issuance/uniqueness, exact delivered-text, marker framing, generated/delivered tracking, the parallel-batch → impossible-chronology path, undelivered-on-actor-error, terminal-only (a JSON blob in a tool output is never the terminal), turn-cap-no-synthesis, retained failed exploration, tool faults, and end-to-end consumption by the scorer. |
| `scripts/officeqa/judge_validation.py` + `judge_validation_fixtures.py` + `tests/test_judge_validation.py` (9/9) | **Truth-by-construction** judge validator: WE author (question, answer, golden-path) triples grounded in REAL 1941 Treasury-Bulletin bytes, so labels are ground truth (no human reviewer). 12 fixtures (4 supported incl. valid alternative source + published aggregate; 7 unsupported incl. faithfulness/material/wrong-row/wrong-period/fabricated/incomplete/compute-mismatch; 1 deterministic fake-ref). Pass ⇔ `false_accepts==0 and false_rejects==0 and unknowns==0 and deterministic_ok`. |
| `air/83_officeqa_judge_validation.yaml` + `serve_judge.sh` `OQ_JUDGE_VALIDATION` branch | Runs `judge_validation.py` against a served GLM-5.3 TP16 (16×H100, rank-0 client). |
| `air/84_officeqa_collect_smoke.yaml` | **DONE + GREEN** (run 378377632425686, 13m, 8×H100). 2 easy-question GPU smoke of the live collector via `serve_and_eval.sh` (`EVAL_SCRIPT=officeqa/path_report_collect.py` + `OQ_COLLECT_LIVE=1`). Wrote `…/eval/path_report/collect_smoke_v1/captures.jsonl`; inspect-only — no scoring/judge/reward. See the smoke-result note below. |

Verified this pass (all CPU, no network/model/GPU):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/tests/test_path_report.py           # 61/61 passed
PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/tests/test_path_report_collect.py   # 19/19 passed
PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/tests/test_judge_validation.py      # 9/9 passed
PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/path_report_pilot.py --self-check    # SELF-CHECK PASSED
PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/path_report_collect.py --self-check  # SELF-CHECK PASSED
PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/judge_validation.py --self-check     # SELF-CHECK PASSED (oracle)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts:scripts/reward \
    python3 scripts/reward/tests/test_reward_contract.py                               # 43/43 (unchanged baseline)
```

**Collector → scorer cross-module check (CPU):** a `captures.jsonl` produced by the collector's
state machine feeds straight into `path_report_pilot.py --episodes … --out …` (no judge). A
fabricated observation id scores **zero deterministically** (no judge needed); a valid report
with references that resolve returns **UNKNOWN** with the judge disabled (fail-closed — a missing
judge never yields a positive). Schemas align field-for-field.

The capture-record schema (emitted by the collector, consumed by the scorer):
`{episode_id, question, question_requirements, terminal_text, termination,
answer_correct (separate; null until a reviewer labels it),
observations:[{observation_id, order, tool, args, outcome, delivered_text|null,
generated_by_request, delivered_to_request|null}], raw_executions, raw_turns}`. Observation
IDs are **runtime-issued** and looked up in that record — never recovered from model markers.

### Judge validation — GPU run DONE and GREEN (2026-09-14 UTC)

`air/83` run **289811683661869** (GLM-5.3 TP16, 16×H100, `reasoning_effort=high`,
`JUDGE_MAX_TOKENS=10240`) → **12/12 match, 0 false-accepts, 0 false-rejects, 0 UNKNOWNs,
deterministic_ok=true** (`…/eval/path_report/judge_validation_v2/summary.json`; confusion
`supported→supported 4`, `unsupported→unsupported 7`, `+1` deterministic fake-ref). First
real GLM-5.3 run had **zero parse UNKNOWNs** (the dialect fix holds — one attempt per case).

The **only** miss on the first attempt (run 748254700482351) was a report-**faithfulness**
false-accept: a path step claimed "Dec 1940 = 400" while the delivered bytes showed 375, yet
the answer 375 was independently supportable, so the judge scored it 0.8. Fixed by a CRITICAL
FAITHFULNESS RULE in `SUPPORT_JUDGE_SYSTEM` ("if ANY step materially misstates its cited
observation — a different value/row/period/units than the delivered bytes — the path is
UNSUPPORTED, even if the answer could still be justified"). The re-run flips it for exactly
that reason and does **not** over-reject the two hardest supported cases (valid alternative
source, published aggregate) — verified from the per-case rationales.

### Collector live smoke — GPU run DONE and GREEN (2026-09-14 UTC)

`air/84` run **378377632425686** (base Qwen3.5-35B-A3B, TP8, 8×H100, 13m) collected 2 easy
questions live. This closes the last unverified glue: the served actor emits parseable
qwen3_coder tool calls, and the full runtime path works on REAL data.

- **Both episodes**: 15 delivered observations each across 4 of 5 tools (search/read/grep/list),
  runtime observation IDs issued + delivered with clean sequential chronology (`gen@k → deliv@k+1`,
  nothing undelivered, empty greps retained). No `<FINAL_ANSWER>` forcing anywhere.
- **UID0002 → `terminal_report`**: the actor emitted a well-formed terminal JSON
  `{"answer":"507 million dollars","path":[{"id":"obs_7","observation":"obs_7","claim":"…VA FY1934 = 507 … excludes revolving funds …"}]}`;
  `parse_report` → valid, `check_references` → **valid** (cites a real delivered read). Interface works.
- **UID0006 → `turn_cap_tool_call`**: the model was still issuing a `compute` call on turn 16, so it
  hit the cap with no terminal JSON. Per contract the final call was **not dispatched and not
  synthesized** → `parse_report` = malformed ("no terminal JSON"). The no-synthesis guard fired.

Two REAL findings for the writeup (behaviour, not collector bugs): (a) **thin paths** — UID0002's
path is a single step despite 15 observations (the reporting prompt may under-encourage citing the
full read+grep chain); (b) **turn budget** — an easy question (UID0006) can exhaust 16 turns still
exploring and yield no report, so the ~20-question collection will have some no-report episodes.

### Human-review gate POSTPONED → automated + truth-by-design (2026-09-14 UTC)

Owner: independent human review is unavailable, so the pilot no longer depends on it. Decision =
**"Both, decoupled"**: **(A)** verifier soundness measured on a scaled truth-by-construction
battery (no human — we author the label from real bytes; the `air/83` method); **(B)** actor
usability on real rollouts labeled by machinery only. Contract updated accordingly
(`officeqa_path_report_pilot.md` §4/§5/§6/§7); the gate is postponed, not deleted.

Two DETERMINISTIC automations landed this pass (CPU, no-regret, tested), each removing a human
dependency and shrinking what the judge must be trusted for:

| Automation | Where | Effect |
|---|---|---|
| **answer-correctness from the answer key** | `path_report_pilot.py` (`--answer-key`, lazy `officeqa_reward.score_answer`) | label #1 auto-derived (`score_answer(gold, report.answer, tol)`); explicit record label still wins; abstain/malformed → None. Verified on the real smoke: UID0002 "507 million dollars" vs gold "507" → **True**. |
| **fabricated-value gate** | `path_report.py::check_claimed_values` | a LEAF step asserting a value-like figure absent from every delivered obs → deterministic ZERO **before** the judge. Calibrated on the 12 fixtures: catches `fabricated_value`+`wrong_value`+`wrong_value_material` (defense-in-depth on the judge's old weak spot) with **zero** false-rejects on the 4 supported + `wrong_row`/`wrong_period`/`missing_coverage`/`compute_mismatch`; conservative (year/small-int excluded, unit-scale tolerant), holds on the real smoke (UID0002 VALID). |

Tests: `test_path_report.py` **69/69** (+8 value-gate), `test_path_report_collect.py` **25/25**
(+6 `score_episode` seam), `test_judge_validation.py` 9/9; `path_report_pilot.py --self-check` now also asserts
`fabricated_value` is caught deterministically (judge skipped) and answer-key labeling both ways.

### Deliverable A battery — GPU run DONE and GREEN (2026-09-14 UTC)

`judge_validation_fixtures.py` scaled **12 → 34** (15 supported + 14 unsupported semantic + 5
deterministic), grounded in **three real bulletin blocks** — `FULL_TABLE` (1941-01 monthly),
`VA_TABLE` (1939-01 "Federal Expenditures - General", the real table UID0002 read in the smoke),
`CLAIMS_TABLE` (1995-03 claims-by-country, from UID0006). Covers all §5.4 families × observed
behaviours: correct lookup / compute / valid-alternative / valid-shortcut / **recovered-route** /
**padding-supported** / **sufficient-snippet** (accept); faithfulness (present-but-misstated),
wrong-column / wrong-year / wrong-country / wrong-period / **wrong-units** / **unsupported-comparison**
/ **compute-output-not-document** / **wrong-source** / incomplete-aggregate / compute-mismatch
(judge reject); and deterministic nonexistent / **undelivered** / **impossible-chronology** refs +
**value-fabrication** (caught pre-judge). The judge-validation harness now runs the value gate too;
absent-value cases are deterministic, present-but-misattributed cases are the judge's. Oracle
self-check clean (15→sup, 14→unsup, 5 det, 0 FA/FR/UNK); `test_judge_validation` 9/9.

`air/83` v3 run **528897888311555** (GLM-5.3 TP16, 16×H100, 35m) → **34/34 match, 0 false-accepts,
0 false-rejects, 0 UNKNOWNs, deterministic_ok=true** (`…/eval/path_report/judge_validation_v3/
summary.json`; confusion `supported→supported 15`, `unsupported→unsupported 14`, + 5 deterministic
caught pre-judge). Every must-reject case rejected **for the right reason** (per-case `issues` in
`results.jsonl`): the faithfulness cases (v1's weak spot) caught as e.g. "287 is the November 1940
value, December is 375"; `units_mismatch` caught the "millions of dollars" header; `compute_mismatch`
caught the hardcoded `print(2253)`; `undelivered_ref` independently flagged by BOTH the reference and
value gates (defense in depth). The CRITICAL FAITHFULNESS RULE generalized from 12→34 across three
real table families with **zero** over-rejection of the 15 supported cases. **Deliverable A is done.**

### Deliverable B — launch-ready, CPU-de-risked (2026-09-14 UTC), GPU awaiting owner go-ahead

The real-rollout usability path is fully wired and proven on CPU minus the served judge:

| Piece | Where | Status |
|---|---|---|
| offline scorer CLI | `path_report_pilot.py --episodes <captures.jsonl> --answer-key <csv> --judge-base-url … --out …` | already complete; deterministic gates + answer-key labeling + judge, immutable manifest (episode+code SHAs). **Dry-run on real files (judge off) green**: CSV key parsed, gates ran, integrity ok, outputs written. |
| scoring seam tests | `tests/test_path_report_collect.py` (+6, now **25/25**) | REAL collector record → `score_episode`: locks the contract invariants at the boundary — correct-answer+fake-ref → ZERO; correct-answer+fabricated-value → ZERO; wrong answer → candidate ZERO while support recorded separately; answer auto-labeled by uid. |
| serve-side dispatch | `serve_judge.sh` `OQ_SCORE_EPISODES` branch | parallel to `OQ_JUDGE_VALIDATION`; rank-0 client runs the scorer against localhost, exits non-zero only on an integrity violation (never on low usability). Syntax-checked; non-invasive (default off). |
| collection job (B1) | `air/85_officeqa_collect_deliverable_b.yaml` | base 35B actor, ~30 easy Q → `deliverable_b/collect_v1/captures.jsonl` (air/84 smoke scaled up; same model/TP/serve-len). |
| scoring job (B2) | `air/86_officeqa_score_deliverable_b.yaml` | GLM-5.3 TP16 serves + scores B1's captures against the answer key. Chain verified: B1 out path == B2 `OQ_EPISODES_FILE`; answer key == collector CSV (uids align). |

Full CPU suite green: `test_path_report` **69/69**, `test_path_report_collect` **25/25**,
`test_judge_validation` **9/9**, reward baseline 43/43.

### Deliverable B — GPU runs DONE (2026-09-14 UTC); pilot complete → owner review

Ran on owner go-ahead. `air/85` (run **599030212287940**, base 35B TP8) collected 30 easy-Q rollouts →
`deliverable_b/collect_v1/captures.jsonl`. `air/86` (run **1097882710253790**, GLM-5.3 TP16) scored them
→ `deliverable_b/scored_v1/`. Result: **10/30 valid structured reports** (13/30 hit the 16-turn budget
with no report; 4 malformed JSON; 3 abstentions). On the 10 valid: **0 bad refs, 0 fabricated values**;
**6 scored** (correct answer + supported, mean path-score 0.98), **0 judge UNKNOWNs**. Contract invariants
observed **live**: UID0011 = correct answer "42" but the judge found the path unsupported (never
establishes the "payroll savings" concept) → **zero** (the core anti-reward-hacking property, off-fixture);
labels stayed separate (wrong-answer episodes zeroed with support recorded independently; UID0066 shows
the judge keeping a *minor* citation slip supported while catching *material* misstatements).

**Headline usability gap:** ~⅔ of easy rollouts carry no candidate signal (turn-budget exhaustion +
malformed JSON) — an ACTOR/PROMPT issue, not a verifier-trust issue. Full write-up + recommendation:
**`docs/officeqa_path_report_pilot_results.md`**.

**Pilot complete. STOP for owner review — no RL/SFT.** Next actions (owner-gated, out of pilot scope):
address actor report-yield (turn budget / report checkpoint / format prompt) before any learning run.

### Report-yield fix — landed CPU-tested (2026-09-14 UTC), re-collection GPU-gated

Owner-directed cure for the yield gap (matches a known deep-research failure mode; the air/85 evidence
confirmed most turn-caps had the data in hand but never committed). Two changes in `path_report_collect.py`
(both param/env-gated; legacy no-tool-call path untouched by default):
1. **submit_report tool (Replace contract, `OQ_REQUIRE_SUBMIT=1`):** the actor finishes by CALLING
   submit_report with `{answer, path}` as args; a no-tool-call turn no longer finishes (it is nudged to
   submit). A submit call always terminates; the report comes from the tool args (scorer unchanged). The
   collector advertises the tool via `_render(tok, msgs, tools=ev.TOOLS+[SUBMIT_TOOL_SCHEMA])` (added an
   optional `tools=` param to `eval_officeqa_agentic._render`, backward-compatible).
2. **endgame window (`OQ_NUDGE_WINDOW=N`):** in the last N turns, inject an escalating nudge and LOCK
   search/grep (delivered a redirect, outcome `locked`); compute + submit stay available.
No-synthesis guarantee preserved: only an explicit submit produces a report; a turn cap → `no_submit_at_cap`
(no report). Tests: `test_path_report_collect.py` **32/32** (+7 submit/endgame). Full suite green
(69/32/9 + reward 43/12/18 = 183).

Re-collection (GPU, **owner go-ahead required**): `air/87` (submit + window=6, budget 24, serve 96k — model
max is 262k, so budget is bounded by KV memory, and collection is sequential so long context costs no
concurrency) → `air/88` (score v2). Turn budget is a knob (`EVAL_MAX_TURNS`); hard questions → ~48-64 + a
larger `EVAL_SERVE_LEN`. Compare `scored_v2` summary to `scored_v1` for the yield delta.

### Report-yield fix — GPU RESULT: cures work (2026-09-14 UTC)

`air/87` (run **1029076874889479**) + `air/88` (run **984696241724961**), same 30 easy Q as v1. **The two
cures clearly improved usability** (`deliverable_b/collect_v2/`, `scored_v2/`):

| metric | v1 (budget 16, text terminal) | v2 (budget 24, submit + window 6) |
|---|---|---|
| committed a terminal (valid or abstain) | 13/30 | **24/30** |
| valid reports reaching the judge | 10/30 | **17/30** |
| **scored (correct answer + supported)** | **6/30** | **10/30** (mean path-score 0.98) |
| no report at all | 13 | 6 (4 never-submit, 2 still-calling) |

Contract invariants held live again (wrong-answer valid reports → zero with support recorded separately;
UID0011, unsupported in v1, produced a properly-supported path in v2 with the larger budget → scored 0.9).

Three findings (none a correctness defect):
1. **Serialization bug (mine), fixed:** the tool-call parser returned the submit `path` argument as a JSON
   *string*; the collector stored it verbatim → every v2 report first parsed malformed. Fixed with
   `_report_text_from_submit` (decodes a stringified path/args; +2 regression tests, `test_path_report_collect`
   **34/34**). The existing captures were repaired losslessly offline (`captures_repaired.jsonl`, the exact
   decode the fixed collector now does) — no re-collection spent.
2. **Judge-context coverage gap:** 3/17 judged reports (17–20 observations) overflowed the judge's
   `JUDGE_MAX_MODEL_LEN=32768` → HTTP 400 → UNKNOWN (fail-closed, quarantined, never counted positive).
   Fix: raise the judge context (GLM-5.3 supports far more) and/or trim tool-history in the support request.
3. **Premature abstention (tuning signal):** abstentions rose 3→7, and **all 7 abstained questions have real
   gold answers** (5 were v1 turn-caps). Honest "gave up" beats a silent turn-cap, but they are failures on
   solvable questions — plausibly the window=6 lockout cutting retrieval too early. Tuning candidate: smaller
   window and/or larger budget, re-collect, see if abstentions convert to answers. Owner-gated.

### v3 (easy, bigger budget) + ALL-133-HARD capacity run + scoring — GPU DONE (2026-09-15 UTC)

Owner asked to (a) raise the judge seq len + turn budget and retry, then (b) run **all hard** questions and
confirm most commit an answer, flagging that in-episode **compaction** might be needed ("significantly
complicates things"). Decisions via AskUserQuestion: **parallelize + run all 133**, and **measure overflow
first** (build compaction only if the data warrants it).

**Easy v3** — `air/91` (collect, run **1068068600365379**) + `air/92` (score, run **1009031888706851**),
30 easy Q, budget 48, window 6, submit contract (`deliverable_b/collect_v3/`, `scored_v3/`):

| metric | v1 | v2 | **v3** |
|---|---|---|---|
| committed a terminal | 10/30 | 24/30 | **30/30** |
| valid → judge | 10 | 17 | **20** |
| scored (correct+supported) | 6 | 10 | **14** (mean **0.971**) |
| UNKNOWN (judge overflow) | — | 3 | **0** |
| overflow / malformed | — | 0 | **0 / 0** |

Non-commitment is **solved for easy** (30/30, 0 overflow). Contract held: 6 wrong-answer valid reports → zero
(5 kept report-fidelity), 1 correct-but-unsupported → zero. **10/30 abstain and all 10 have gold answers**
(premature) — the coverage ceiling is answer quality, not the harness.

**Hard (all 133)** — `air/93` (collect, run **1017890441708241**, 2h33m) + `air/94` (score, run
**642039525475275**), budget 64, window 8, `EVAL_SERVE_LEN=262144` (`deliverable_b/collect_hard/`,
`scored_hard/`). Collection coverage:

| outcome | n / 133 |
|---|---|
| committed a terminal report | **119 (89%)** — 88 valid, 29 abstention, 2 malformed-submit |
| context overflow (`actor_error`, HTTP 400) | **8 (6%)** — deep 49–63-turn non-submitters |
| no-submit-at-cap / turn-cap-tool-call | 5 / 1 |

Three-label scoring (0 UNKNOWN; judge robust at 128k): **40 scored** (mean **0.980**, graded 0.9–1.0),
**64 zero**, **29 abstention** (all 29 have gold → premature). The verifier's core promise, on real rollouts:
**4 correct answers were ZEROED for an unsupported path** (UID0018/0068/0177/0195 — right number, judge ruled
path unsupported), and the deterministic gates independently caught **2 fabrications** (UID0055 unresolved
refs, UID0121 fabricated value). This is contract (a)/(c) working live: right answer + bad path ⇒ no reward.

**Capacity verdict:** 262k + 64 turns is sufficient — 89% commit, obs up to 63 handled, judge request maxes at
~141k chars (~59k tokens). **Compaction is NOT needed** and was NOT built (measure-first paid off): overflow is
a 6% tail of episodes that *never chose to submit* — the same non-commitment as the abstentions, so compaction
would mostly convert overflow→no-submit, not overflow→answer. The real coverage lever is **premature
abstention** (easy 33%, hard 22%; 100% on answerable questions), a retrieval/anti-give-up problem, not context.

**Infra landed this session (CPU-tested; `test_path_report_collect` 37/37, `test_path_report` 70/70,
`test_judge_validation` 9/9, reward baseline unchanged):**
- **Parallel collector** — `path_report_collect.collect_live(concurrency=)` / env `OQ_COLLECT_CONCURRENCY`
  (default 1 = old sequential). Each episode is an independent `run_episode` (own loop+session) on a thread
  pool; input order preserved; one bad episode → `collector_error` record, never kills the batch. 133 hard at
  conc 16 ran in ~2.5h vs ~10–18h sequential. Also prints a termination histogram.
- **Parallel scorer** — `path_report_pilot.score_episodes(concurrency=)` / CLI `--score-concurrency` / env
  `OQ_SCORE_CONCURRENCY` (default 1). Judge HTTP is I/O-bound → thread pool, order preserved, per-episode
  progress to stderr. **Needed:** sequential scoring of 88 valid hard reports would exceed the job timeout AND
  write nothing (results are written once at the end); the first `air/94` was cancelled and relaunched at
  conc 8 for this reason.
- **Judge KV-fit lesson:** GLM-5.3 at TP16 / 0.90 util does **NOT** fit `max_model_len=262144` (needs
  22.6 GiB/GPU KV, has 15.6 → vLLM refuses to start; the first `air/92` FAILED that way, and CUDA-graph
  profiling eats most of any util bump). The judge prompt is bounded by `OQ_MAX_HISTORY_BYTES` and MEASURED at
  ≤141k chars (~59k tokens), so **`JUDGE_MAX_MODEL_LEN=131072` is right-sized** (>2× headroom, KV ~11.3 GiB,
  fits) and still fixes the v2 32k overflow. Size the judge window from the measured request, not from a round
  "256k" target.

**Stop condition:** pilot verifier-soundness (Deliverable A 34/34) + usability on real rollouts (Deliverable B,
easy + hard, machinery-only labels) are DONE. Open, owner-gated direction: **reduce premature abstention**
(retrieval quality / anti-give-up endgame nudge — fabrication gate + judge keep it honest) — NOT compaction,
NOT RL/SFT. Nothing committed.

### Anti-abstention + 3-phase funnel — easy A/B GPU RESULT: safe, modest recovery (2026-09-15 UTC)

Owner-directed fix for premature abstention: (1) anti-abstention wording ("commit your BEST SUPPORTED answer
when you have relevant figures; reserve DATA NOT AVAILABLE for genuinely-nothing-relevant"); (2) a 3-phase
funnel — explore (all tools) → last `OQ_NUDGE_WINDOW=20` forbid ALL retrieval (search/grep/read/list; compute+
submit stay) → last `OQ_SUBMIT_ONLY_WINDOW=5` only `submit_report`; (3) the **entire structure stated
prominently in the system prompt** with the real numbers filled in — the OOD guard, so the actor plans for
vanishing tools instead of being surprised. Budget raised 48→80. All CPU-tested (`test_path_report_collect`
38/38, `test_path_report` 70/70, `test_judge_validation` 9/9); reward baseline unchanged.

Easy-30 A/B — `air/95` (collect, run **530092572269017**) + `air/96` (score, run **583853565587709**), same
30 easy Q as v3 (`deliverable_b/collect_easy_v4/`, `scored_easy_v4/`):

| metric | v3 (pre-fix) | **v4 (anti-abstention + funnel)** |
|---|---|---|
| abstention | 10 | **4** |
| scored (correct+supported) | 14 (mean 0.971) | **17** (mean **0.976**) |
| zero | 6 | **9** |
| UNKNOWN (judge fail) | 0 | **0** |
| ref-invalid / value-fabricated | 0 / 0 | **0 / 1 (zeroed)** |
| answer_correct (key) T/F/none | 14/6/10 | **19/6/5** |

**Headline is SAFETY, not the count: 0 inflation** — verified across all 30, no score>0 episode violates
support, faithfulness, or answer-correctness. Pushing the actor to commit did **not** game the reward. All
three gates each fired independently on a *different* episode: **UID0023** answer `2.24` correct per key but the
value appears in no delivered obs → deterministic fabrication gate → **0** (before the judge); **UID0041** answer
`0.011` correct + value valid but **judge** ruled path unsupported → **0**; **UID0014** path supported but answer
`997.2` wrong → **answer gate** → 0. Three separate labels, three separate zeros — the decoupling is real and
holds under answer-pressure.

**Honest coverage read:** abstention 10→4 (−6) is a real, direct prompt effect (model commits more). But of the
6 recovered commitments **only 1 (UID0038) was genuinely correct+supported**; the other 5 were correctly zeroed
(3 wrong, 1 fabricated-but-correct, 1 unsupported-but-correct, 1 malformed). Aggregate `scored 14→17` = +1
recovery plus net +2 churn among already-committing episodes (4 zero→scored, 2 scored→zero) — a single-run,
prompt-changed re-roll, so per the multi-seed rule that +2 is **directional, not established**. The funnel is
**inert on easy** (episodes commit before the last-20/last-5 windows). One malformed submit + one `actor_error`
appeared (budget-80 tradeoff: don't-give-up → longer trajectories), 1/30, acceptable.

**Verdict:** the anti-abstention push is **safe to apply** (the gates absorb the extra wrong/unsupported
commits) and gives a modest but real coverage gain on easy. The full funnel+budget gets its real test on
**hard** (v3 hard: 29/29 abstentions had gold, so recovery headroom is large, and the funnel actually engages)
— that hard collect+score is the natural next GPU job, **owner-gated**. Nothing committed.

### Phase-1 GRPO training smoke — GPU RESULT: pipeline works end-to-end + reward flows (2026-09-15 UTC)

The pivot from offline verifier to real GRPO training. `air/99` prepped the EASY split (102 train / 11 val,
`agent_name=path_report_agent`, funnel-filled system prompt, `ground_truth=gold`). `air/100` ran the first
end-to-end training smoke: 32 GPU / 4 nodes (2 train + 2 GLM-5.3 judge), the custom `path_report_agent` loop
(submit-terminate + 3-phase funnel + observation ledger), `rate_limited` reward manager calling the in-loop judge
via `officeqa_path_report_reward.compute_score`, `norm_adv_by_std_in_grpo=False`.

Two batch-assembly bugs surfaced (both in-image-only, now fixed + CPU-tested in `tests/test_path_report_agent_loop.py`):
1. **`DataProto.concat` non-uniform schema** (`Key '_pr_obs' is not present…`): finalization ran only on submit, so
   turn-capped samples kept scratch keys and lacked `path_report_record`. Fix: a `run()` override + pure
   `finalize_extra_fields` finalize EVERY termination path to one uniform record, dropping all `_pr_*` scratch.
2. **`detach_utils.py:153 abs(None-None)`**: pre-seeding `_pr_*` into `agent_data.extra_fields` before `super()` at
   turn 0 skipped the base ToolAgentLoop bulk-copy that carries `min/max_global_steps` (the fully-async staleness
   stamps) → None. Fix: stash episode identity AFTER `super()` (funnel gating, which only sets `_active_tools`, stays
   before). Gotcha: a custom `_handle_generating_state` must not write `extra_fields` before `super()` on turn 0.

Note: `run_grpo_fully_async.sh`'s exit-guard reported bug 2 as job "SUCCESS" (it greps for a benign teardown phrase);
a green air status is NOT proof on this stack — always grep logs for `Error executing job|Component failed|Traceback|NoneType`.

**Result (run 318555382588046, clean completion, 1h36m, 0 failures — genuine success, verified past the guard):**
8 trainer steps (global_step 1→15), rollouter hit `total_rollout_steps=256` → clean queue-termination teardown.
`critic/score` mean/max per step: steps 1-3 = 0/0, step4 = 0.0074/**0.95**, step5 = 0/0, step6 = 0.0188/**1.0**,
steps 7-8 = 0/0. A 0.95–1.0 score is a SCORED graded reward — reachable ONLY if the judge was called in-loop and
returned supported + the answer was correct + the path faithful (a broken judge path floors at 0.0). `critic/advantages`
non-degenerate on reward-bearing steps (max 0.71 / 0.40, min < 0) → the `norm_adv=False` graded signal is real, not
collapsed. `actor/pg_loss`/`grad_norm` non-zero → policy updating. `num_turns` mean ~29 (=deep multi-turn, 16-assistant
cap), `aborted_ratio 0.0`, `response_length/max ~9.7k` << 47k budget.

**Verdict:** the Phase-1 machinery is **proven** — custom loop + funnel + submit-terminate → batch assembly → in-loop
judge → graded reward → GRPO advantages → weight sync all fire, and the trajectories are usable + in-distribution for
GRPO. Reward is **sparse** (2 of 8 steps earned it; rare per 16-sample batch on easy) — enough to prove the loop, NOT
yet a sustained learning curve. Next (owner-gated): a longer/bigger Phase-1 run for a real reward curve, and/or Phase-2
synthetic hard-question generation. Nothing committed.
