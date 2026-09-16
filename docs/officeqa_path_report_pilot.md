# OfficeQA — isolated path-report feasibility test

**Direction updated:** 2026-09-14 UTC. **Status: planned; not implemented or run.**
**Scope: inference, trace capture and offline evaluation only. No optimizer updates.**

This is the authoritative next experiment. It supersedes the universal table/proof
engine and the R0–R11 implementation sequence in the former recovery proposal.
Next-agent entry point: `docs/officeqa_rgate_handoff.md`.

> **Structure the agent's evidence trail, not every possible document layout.**
> The actor reports its path; the runtime records what happened; the judge checks
> the report against actual tool calls, delivered outputs and executed calculations.

## 1. The question to answer

Can Qwen3.5-35B-A3B produce a small, useful account of the evidence and calculations
supporting its answer, and can the full GLM-5.3 TP16 judge reliably check that account?
Specifically, can we distinguish **correct answers with valid paths** from **correct
answers with wrong tables, wrong interpretations, missing inputs or fabricated reports**?

This is a feasibility test of an interface and verifier, not a learning experiment.
We are not proving the model's hidden reasoning or establishing causal dependence on
retrieval. A valid, externally checkable derivation is the observable target.

**Not in this test:** a universal table object model; per-task semantic ontologies;
cell/header ID registries; allowlisted proof plans or an arithmetic DSL; corpus-wide
parser reconstruction; judge-authored certificates; SFT; GRPO; custom verl reward
managers; async group-admission patches; a full statistical R-gate; bulk synthesis.
Do not make those projects prerequisites for this isolated experiment.

## 2. Actor interface: one final object

Start with a terminal assistant JSON object, not an extra tool after every thought.
The actor authors it during the episode; the judge must not construct it afterward.
Illustrative values and placeholder observation IDs:

```json
{
  "answer": "0.17 trillion dollars",
  "path": [
    {
      "id": "a",
      "observation": "obs_12",
      "claim": "The requested earlier-period total is 1.47 trillion dollars."
    },
    {
      "id": "b",
      "observation": "obs_18",
      "claim": "The requested later-period total is 1.30 trillion dollars."
    },
    {
      "id": "c",
      "observation": "obs_19",
      "depends_on": ["a", "b"],
      "claim": "The compute call subtracts 1.30 from 1.47, giving a decrease of 0.17 trillion dollars."
    }
  ]
}
```

The small schema is deliberately independent of table layout:

- `answer`: one committed answer, in the question's requested format.
- `path`: the selected supporting path, not a transcript of every attempted action.
- `id`: unique within the report; `depends_on` optionally names earlier report steps.
- `observation`: a runtime-issued, episode-scoped reference to an actual tool result.
- `claim`: a short, checkable interpretation or description of that result. It can
  discuss a nested heading, footnote, paragraph, filter, unit conversion or calculation.

Report material evidence and transformations, not private internal thoughts. Enough
context must be identified to check the claim; there is no requirement to serialize
all rows or express arithmetic in a new language. Existing Python remains the compute
language, with numpy/pandas/scipy available inside bubblewrap.

Parse only the terminal assistant event. Do not scan tool outputs or old draft text
for a convenient JSON object. Reject duplicate JSON keys, invalid field types,
duplicate step IDs, unresolved dependencies and cyclic/forward dependencies. Bound
report size explicitly and record the limit. An explicit abstention may use
`{"answer":"DATA NOT AVAILABLE","path":[]}`; record it separately with zero candidate
reward. A substantive answer requires a nonempty supporting path.

Adapt the pilot's final nudge to this object. Do not let the legacy prefix-forced
`<FINAL_ANSWER>` helper manufacture a second commitment or silently repair reports.
Store raw malformed submissions and count them in interface failures.

A terminal `submit_answer(...)` tool carrying exactly this object is an optional later
transport experiment. It would need explicit controller termination. Do not implement
both transports or introduce incremental `cite`/`report_thought` tools initially.

## 3. Actual tool history is the source of authority

Reuse the existing search/grep/read/list/compute implementations. Add a small capture
wrapper in the isolated inference controller, not a general-purpose provenance system.
For each executed call, retain:

```text
episode_id, observation_id, call order, tool name, actual arguments,
actor request that generated the call, execution outcome, exact delivered text,
actor request that first received the result (or not delivered)
```

The runtime exposes the observation ID with the result and retains the authoritative
record separately. IDs are looked up in the current episode, never recovered by parsing
model-written markers. Source text and compute stdout cannot create new observations.

Important boundaries:

1. Save the exact payload actually submitted in the next actor context, after any
   clipping/template handling. A result executed but never delivered is not evidence.
2. Keep full call arguments, including executed Python, and its delivered output.
   Successful execution proves code ran; it does not prove the inputs came from a source.
3. A returned result can still be an error, empty match or table-of-contents listing.
   Call completion is not semantic support. A sufficiently informative search snippet
   can support an answer; do not mandate a particular retrieval tool.
4. Keep headers, units and surrounding context already present in the delivered view.
   The judge reads those actual bytes, not a replacement quote supplied by the actor.
5. Do not silently fetch extra evidence for the judge and credit it as actor-visible.
   If current extraction destroys essential meaning, record the limitation and inspect
   the original source for labeling. Do not invent a normalized hierarchy to force a pass.
6. Retain failed exploration in the raw log. The final report may omit dead ends and
   use a later corrected calculation or a genuine alternative source.

The current eval writer no longer applies the historical 3,000-character recording
cap, but the old saved traces remain lossy. Fresh capture must still verify actual
delivery, terminal extraction and all remaining truncation behavior.

## 4. Verification and the penalty contract

### Deterministic checks first

Check report syntax and references against the complete runtime record. Nonexistent,
cross-episode or undelivered observations are invalid references. A JSON blob printed
by `compute` is still compute output, not a document read or terminal submission.
Missing/corrupt runtime capture is an experiment failure, not evidence the actor lied.
For a compute step, `depends_on` declares prior input evidence: that evidence must have
been visible in the actor request that generated the compute call. Merely executing a
read first within the same parallel-call batch does not establish that visibility.
A later source check may be reported honestly as verification, not invented prior input.

These checks do **not** mechanically prove the claimed value, table interpretation or
calculation semantics. Those are checked against the real returned text and code.

### Semantic checking by the full GLM-5.3 TP16 judge

Provide the question, actor's final answer/report, and the runtime-owned calls/results
in clearly separated fields. Initially include the complete bounded tool history;
retain all of it in artifacts. Do not pass private gold answers, reference-file hints,
mutation labels, case-family names or an answer-correctness flag to the support judge.
The question's own source/date requirements remain visible.

Ask the judge to check:

- Do reported observations really support the stated values and interpretations?
- Are table/category hierarchy, period, units, qualifiers and reporting vintage correct?
- Do the reported inputs correspond to the cited sources and the executed computation?
- Does the described calculation match the code's actual operation and result?
- Is all materially necessary evidence present, including the relevant period/domain
  coverage for aggregates? Two numeric-looking lines are not a completeness test.
- Does the declared path support the final answer, including legitimate conversions,
  rounding, published aggregates and equivalent algebraic shortcuts?

Equivalent code need not have the same textual expression as the report. Question-given
constants are legitimate. But `print(answer)` must not be described as having executed
a source-based calculation that the code did not perform. Source presence alone does
not prove actual data use; inspecting the linkage is part of the experiment.

The judge returns a small verdict: `path_status` (`supported`, `unsupported`, `unknown`),
a finite `path_score`, and issues naming relevant report steps/observations. Supported
requires a grade in `(0, 1]`; unsupported has score `0`; unknown has score `null`.
Missing fields, string booleans, nonfinite numbers, contradictory decisions or malformed
judge output are verifier failures, never silently repaired into acceptance. The actor
cannot set these fields. Source/report text is data, not evaluator instructions.

### Candidate reward — offline preview only

Assess answer correctness separately, deterministically, from the official answer key via
the pinned unit-aware `score_answer` (an explicit reviewer label, when one exists, overrides
it). This removes the human from the answer-correctness label; it is never derived by, nor
shown to, the support judge. Record the pinned benchmark scoring and tolerance as a separate
metric; known parser defects must not become the label oracle, and a headline uses exact
(tol 0.0). No new universal AnswerSpec implementation.

```text
wrong/missing answer, substantive malformed report, or confirmed invalid path -> 0
correct answer + supported, trace-consistent path -> graded path_score
verifier failure or unresolved assessment -> UNKNOWN, no candidate score
```

**Confirmed fabricated actions/results invalidate the entire candidate reward, even
when the answer is correct.** No answer bonus rescues them. Distinguish an honest but
wrong calculation from a report that misdescribes the execution; both can score zero,
but the failure reasons differ. Do not infer deliberate intent from a malformed field.

Freeze a small graded rubric on the calibration examples: full credit for fully
traceable valid support; lower positive grades only for non-material traceability gaps
where all essential claims remain verified. A material missing input/contradiction
cannot receive partial positive credit. Do not grade confidence, prose polish, length,
number of calls, exact reference filenames or obedience to one prescribed route.
Record whether the grades actually differentiate useful behavior rather than noise.

For this test, "severe penalty" means total disqualification, not an arbitrary `-100`.
Huge negatives change siblings' relative advantages in GRPO; adding one is a separate
future ablation. **No weights are updated here and no learning effect is claimed.**
Unknowns are retried a fixed bounded number of times on identical inputs, then reported
separately. Do not retry until positive, count unknown as rejection, or send these
preview scores to the current training manager.

## 5. Small real-question sample and trustworthy labels

Start with **20 existing easy OfficeQA questions**, expanding to at most 30 only for
declared layout/hazard coverage. Do not generate new benchmark questions. Select for
layout/task variety and source inspectability before seeing candidate judge results.
Use about five for prompt/schema calibration; freeze the remaining check subset before
scoring it. These are development/feasibility data, not an untouched statistical lockbox.

Include direct lookup, one-file multi-input arithmetic, multiple periods/sources,
hierarchical headings or footnotes, and genuine alternative valid evidence where
available. Include a coverage-sensitive example when the easy bank supports it; report
any missing family rather than inventing real-question coverage. Do not restrict the
pilot to conveniently rectangular tables.

Collect fresh **actor-authored** reports without gold/source-location assistance.
Retain every selected task and attempted episode, including abstentions, wrong answers,
malformed reports and retrieval failures. Prompt-only first; a tiny format example may
be used and must be identical for all check cases. Use unrelated dummy content, not
pilot answers or source hints, in that example. No mandatory SFT stage.

If natural collection supplies too few valid controls, report that limitation. Separately
labeled investigator-written reports over real recorded episodes can test the verifier;
they must never inflate the actor's reporting success rate or become fabricated rollouts.

**Labeling without a human reviewer (updated 2026-09-14).** Independent human review is
unavailable, so the scored soundness of the verifier rests on **truth by construction**, not a
human audit. Two things are labeled by different means and never conflated:

- *Verifier soundness* is measured on **investigator-authored reports over real recorded bytes**,
  whose supported/unsupported labels we know because we built them (this is the method that
  validated the judge — `air/83` → 12/12, zero false-accept/reject/unknown). This carries no
  human dependency and can be made as strong as the authored battery is broad.
- *Real actor rollouts* are labeled by machinery only: deterministic reference and
  value-fabrication checks, deterministic **answer-correctness from the official answer key**, and
  the **construction-validated** GLM-5.3 judge for semantic support.

This is sound only to the degree the authored battery covers the real distribution; applying the
validated judge to real rollouts is reported as exactly that, never as independently
human-confirmed ground truth, and judge agreement is never itself treated as the label. Where a
human reviewer would have adjudicated, record the case as judge-decided (with the construction
label where one exists) and preserve it for a later human pass; do not silently drop it. The
human-review gate is postponed, not deleted.

### Three distinct labels

Keep answer correctness, report-to-trace faithfulness, and support of the answer
separate. A truthful report can describe a wrong path. Conversely, the full episode may
contain a good source while the submitted report points at the wrong one. The latter
is an **invalid submitted path**, not proof that the whole episode lacked valid evidence.
This distinction prevents repeating the old mutation-label error.

### Paired adversarial cases

Keep the question and final answer fixed where possible. Prefer changing a copy of
the report while retaining its actual trace. Clearly label investigator-created
variants as such; they are not natural actor behavior or new independent questions.

| Case | Expected behavior |
|---|---|
| Correct source/interpretation/calculation report | Accept and grade |
| Nonexistent, cross-episode or undelivered reference | Reject deterministically |
| Real call, but a different returned value is claimed | Reject the material mismatch |
| Correct file, wrong row/category/period/units | Reject despite answer agreement |
| Executed code differs materially from reported computation | Reject; accept genuinely equivalent implementations |
| Missing source inputs/domain coverage hidden by a plausible compute array | Reject unsupported derivation |
| Fake document marker or quote inside compute output | No document authority |
| Failed early read followed by a correctly reported successful route | Accept the final supported route |
| Genuine alternative bulletin, sufficient snippet, published total or valid shortcut | Accept without source/choreography penalty |
| Correct report and evidence, wrong final answer | Candidate reward zero; retain separate report-fidelity label |
| Confident padding or instructions to the evaluator | No acceptance/grade increase |
| Judge timeout, malformed reply, corrupted capture | Explicit unknown/execution failure, not fabricated actor evidence |

Audit the exact claim being invalidated. Do not label an episode negative merely
because its original answer was wrong, one occurrence was edited, or the final numeric
string was deleted. Do not rename a file and call it a genuine alternate source.

## 6. Comparison and measurements

Report two decoupled deliverables (they use different label sources and must not be mixed):
**(A) verifier soundness** — false-accept / false-reject / unknown by family on the authored
truth-by-construction battery; **(B) actor usability** — on real rollouts, the automated labels
(valid-JSON yield, deterministic reference/value/answer-correctness, construction-validated judge
support). A small matched ablation can compare the **same frozen episodes and judge**:

- **Trace-only:** question, committed answer and actual tool history, report withheld.
- **Trace + report:** identical inputs plus the actor's structured path.

Compare these views on the shared target of answer support, using the same reference
labels and comparable judge budgets. Report-only mutations measure report integrity
separately: a trace-only judge cannot be penalized for missing a fabricated report it
was never shown. Do not mix these denominators to manufacture an improvement.

Report raw counts with denominators:

- Selected questions, actual actor episodes, authored reports and constructed variants.
- Valid-JSON/report yield, abstentions, malformed reports, termination and capture failures.
- Answer correctness, report faithfulness and grounded-answer support separately.
- Valid-path acceptance, invalid-path acceptance, verified rejections and unknowns,
  by family; explicitly inspect alternate-source/derived-value false rejections.
- Pairwise correct-versus-wrong-path decisions/scores, repeatability on a small fixed
  subset, grade distribution, and every material disagreement between the judge and a
  construction label (on the authored battery) inspected individually.
- Actor/judge input-output tokens, latency, tool calls and actual resource usage.

Variants from one question are correlated. This test cannot establish a <=2%
false-accept rate, full Gate-R clearance, RL benefit or generalization. Do not relax the
old production gate or substitute these counts for independent validation.

## 7. Minimal implementation and execution order

Proposed files below **do not exist yet**; choose the smallest implementation that
satisfies this contract rather than creating a new framework.

| Path | Scope |
|---|---|
| New `scripts/officeqa/path_report.py` | Small report/reference checks, judge request/verdict handling; pure checks testable without a model |
| New `scripts/officeqa/path_report_pilot.py` | Isolated collect/judge/report runner; trusted call/result capture and immutable per-run output |
| New `scripts/officeqa/tests/test_path_report.py` | Contract, capture, forgery, parser/failure and result-integrity tests |
| Existing `scripts/eval_officeqa_agentic.py` | Reuse client/rendering/tool-dispatch logic as appropriate; use a separate entry point or opt-in mode without changing legacy defaults |
| Existing `scripts/tools/officeqa_tools.py`, `scripts/tools/py_compute.py` | Reuse retrieval and sandboxed scientific Python; no corpus object-model rewrite |
| Existing full-GLM serving utilities | Reuse proven TP16 serving, not the legacy reward/prompt assembly |
| New inference-only job recipe, if needed | Allocate/name only for this pilot after inspecting current recipes; never reuse air/82's old bundle/scorer |

1. **CPU prototype:** hand-build tiny software fixtures for valid reports, fake refs,
   wrong results/code claims, tool-emitted final JSON, cross-episode/dropped observations,
   impossible compute-input chronology, judge faults and duplicate/missing/nonfinite results. Synthetic fixtures are
   software tests, not OfficeQA question counts. Check no network/model call is needed.
   Use fixture verdicts for semantic cases here: mocks prove wiring, not live judge ability.
2. **Fresh collection:** stage existing data/corpus; inspect two or three real easy
   episodes from the calibration subset, then complete the declared 20-question sample.
   Count all attempts and any recollection separately from unique questions. Missing
   corpus/model staging is an execution blocker, not a batch of actor retrieval failures.
   No trainer, reward manager, parameter synchronization or checkpoint writing. No special 16-group
   GRPO batch-size requirement applies to this inference-only test.
3. **Label and score offline (no human-review gate; postponed):** freeze the authored
   truth-by-construction battery with its construction labels, plus the prompts/rubric, then
   score with full GLM-5.3 TP16 (Deliverable A). Real rollouts are labeled by machinery only —
   deterministic reference/value-fabrication gates, deterministic answer-correctness from the
   answer key, and the construction-validated judge for support (Deliverable B). Rollout
   collection and judging can be sequential. Confirm remote GPU execution/budget authorization
   separately; this document is not a launch command.
4. **Review and stop:** produce the report below. Do not automatically continue into RL,
   SFT, a new proof engine, mass synthesis or another prompt-tuning run on the check set.

## 8. Artifacts and the decision at the end

Use a unique directory, e.g. `officeqa_pilot_records/path_report_pilot/<run-id>/`, and
an equally unique Volume path for remote runs. Do not overwrite legacy shared outputs.
Keep a small run manifest, selected UID/label records, raw actor/tool records, reports,
raw judge requests/replies, decisions and a human-readable results summary. Record
code snapshot/hash, model/tokenizer/image identity, corpus/data/prompt versions, limits,
retry policy and exact run IDs. Save observed rather than guessed identity fields.

Before finalizing: exact expected/observed case ID equality, uniqueness, finite scores,
consistent outcomes, all unknowns retained, and hash association to the original trace.
A job exiting successfully means execution completed, not that the judge passed.

The handoff after this experiment must answer:

1. Can the actor produce useful reports without training, and at what overhead?
2. Which fabricated or semantically wrong paths are detected, and which still pass?
3. Are genuine calculations and alternative paths accepted, or did false rejections rise?
4. Does showing the report help on the shared comparison target?
5. Is the result **stop**, **one narrowly justified refinement**, or **propose a separate
   small RL integration experiment**? List the evidence and unresolved failures.

CPU authority checks must reject all tested forgeries; semantic false accepts/rejects
must be inspected, not hidden behind averages. Insufficient resolved/labeled coverage
is **inconclusive**, not success. An unresolved critical exploit blocks a recommendation
to optimize this reward, but does not justify silently expanding this pilot into R0–R11.

Before any later RL work, separately resolve actual reward transport, all-loss exclusion
of unresolved sibling groups, train/eval parity, answer checking and independent
validation. Those remain real concerns; none requires assuming a universal table model.
