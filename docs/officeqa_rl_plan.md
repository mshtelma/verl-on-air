# OfficeQA — path-report pilot and conditional RL roadmap

**Direction updated:** 2026-09-14 UTC. Owner: michael.shtelma.

> **Next: one isolated, inference-only path-report experiment.**
> Train no policy and launch no larger RL run as part of it.
> Active experiment: `docs/officeqa_path_report_pilot.md`.
> Takeover: `docs/officeqa_rgate_handoff.md`.

The owner rejected the previous universal table/proof-engine direction as too heavy
for heterogeneous documents. This plan replaces it, rather than adding another
prerequisite to it. The old R0–R11 sequence is no longer the immediate backlog.

## 1. Goal and chosen direction

The eventual goal remains improving Qwen3.5-35B-A3B's grounded document answering,
with a useful additional ability: **reporting the evidence and calculations supporting
an answer in a small structured form**.

The next test asks whether that report is practical to generate and useful to verify.
It does not test learning. Structure the references and claimed dependencies, not
all possible table layouts, header hierarchies, financial concepts or operators.

```text
Existing real easy questions + current retrieval/scientific-compute tools
                              |
                              v
Actor produces answer + concise structured path
Runtime separately records actual calls, delivered outputs and executed code
                              |
                              v
Simple reference checks + full GLM-5.3 TP16 semantic audit
                              |
                              v
Small independently reviewed comparison / fabricated-report tests
                              |
                              v
STOP and review: abandon, narrowly refine, or propose a separate RL experiment
```

The actor authors the report from the first experiment. Do not have the judge invent
a certificate and claim the policy learned reporting. The judge checks the declared
path against actual evidence. A valid report is not proof of hidden internal causality.

### Scope boundaries

- **Keep:** real easy questions, full GLM-5.3 TP16, bubblewrap, numpy/pandas/scipy,
  actual tool records, one terminal commitment, graded valid reward, zero for invalid
  answers/paths, and distinct verifier-failure outcomes.
- **Remove from the immediate plan:** universal table IR, per-task ontologies, typed
  cell/slot registries, a proof-plan DSL/operator engine, judge-proposed-first
  certificates, and a mandatory multi-package training-framework rewrite.
- **Do not replace:** source verification with the model's self-report, or the old
  failed validation gate with a small feasibility result.

## 2. Current state — evidence, not readiness claims

The new pilot is **not implemented and has no result**. This documentation rewrite
changes no production code, runtime YAML values, dataset or model weights.

| Component | What is actually established |
|---|---|
| Retrieval and eval infrastructure | Existing search/grep/read/list/compute and agentic evaluation have been exercised; reuse them with isolated capture/report handling |
| Scientific compute | Bubblewrap boundary and scientific Python implementation exist; historical sandbox run `510519807483794` exercised the boundary. Preserve it |
| Full judge serving | GLM-5.3 TP16 was exercised in the historical judge jobs; keep the full model |
| Answer-mode training | Run `616024033111491` completed two optimizer steps. Not evidence of learning, grounded-reward safety or actual UNKNOWN-group exclusion |
| Current grounded audit | Unsafe historical v2.1 behavior remains. No new reward is approved for optimization |
| Promotion-guard work | Protocol/helper definitions exist in `officeqa_grounded_reward.py`, but the inspected `compute_score` and launchers do not call them. Do not claim an enforced guard or completed R1 |
| Fresh recording | Current eval source saves full tool-return strings rather than the old 3k recording cap; actual delivery/terminal/truncation checks are still needed for the pilot |
| Path-report prototype | Not implemented; no collection, case selection, live judging or RL run has occurred |

Old source comments naming `officeqa_proof_v1`, R1–R11 or earlier section numbers
refer to the superseded design. They are not instructions to implement it. Preserve
pre-existing partial work; the isolated runner must not use the legacy training scorer.

### Historical completed runs

The last archived Jobs API check was **2026-09-14T10:58:22Z**. This documentation
update did not poll the workspace again. All five reviewed jobs were terminal:

| Run | Historical outcome |
|---|---|
| `878119239646411` | air/82 attempt 1 failed on scorer import |
| `657777974855885` | attempt 2: 317 scored; 78/80 labeled controls accepted, 121/220 labeled negatives accepted; gate failed |
| `591895497446045` | attempt 3: 317 scored; 63/80 controls, 72/220 negatives; gate failed |
| `543029753940588` | attempt 4: 339 scored; **16/80 controls, 5/242 negatives**; gate failed |
| `616024033111491` | answer-mode smoke: two real optimizer steps; successful integration smoke only |

Attempt 4's reported case upper95 was **4.295%**, base-ID upper **7.414%**, UNKNOWN
11/339 and wrong-answer nonzero 0/17. **55 of its 62 scored control rejections came
from literal gold-in-quote matching.** The labels were also uncertified and sometimes
incorrect, so these are not independently established semantic error rates.

Source: `officeqa_pilot_records/rgate_review_2026_09_14/README.md` and its exact-hash
reports/scores. Keep that sealed archive unchanged. Its old implementation recommendations
are historical; this document and the new pilot plan govern next actions.

## 3. Immediate experiment

The complete contract and minimal file map are in `docs/officeqa_path_report_pilot.md`.
Do not duplicate or evolve a second schema in this roadmap.

1. Build a small offline controller/report checker and CPU fixtures.
2. Select about 20 real easy questions, with at most 30 for declared coverage; collect
   fresh actor-written reports and exact actual-tool records. Keep natural failures.
3. Independently check labels, create a small set of clearly marked report variants,
   then audit with the full GLM-5.3 TP16 judge.
4. Compare supported and unsupported reported paths; optionally compare trace-only
   versus trace-plus-report on the shared answer-support target.
5. Publish results and **stop for a decision**. No automatic RL/SFT continuation.

No special 16-group async batching, training parquet, parameter sync or checkpoint
requirement belongs in this inference-only test. The new pilot is not another air/82
run and must not consume the old mutation labels as certified truth.

### Penalties and grading

A materially fabricated action/result or unsupported path disqualifies the entire
candidate reward, even when the number is correct. Valid paths retain graded scores.
Wrong answers receive zero; a verifier outage or unresolved judgment is UNKNOWN, not
an ordinary negative. These are **offline reward previews**, not optimizer inputs.

Do not add huge negative constants. Mean-relative GRPO can give positive advantages
to other invalid siblings when one sibling receives a very large negative reward.
Keep standard-deviation normalization OFF if a graded treatment is later approved;
inspect actual group advantages before changing the penalty scale.

The report need not list every dead end. Earlier mistakes followed by a correctly
reported valid solution, genuine alternate sources and equivalent computations should
not be penalized. Report-faithfulness and answer-support labels are separate.

## 4. Lessons to carry into the small test

- Actual tool calls **and returned content/code** are the authority—not filename
  mentions, flat printable markers or actor-authored compute stdout as source data.
- Do not require a derived answer to appear literally in a quote, count files as
  operands, or infer requested periods from arbitrary year strings.
- A wrong original answer, deleted final string or swapped early row does not prove
  the remaining episode has no valid support. Audit the particular report claim.
- Search snippets may suffice; published aggregates and later corrected routes may
  be legitimate. Do not mandate the reference traversal.
- The original 246 traces have historical recording loss; changing today's writer
  cannot reconstruct what was lost. Use them for diagnosis, not fresh capture claims.
- Missing/duplicate/NaN results and unknown judgments must not create a false-green
  report. Save raw requests/replies and exact case/episode associations.
- Heterogeneous source layouts still need semantic judgment. If necessary context is
  destroyed or ambiguous, report that limitation rather than declaring it solved.

The review's five residual accepted negatives remain useful examples, not automatically
reusable labels: surviving later evidence (UID0002), formatting versus retrieval
(UID0148), derived reported values (UID0204), period coverage (UID0189), and shortcut
versus category hierarchy (UID0122). See the archive for their full qualifications.

## 5. Conditional future work — not the next-agent assignment

Only if the isolated test is useful should we propose a separate small RL integration
experiment. It would need evidence for these requirements, not the old object model:

- Trusted report/tool-history transport through the **actual** training loop, with
  correct terminal extraction, observation-token masks and offline/train parity.
- Correct answer handling for the chosen tasks, including units/scale/precision.
- Verifier UNKNOWN preserved through every failure layer, with unresolved whole
  sibling groups excluded before all losses. Advantage zeroing alone is insufficient.
- Bounded async refill/backpressure and explicit completion; no silent shrinkage,
  deadlock or false-green run after an optimizer failure.
- Independently checked reward behavior, adequate validation scope and an explicit
  owner decision about what limited training experiment is authorized.
- A matched answer-only versus report-aware graded-reward comparison under the same
  controller, data, initialization and budgets. If SFT is used, account for it separately.
- Independent grounded-correct evaluation, not just improvement in the training judge's
  score; checkpoint save/reload evidence whenever that capability is claimed.

These are deferred integration/measurement requirements. Do not implement a universal
schema or proof interpreter as an assumed fallback. If the small semantic audit fails,
inspect the concrete failure and decide whether a narrow improvement is justified.
Large RL and bulk synthesis remain **NO-GO**.

## 6. Data, evaluation and operational constraints

Use the existing easy partition, not new synthetic OfficeQA questions or the hard
benchmark as pilot training data. The current bank has 113 easy questions; repeated
variants do not create new independent questions. Hard-133 is training-disjoint but
has already been exposed during evaluator development; it is not an untouched lockbox.

A 20–30-question feasibility study cannot establish a <=2% false-accept guarantee or
learning/generalization. Do not weaken the historical R-gate thresholds to rebrand the
new experiment as a pass. A later confirmatory claim requires suitable independent data.
Keep benchmark/backend changes and harness improvements separate from learning claims.

Historical staging locations, to verify before a live run:

```text
/Volumes/main/mshtelma/verl/data/officeqa/officeqa_full.csv
/Volumes/main/mshtelma/verl/data/officeqa/treasury_bulletins_clean.zip
/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B
/Volumes/main/mshtelma/verl/models/GLM-5.3
```

Do not use the known-garbled `treasury_bulletins_transformed.zip`. The clean view is
not automatically semantically perfect either. At the documentation check, the local
corpus and Volume CSV were absent on this host; remote/staged availability is not
inferred from these path names.

Reuse established `df1`/air serving patterns after separate GPU authorization. The
last exercised image was `michaelshtelma587/verl-megatron-air:v7`; record actual image,
model, tokenizer and code identities for any new job. Never overwrite shared historical
score/trace filenames. Preserve unrelated uncommitted files and do not commit unless asked.

## 7. Document map and historical preservation

| Document | Role |
|---|---|
| `docs/officeqa_path_report_pilot.md` | Single authoritative isolated-test contract and execution plan |
| `docs/officeqa_rgate_handoff.md` | Next agent's focused assignment and environment/status notes |
| `docs/officeqa_rgate_recovery_plan.md` | Compatibility pointer; old R0–R11 plan superseded |
| `docs/officeqa_proof_certificate_design.md` | Direction-change record; old certificate design superseded |
| `officeqa_pilot_records/README.md` | Historical experiment index and current artifact policy |
| `officeqa_pilot_records/rgate_review_2026_09_14/` | Sealed final review, exact-hash evidence and frozen-code reproducers |
| `docs/history/officeqa_pre_path_report_2026_09_14/` | Exact compressed copies of pre-pivot documents; not current instructions |

The detailed historical baseline diagnosis, original experiments and superseded repair
ideas remain in those snapshots. They are preserved for evidence, not assigned as the
next implementation workload.
