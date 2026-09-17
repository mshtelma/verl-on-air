# OfficeQA Phase-2 — Valid questions, simple storage, measured scaling

**Status: revised PLAN, 2026-09-17. Runtime changes below are NOT implemented.**
This revision incorporates the critical review, simpler ingestion, and the owner's selected focus:
**main pool = no table/page/line hints, minimal necessary vintage constraints, and meaningful
multi-step analysis.** It is the Phase-2 specification, not authorization to launch GPU
jobs or training. Earlier pilot/handoff documents retain historical results; their readiness
claims do not satisfy the gates here.

## 0. Goal, decisions, and corrected starting point

**Goal:** build approximately **400–600 valid, provisionally learnable main-pool questions**,
then export the training split. Report accepted-bank, probed, selected, validation, and actual
training counts separately: reserving validation reduces the final training count. Fully referenced
controls do NOT count toward the main-pool target. The target is a planning estimate, never a
reason to relax validity, hint-policy, or analytical-content requirements.

Retained decisions and simplifications:
- **Main-pool policy:** keep physical solution locations private; publish only the question's
  concepts, periods, units, answer convention, and necessary vintage constraints. Build substantive
  multi-step questions from audited source series, not harder-looking wording around a simple sum.
- **Separate controls:** explicitly tagged table-localized or simple-aggregation tasks can test
  machinery; they are excluded from default main-pool selection, training export, and yield counts.
- **Compute ceiling:** 16 GPUs, initially two independent 8-GPU single-node jobs, four TP2
  replicas/collectors per node. TP2 is a benchmark candidate, not a proven throughput win.
- **Question bank:** immutable `questions.jsonl` plus a small manifest; no Parquet bank required.
- **Durable probe output:** the existing per-episode JSON capture, written once per attempt.
- **SQL result store:** a rebuildable UC Delta projection of those SAME JSON files, ingested
  with `COPY INTO` by one standalone, post-run client. No result Parquet, second result buffer,
  direct per-episode SQL writes, JSONL fallback, or required background ingestion thread.
- **Parquet remains only at the final training-data boundary** used by the existing pipeline.
- **Existing 150:** revalidate their inputs as seeds/controls. A retained main candidate must meet
  the new hint and analysis policies and be re-probed under its new qid. Simple templates do not
  become main-pool tasks merely by rewording or relabeling them. Do not port old probe results.
- **Curriculum:** a mixed pool, not an imposed easy-to-hard schedule. Difficulty is relative to
  a specified actor/controller, not an intrinsic property of a question.
- **Real hard-133:** training-disjoint evaluation, with semantic duplicate exclusion. It has
  evaluator-development exposure and must not be called an untouched lockbox.

The checked-in `officeqa_pilot_records/hard_synth_probe_learnable.json` contains **1,199 samples**:
149 questions have eight, `HS0019` has seven. Its 100-question learnable list uses the old probe
criterion. The review's replay of the available captures found 585 old-probe passes versus 534
under training report parsing/exact answer matching, before semantic judging. These are diagnostic
results, not certification of the labels or of a new trainable pool. Preserve historical artifacts.

## 1. First fix the scientific contract: what makes a question valid?

### 1.1 Arithmetic correctness, extraction correctness, and answer uniqueness are different

Computing a number from selected cells proves only the arithmetic **conditional on those cells**.
It does not prove that the cells were assigned to the right column/month, that their units and
qualifiers are understood, or that the question uniquely selects that reporting vintage.

Admission therefore requires all three:
1. **Faithful extraction:** the selected source cells, headers, periods, units, and relevant
   footnotes are recoverable and have a defensible interpretation.
2. **Unambiguous task:** the actor-visible question specifies the interpretation that determines
   the answer; no private metadata supplies a missing source/date/concept constraint.
3. **Correct calculation:** a supported template computes the uniquely specified result with
   an explicit unit/precision policy.

The guarantee is deliberately limited: **a unique answer under the declared source contract in
our frozen corpus**, not proof of an underlying historical/economic truth. A matching text excerpt
alone does not certify that OCR matches the original bulletin. Suspected transcription damage
requires source-page/HTML adjudication or exclusion, not a stronger label from the same parser.

### 1.2 Main-pool default: precise questions, private solution locations

**Keep the solution private, not the definition of the problem.** Main-pool questions state the
financial concepts, calendar/fiscal periods, units, comparison/aggregation semantics, and answer
precision. Include edition month/year constraints only when needed to resolve admissible vintages.
Do NOT supply table numbers, page/line offsets, document paths, direct links, quoted table titles
as navigation directions, winning periods, extracted values, search queries, or a solution recipe.
Necessary concept qualifiers remain explicit even when their wording resembles a source header.
Shared tool documentation may explain file/line indexing; the prohibition is on question-specific
solution locators. Normal retrieval results may reveal locations AFTER the actor finds the evidence.

Maintain two distinct payloads:
- **Actor-visible:** business question plus minimal public constraints. A renderer whitelist,
  not the whole candidate dictionary, supplies the question/requirements to collection and training.
- **Private generator/verifier:** exact files/editions, table and column identities, cell locations,
  values, intermediate calculations, and gold answer. Keep these in `golden_path` and validation
  metadata. Do not pre-retrieve the right table or restrict the actor's corpus to the gold files.

Choose eligible reporting vintages deterministically before probing (initially the earliest eligible
edition providing the required complete, validated series). When a constraint is necessary, the
visible question identifies the edition governing each operand; exact table locations remain private.
We are asking what those vintages report, not implicitly asking for the latest revised figures.
If no edition is named, the candidate must pass the stronger policy in §1.3; never omit the constraint
merely to make retrieval harder.

Private audit example in the supplied corpus:
- `treasury_bulletin_1943_10.txt`, Table 4: Agriculture Department, June 1943 = **313**.
- `treasury_bulletin_1944_01.txt`, Table 4: the corresponding value = **318**.

The existing generator ignores the earlier partial-year series before checking consistency
(`hard_question_gen.py::_monthly_series`). An explicit vintage can resolve which record governs;
it cannot certify that OCR or accounting interpretation is correct. Reject unresolved damage or
conflicting representations within that scope. No majority vote, modal vector, benchmark matching,
actor-success-based source choice, or widened numeric tolerance may resolve a contradiction.

Illustrative MAIN candidate (not an admitted question until its source/semantic checks pass):

> Using the monthly figures reported in the January 1944 Treasury Bulletin, how many percentage
> points higher was the Agriculture Department's share of war-activity expenditures in its
> highest-share quarter of 1943 than its share for the full calendar year? Calculate each share
> from aggregated expenditures and round only the final difference to two decimal places.

The edition resolves vintage; the actor still must find the appropriate concepts/table, retrieve
two monthly series, form quarterly shares, identify the maximum, and compare it with the annual
share. Do not tell it the table or winning quarter. §1.6 specifies the private calculation.

**Removing hints requires revalidation.** Check the FINAL public wording against plausible matching
tables/columns within the allowed vintage: if another interpretation gives a different answer,
refine the financial concept/qualifiers or reject the main candidate. If only supplying a physical
table locator makes the question usable, tag it `pool_role=control`, not `main`. Blindly deleting
"Table 4" from an already validated question does not preserve its validity.

A fully table-localized version may be retained as a separately tagged calibration/control task,
with its own qid and probe results. Search snippets, grep, reads, and valid equivalent calculations
remain acceptable routes for main tasks; no minimum tool count or prescribed golden traversal.
The verifier enforces only the STATED vintage/concept requirements, not a hidden exact-table rule.

**Trade-off:** a named edition still narrows document search. Report vintage-constrained main tasks
separately from full-corpus retrieval tasks; do not call either guaranteed hard from its wording.
Probe the actual no-location-hint question and measure transfer to hard-133 under a frozen protocol.

### 1.3 Omitting an exact edition: stronger checks, not the initial focus

Main-pool focus does not require solving unrestricted corpus-wide source selection immediately.
If an edition constraint is omitted, one of these must establish an unambiguous answer:
- **Answer invariance:** audit all admissible relevant source interpretations as below.
- **Explicit selection rule:** a public rule such as "the latest bulletin published by [cutoff]
  containing the complete requested monthly series" selects a unique eligible source/answer.
  Validate the candidate editions and rule within an audited family. A vague "as of [date]" is
  not enough; do not silently choose different revisions for individual months. This is an optional
  later family, not a new global reprint resolver or a way to inflate the initial main-pool count.

For invariance, agreement among complete twelve-month vectors alone is insufficient:
- Audit **all overlapping required month cells**, including partial reprints, before assembling
  annual vectors. Compare compatible table concepts, qualifiers, periods, and normalized units.
- Conflicting values for a required month, ambiguous aliases, or an unparsed potentially relevant
  table make invariance unresolved. Missing parser coverage is not evidence of agreement.
- Restrict a half-year question's check to its required months. Do not certify a conflicting
  input series merely because discrepancies happen to cancel in its final aggregate.
- Record audit scope and coverage; agreement within supported extraction families is not a
  proof about every possible reading of the entire corpus.

Keep a small overlap-conflict audit/regression fixture for supported families now. Do NOT build a
universal financial ontology, all-document table IR, or proof interpreter as a prerequisite.

### 1.4 Minimal per-question specification and quality gates

Extend the existing `golden_path` records; use ordinary dictionaries, reusable aggregation helpers,
and the small compositional family set in §1.6. No new DSL is needed. Store:

| Part | Required information |
|---|---|
| Task | Template/version, `pool_role=main` or `control`, exact visible question, public hint/vintage policy, private operation and ordered operand roles, required periods/months |
| Source selection (PRIVATE) | Edition/file + content hash, table caption/locator, full header path and raw category label, meaningful qualifiers/footnotes |
| Inputs/derivation (PRIVATE) | Exact cell locations and raw strings, parsed decimals, units/scale and periods; intermediate values and source/table/input counts as audit descriptors, not proof of hardness |
| Answer policy | Scalar numeric kind for v1, requested output unit, precision and rounding rule; no gold value in this policy |
| Gold (PRIVATE) | Canonical answer string computed from the admitted inputs, stored separately from the identity/policy |
| Review | Validity and public-prompt-audit status/reasons, validation version, source-page-check status, related-derivation grouping and evaluation-exclusion decision |

Mechanical gates for EVERY accepted question:
- Preserve identity separately from display cleaning. Footnote stripping and dedup-suffix removal
  must not merge different columns or concepts; a label such as "Agriculture" alone is insufficient.
- Require an unambiguous calendar/fiscal-year interpretation and exact expected month coverage
  for EVERY operand series (12 for annual/quarterly analysis or the declared subrange). Reject
  duplicate, missing, null, uncertain, or shifted month cells.
- Parse entire numeric tokens with an explicit allowed footnote/revision-suffix grammar. Reject
  unrecognized suffixes, OCR guesses, nonfinite values, and ambiguous separators; do not accept
  an arbitrary numeric prefix as `parse_number` currently can.
- Require known units and compatible definitions across operands. Restrict summation to monthly
  flows, not stocks, rates, or year-to-date totals. Reject unexplained concept/coverage changes.
  Permit contextually audited total/denominator columns as PRIVATE operands even if the legacy
  standalone-category filter rejects "Total"; never merge totals from different table scopes.
- Keep template-specific conditions explicit: positive inputs for geometric means; positive base
  totals/denominators for growth and ratios; subset membership when calling a ratio a "share";
  directional and signed/absolute semantics; ratio of aggregates versus mean of ratios; and one
  final rounding step. Maxima with ties are valid only when the requested scalar remains unique;
  do not hide a tie-break convention or supply the selected period as a hint.
- Use deterministic decimal arithmetic/precision appropriate to each template. Declare rounding
  (including ties) in the contract and wording. A published annual total is a diagnostic cross-check,
  not a replacement for the requested monthly sum. Explain permissible source-rounding differences;
  quarantine unexplained discrepancies rather than assuming equality or ignoring them.
- Re-read cited bytes independently of generation, verify column/period selection, and recompute
  with independently authored test expectations. A second call to the generator is not independent
  validation. Verify all selected figures can be obtained through the actual bounded actor tools.

### 1.5 Semantic auditing without another large infrastructure project

Start with a short allowlist of understood table families and the three compositional families
below; retain the existing six simple templates as helpers/controls. Audit representative REAL
contexts and positive/rejection fixtures, especially reprints, hierarchy, footnotes, units,
fiscal/calendar boundaries, OCR, and the final hint-free public wording. An optional bounded
machine audit sees question + source context without generated gold or actor pass-rate. It is a
check, not an authority that can override a mechanical contradiction.

Human review remains unavailable unless explicitly arranged. Do not invent human certification.
If a flagged family cannot be adjudicated from reliable source context, exclude it and report the
lost yield. Reuse existing authorized tools/judging for bounded audits; no new always-on service.

**Gate V:** every admitted candidate passes the mechanical and public-question contracts and
belongs to an audited supported family; no unresolved case is valid. For `main`, this includes
the no-location-hint policy. Only then spend GPUs on difficulty; validity alone is not hardness.

### 1.6 Main-pool generation: three meaningful analytical compositions

Implement three ordinary template functions over audited series, not a generic reasoning graph.
Each task requires dependent analytical steps: intermediate quantities determine the final
comparison. Do not pad a simple question with irrelevant operations, giant contexts, or extra
independent subquestions. Standalone year sums/means and the old two-year differences are not
main-pool material just because they are labelled "hard" or contain many cells.

In the PRIVATE formulas below, `S(X,p)` is the sum of printed monthly flow values for concept X
and period p. Periods, concepts, units, vintage and final precision are defined in the public task;
source locations, intermediate values and gold are not. All answers remain scalar for v1.

| Family | Analytical task and private calculation | Admission requirements |
|---|---|---|
| `quarter_share_gap` | Form four quarterly shares, select the largest, compare with the annual share: `100 * (max_q(S(A,q)/S(T,q)) - S(A,year)/S(T,year))` percentage points. | Two complete compatible monthly series; A is a documented component of T; positive quarterly/annual denominators; no pre-revealed winning quarter. |
| `relative_growth_gap` | Aggregate two concepts in two years, compute each annual growth rate, compare them: `100 * ((S(A,y1)/S(A,y0)-1) - (S(B,y1)/S(B,y0)-1))` percentage points, A minus B. | Four complete series; positive base-year totals; each concept comparable across years; necessary operand-vintage mapping public; no table locations supplied. |
| `cross_table_ratio_change` | Retrieve economically meaningful numerator/denominator series from different audited tables, aggregate two periods, compare ratios: `100 * (S(X,p1)/S(Y,p1) - S(X,p0)/S(Y,p0))` percentage points when ratios are expressed as percentages. | Verified cross-table concept/period/unit alignment and positive denominators. A ratio need not be a subset share; wording must not pretend it is. Reject arbitrary combinations of unrelated measures. |

These descriptors guide generation, not how the actor must solve the task. A genuine shorter
supported derivation is allowed. Do not remove valid summaries or add tool-count requirements to
force a longer trajectory. Cell counts, number of tables, and number of operations are not an
empirical hardness certificate.

Initial priority is these families with minimal vintage constraints and no physical location hints.
Fiscal/calendar alignment and scalar trend/regression families can follow ONLY after their source
and answer contracts are validated; they are not prerequisites for this first main-pool batch.

## 2. Experimental validity: selection is a proxy, not a learning guarantee

### 2.1 Shared scoring and separate failure types

Introduce one small shared synthetic-answer contract used by the probe and synthetic training
reward. Parse only an explicit valid terminal `submit_report`; no last-number fallback from
reasoning, error text, unfinished calls, or malformed reports. Normalize legitimate unit/format
variants according to the question, then compare at its declared precision. Remove the blanket
`abs_tol=0.5` / `rel_tol=0.001` rule. Preserve the real benchmark's pinned scoring policy separately;
changing synthetic scoring must not silently redefine a published benchmark metric.

Record three separate outcomes:
- answer correctness;
- report/reference/value validity and semantic support where actually judged;
- execution validity (the actor was evaluated under the declared functioning environment).

Cheap triage uses **answer correctness AND valid submission AND the deterministic path gates**.
It is an upper bound/proxy for positive grounded reward, not its equivalent. Run a bounded,
preselected, stratified semantic audit of real synthetic rollouts across templates and observed
bands before calling the pool training-ready. Include negative and suspicious cases; do not
validate only successful-looking rollouts. Preserve judge UNKNOWN separately.

Missing corpus/index, unavailable sandbox/server, transport errors, or a harness context overflow
are infrastructure failures, NOT incorrect answers. Retry only declared transient failures with
bounded attempts and unchanged sample keys. A valid policy abstention, wrong answer, or declared
turn-budget exhaustion is a model outcome and must NOT be retried to obtain a success. A deliberately
specified context-budget stopping policy can count as a model failure only if implemented and
matched across probe/training; an accidental HTTP context error cannot substitute for that policy.

### 2.2 Sample counts, uncertainty, and the actual optimization signal

Base configuration: named checkpoint, 80 turns, temperature 0.7, **eight valid samples/question**;
freeze all other effective decoding and tool settings too. Probe the FINAL no-location-hint main
question, not an oracle-location variant. Report main/control roles and analytical families
separately; never transfer a control's difficulty estimate to a main question. Do not assume eight
valid samples when eight attempts were scheduled. Incomplete questions are excluded from export
with counts/reasons visible; do not assign a normal bucket from one or two available records.

At n=8, thresholds 0.05/0.95 simply distinguish 0/8, 1–7/8, and 8/8. Name these
`observed_all_failure`, `observed_mixed`, and `observed_all_success`, with `incomplete` separately.
The mixed set is **provisionally learnable**. For a true success probability of 0.2, 0/8 occurs
about 17% of the time: an extreme observation does not prove zero future gradient.

If budget warrants refinement, predeclare a separate fixed follow-up probe of 16 fresh samples
for the extreme groups, with its own probe manifest/ID and fresh sample indices 8–23. Finish that
batch rather than stopping at the first success/failure. For those questions, provisional selection
uses the fixed follow-up rate (1–15 successes out of 16); retain the initial eight for audit. Report
stage-specific counts/rates; pooled rates are descriptive, not an optional-stopping confidence
claim. The original eight-sample result stays immutable. This is optional, not an online sampler.

GRPO needs within-group variation in the **actual reward**. All answer-correct samples can still
vary in support/path score. Validate that the cheap proxy captures useful reward variation on the
small judged subset; retain extremes for later reassessment rather than declaring them useless.
The pinned async recipe lacks built-in online difficulty filtering; offline triage is only a
practical heuristic, not a theoretical replacement. A different training initialization checkpoint
requires a matching probe before treating its difficulty bands as current.

### 2.3 Related questions, holdouts, and reporting

Assign a fixed train/validation split by related derivations, not row position or independent qid
hashing. Group templates/wordings/vintages that reuse the same concept-period monthly series,
including paired location-hinted controls. New compositions connect ALL their input groups.
Prefer assigning base-series split blocks before composing and reject cross-split pairings; a
shared denominator must not bridge training and validation. Keep connected groups together and
report large groups/insufficient coverage rather than breaking them to hit a validation ratio.
Corpus-document sharing alone is not forbidden; leakage of the same underlying calculation is.

Exclude real hard-133 duplicates and close derivation variants using question semantics plus
source/concept/period/operation footprints where available. A changed UID, wording, unit scale,
or vintage is not sufficient separation. Ambiguous overlap needs adjudication or exclusion;
equality of numeric answers alone is not a duplicate test. Add the known generator recreation of
UID0003 as a regression case. Keep benchmark gold/exclusion metadata outside actor tools.

Choose default export membership from `pool_role=main` AND the training partition only. A control
that happens to have mixed outcomes must still be excluded; `difficulty=hard` alone is not enough.
Keep the synthetic validation partition fixed, including its easy/hard outcomes, rather than moving
questions into training after probing. Control-based policy training would require a separate,
explicitly approved mixture, with related-derivation leakage checks; it is not part of this export.
Record the selection protocol; do not tune it repeatedly using hard-133 outcomes. Report validity
yield, main/control counts, analytical families/independent groups, public hints, execution coverage,
observed rates, semantic-audit limitations, and later paired held-out performance separately.

### 2.4 Main-pool admission and useful failure modes

A main training candidate must pass **all** of: Gate V, the no-location-hint public audit, a
meaningful supported composition (§1.6), split/benchmark exclusion, complete execution coverage,
and the declared difficulty-selection criterion. Do not fill a main-pool shortfall with controls,
standalone aggregations, ambiguous questions, or more complicated-sounding paraphrases.

Use the bounded real-rollout audit to distinguish retrieval, concept/period interpretation,
calculation, and report failures. Infrastructure defects or unclear gold are not useful hardness.
If a family is consistently solved under the final prompt, retain it for diagnosis rather than
claiming it is hard because its reference calculation has several steps. Keep too-hard/extreme
observations for the already specified reassessment policy; do not infer impossibility from 0/8.

## 3. Identity, durability, and deterministic replay

### 3.1 Small manifests, explicit keys

- **`qid`:** SHA-256 of a canonical immutable question/input specification: exact wording,
  template/version, source identities/hashes, concept/period/ordered roles, operation, and answer
  policy. Serialize only these explicit specification fields deterministically; omit computed
  answers/subtotals and audit/outcome metadata. Do not sort away operand roles. The same spec
  producing conflicting gold answers is a hard error, not a second acceptable qid. Also reject
  conflicting active questions hidden behind version/wording changes; qid dedup does not replace
  semantic grouping/exclusion in §2.3.
- **Bank manifest:** hash of sorted `questions.jsonl`, frozen corpus inventory/checksums,
  generation/validation versions, public hint/vintage policy, pool roles, exclusions and input-group
  split membership. A new bank or corpus release is immutable and separately reviewed; never append
  blindly to a bank in flight. Adding/removing a locator changes the question/qid and requires a
  new probe association; control and main results are not interchangeable.
- **`probe_id`:** fingerprint of the selected bank/qids and effective actor environment: checkpoint,
  tokenizer, image/code/dependency versions, retrieval/index and tool limits, controller/funnel,
  sampling/seed policy, serving topology/context and declared sampling schedule. No mutable aliases.
- **Logical sample key:** `(probe_id, qid, sample_idx)`. Each collector invocation gets a unique
  `attempt_id` and executes a key at most once; restarting an episode uses a new attempt namespace.
  Seeds are deterministically derived per sample/request; record them. Repeat HTTP attempts must
  not silently become additional independent samples. Seeds do not promise bit-identical GPU reruns.
- **Score/export manifests:** freeze selected capture hashes, scorer/contract versions, thresholds,
  split and output hashes. Re-scoring does not mutate captures or require regeneration; scoring
  versions are recorded separately from the actor experiment. No latest-timestamp-wins semantics.

Idempotency means **reusing validated immutable observations and replaying deterministic transforms**,
not asserting that stochastic generation will reproduce identical bytes.

### 3.2 Output ownership and recovery

Proposed layout (one flat final-parts directory simplifies ingestion):

```text
banks/<bank-id>/{questions.jsonl, manifest.json, validation.jsonl}
probes/<probe-id>/probe_manifest.json
probes/<probe-id>/_staging/                         # never an ingestion source
probes/<probe-id>/parts/<qid>_s<k>__<attempt-id>.json
probes/<probe-id>/workers/<shard>__<attempt-id>.json # worker status/metrics, unique owner
probes/<probe-id>/finalized/<manifest-id>.json      # one finalizer owns global membership
```

Every capture carries its logical key, attempt/worker identity, schema version, seed/config
association, bank-validated pool role/analytical family/public hint policy, execution status/error
category, and the existing report/observations/raw turns. Gold and the private golden path live
in the bank, not in actor context.

Write a complete capture outside the watched prefix, close and validate it, then publish a unique
final file. Test the real Volume publication/readback semantics: do not assume local `os.replace`
proves atomicity on FUSE. On resume, parse and validate schema, key, configuration, and record
completeness; an empty/corrupt/wrong-probe file is not completion. Preserve rejected attempts with
reasons and retry eligible missing work. No overwrite of previous attempts or historical captures.

Run one owner per shard. Multiple execution-valid captures for one key (including wrong answers
or abstentions) must not increase sample count: use deterministic, outcome-independent attempt-ID
ordering, not best answer or latest timestamp. After all known workers have stopped, one finalizer
lists validated attempt files and their hashes and freezes the chosen attempt per logical key.
Unchosen attempts remain auditable. Later arrivals do not silently alter a sealed result. Resume
and finalization share the same validation/selection helper.

Workers never scan all parts to rewrite a shared `captures.jsonl` or global `manifest.json`.
Global consolidation is optional and belongs only to the finalizer. Keep memory bounded by
streaming records rather than having every worker load all captures.

**Gate C:** finalization checks exact expected-key membership, uniqueness, schema/config validity,
checksums, and eight valid model outcomes per MAIN question (or its declared follow-up budget).
A 950-main-question initial probe expects **7,600 main logical slots**, not 7,600 arbitrary rows.
Controls must satisfy their own declared budgets/key sets and cannot fill missing main slots. Partial
reports are allowed and clearly marked; normal final export fails until missing/invalid slots are
resolved or an explicit reduced cohort is frozen with exclusions disclosed. Never silently shrink
individual denominators or classify infrastructure failures as difficult questions.

## 4. Storage: ingest the existing JSON captures, after collection

```text
Frozen corpus → validated questions.jsonl → sharded GPU collection → immutable JSON parts
                                                                      |
                              CPU finalization + one COPY INTO client ←┘
                                              |
                            UC Delta attempt-results projection
                                              |
                         shared CPU scoring + frozen selection/split
                                              |
                           train.csv → {train.parquet, test.parquet}
```

**One durable representation, one ingestion path.** No result Parquet writer, no second compact-row
buffer, no per-worker SQL commits, no direct-INSERT replay queue, no JSONL/Parquet fallback matrix.
GPU jobs do not need warehouse credentials and never wait for SQL. Ingestion runs after BOTH jobs
stop; job A does not own a background loop that can abandon job B's tail.

`probe_ingest.py` is a small standalone client using the SQL Statement Execution API:
- Ingest only validated files enumerated by the frozen manifest, using `COPY INTO` with
  `FILEFORMAT=JSON` and a typed `SELECT` projection. Batch explicit file lists within the documented
  API/SQL limits. Do not point unrestricted ingestion at staging, corrupt files, or other probes.
- Keep the Delta table compact: probe/qid/sample/attempt IDs, schema/config identifiers, pool role,
  analytical family and public hint policy, termination/execution status, raw terminal report text,
  observation/delivery counts, worker ID, and normalized capture URI. File hashes belong in the
  frozen manifest and are checked by the client; no self-referential file hash or extra
  manifest-ingestion table is needed. Do not copy
  the large observation/raw-turn arrays into Delta.
- Leave numeric correctness, report validity, and difficulty to the shared Python scorer. It reads
  the frozen bank and selected captures for deterministic path checks; the compact SQL row alone
  cannot reconstruct reference/value validity. SQL is a queryable projection, not a second oracle.
- Poll statement IDs to terminal success; HTTP 200 is not proof of SQL success. Handle pending,
  failed/canceled statements, bounded transport retries and all result chunks. Preserve statement
  IDs/errors. Parameterize values, validate identifiers, and never log credentials.
- Use one ingestion client with one statement in flight. `COPY INTO` tracks loaded files, not
  business keys, and overlapping invocations can conflict; a warehouse is not a global mutex.
  Immutable filenames plus manifest-selected logical keys give the application-level guarantee.
- Loaded files must never be appended to or edited: `COPY INTO` can skip a previously loaded file
  even after modification. Rebuild a projection from frozen source files into an explicitly chosen
  fresh target when needed; do not overwrite a shared historical table as automatic recovery.
- Warehouse outage leaves all episode data intact. Replay ingestion later, verify source hashes
  against the manifest, and compare keys/attempts/URIs and projected values with Delta. Scoring
  selects the frozen chosen attempt, not every attempt row. Ingestion failure is visible but does
  not require rerunning actors. A local-files scorer supplies the reference result for parity tests.

Select profile/warehouse and verify UC permissions at implementation time using the Databricks
core/SQL/Unity Catalog guidance. The previously listed df1 warehouse is historical discovery, not
an assertion that it is currently running or authorized. Authentication belongs to the standalone
client's trusted environment, not a reused Docker-pull secret or a token embedded in GPU configs.

If raw JSON ingestion becomes a measured bottleneck, compact JSONL batches can be produced in a
separate replayable post-processing optimization. They are NOT part of v1. Periodic visibility can
likewise be added later without changing storage ownership. No image change is expected for this
storage design; the final dataset prep still uses its existing `datasets`/Parquet dependencies.

## 5. GPU layout: hypotheses to measure, not capacity guarantees

The initial candidate is **4 × TP2 per node**, pinned to `{0,1}`, `{2,3}`, `{4,5}`, `{6,7}`, with
one collector process and prewarmed read-only BM25 instance per replica. Shard by the BASE qid
before expanding samples: `int(qid, 16) % shard_count`. Never use Python's process-dependent
`hash(qid)`. Test partition completeness across independent interpreters/hash seeds. Validate
shard indices and reject duplicate/unassigned ownership.

Small expert widths and two KV heads motivate benchmarking lower TP, but do not prove TP8 is
wrong or that four replicas deliver four times the throughput. Account for duplicated weights,
profiling/activation/communication/CUDA-graph memory and the model's hybrid attention/state caches.
TP4 fallback means two workers/node and four total at the same **16-GPU ceiling**, not an implicit
upgrade to 32 GPUs. Freeze the revised layout/assignment before collecting a new probe.

- Start at 16 concurrent episodes/replica; tune against memory, preemption, CPU tool latency,
  host RAM, and end-to-end throughput. Do not assume the GIL or GPU compute is the sole bottleneck.
- Measure **actual actor input tokens plus generation headroom**, including long-tail trajectories.
  Judge-request length is a different quantity. Test 131072 versus the required actor budget;
  halving maximum context does not by itself double KV memory. No unmeasured "rare overflow" waiver.
- Compare TP2/TP4 and, where affordable, TP8 on the same representative question/sample set and
  fixed controller; store separate probe/config identities. Include long contexts and tool-heavy
  cases, not only the first easy-to-generate questions.
- Record successful, execution-valid episodes per GPU-hour, token throughput, latency quantiles,
  actor/collector errors, CPU utilization, vLLM running/waiting/preemption/cache metrics and GPU
  memory/utilization. GPU busy percentage alone is not an acceptance criterion or proof of SM use.
- Stage and validate model/corpus/index once per node before collectors import tools. Fail on
  missing data or an unintended keyword-search fallback; pin and record retrieval dependencies.
  Add tool deadlines and a no-progress watchdog so a stuck regex/tool cannot idle GPUs indefinitely.
- The orchestrator health-checks every server, propagates child failures, writes worker-specific
  status and cleans up process groups. Size full-run timeouts/cost ceilings from measured throughput
  including startup/tails/retries. The two jobs may run sequentially if capacity is unavailable.

## 6. Existing runtime defects: required fixes, with explicit scope

These are implementation requirements, not claims that earlier green tests established safety.

| Boundary | Required change / acceptance test | Gate |
|---|---|---|
| Compute isolation (`scripts/tools/py_compute.py`) | Add PID isolation, fresh `/proc`, and session isolation; preserve filesystem/network restrictions. A sandboxed snippet must not signal a dummy outside process. Test the full command on AIR, fail closed, never enable the unsandboxed fallback. | Before any live actor probe |
| Training evidence (`path_report_agent_loop.py`) | Record AFTER actual tool clipping/rendering; mark delivery only at the consuming generation. A figure clipped from the actor context must not be recorded as delivered. Same-request read/compute dependencies must fail chronology checks. | CPU parity before full probe; live transport proof before GRPO |
| Controller parity (`path_report_agent_loop.py`, collector) | Share submit-only termination, final-turn submission handling, nudges, tool gates, truncation and generation budgets. Last allowed turn may submit; plain text without submit is not completion. Test against the pinned upstream methods, not only local helpers. | Before matched difficulty probing |
| Answer scoring (probe + synthetic reward) | Shared terminal parser, answer contract, and deterministic gates; preserve separate benchmark policy. Include the replayed loose-tolerance/malformed-terminal failures. | Before full probe |
| Clean-node startup (`dispatch_agentic.sh`, staging) | Wire corpus/index staging into training nodes that execute tools, not just eval; execute retrieval/compute canaries before rollout. No success based on leftover node state. | Before live GRPO |
| UNKNOWN in training (`officeqa_path_report_reward.py`) | Preserve typed unresolved status; exclude affected groups from ALL losses/normalization or stop before an update. Scalar zero/advantage-only masking is insufficient. Export status counters; process-local unconsumed diagnostics are not monitoring. | Before live GRPO |
| Completion (`run_grpo_fully_async.sh`) | A real traceback plus cancellation must remain failure. Neither a cancellation string nor existence of a partial checkpoint proves completion. Verify explicit completion/checkpoint integrity and propagate failures. | Before live GRPO; do not copy the old guard into probe orchestration |
| Judge lifecycle (`serve_judge.sh`, training YAML) | Coordinate with training lifetime/readiness; do not stop a required judge at four hours in a nine-hour job. Judge loss triggers the UNKNOWN policy, not continued ordinary-zero training. | Before live GRPO |

The inference/data plan does not require implementing a universal training framework rewrite.
Training-specific gates can be worked separately, but **no Phase-2 learning run is authorized by
successful generation/probing**, and a pool probed under a different controller remains provisional.

## 7. Implementation map and tests

All paths below are repository-relative; NEW denotes planned files, not existing implementation.
Keep helpers small and specific; use stdlib tests plus in-image integration where dependencies matter.

| File(s) | Work |
|---|---|
| `scripts/officeqa/hard_question_gen.py` | Three compositional main families (§1.6), contextual denominator inputs, private golden paths and whitelisted no-location-hint public rendering, minimal vintage policy, explicit main/control roles; retain quality/overlap checks, strict answers, qids, JSONL manifests, semantic exclusions and input-group split rules |
| `scripts/officeqa/answer_contract.py` (NEW), `scripts/officeqa/path_report_pilot.py`, `scripts/reward/officeqa_path_report_reward.py` | One synthetic answer/submission criterion; declared precision/units and shared use; benchmark scoring stays versioned separately |
| `scripts/officeqa/probe_records.py` (NEW) | Small shared key/schema validation, resume and deterministic attempt-selection helpers; no generic orchestration framework |
| `scripts/officeqa/path_report_collect.py` | Probe/sample metadata and seeds, stable sharding, validated immutable attempt publication, error classification, actor token/latency metrics, per-worker manifests; remove per-worker global consolidation |
| `scripts/officeqa/hard_synth_probe.py` | Qid-aware episode CSV with pool roles, manifest finalization, shared scoring and complete denominators, role/family-separated reports, optional fixed follow-up; qid joins and default main-only train CSV with grouped splits; remove `int(buid[2:])` positional lookups |
| `scripts/officeqa/probe_ingest.py` (NEW) | Standalone typed JSON `COPY INTO`, API polling/replay and manifest/table parity; no Parquet writer or collector-side SQL thread |
| `scripts/serve_and_probe_sharded.sh` (NEW) | Per-node staging, serving replicas, collectors, health/metrics/watchdogs and clean teardown; no warehouse wait |
| `scripts/officeqa/prep_officeqa_path_report.py` | Preserve qids, pool roles, public contracts/requirements and frozen splits; require main + training membership for default train export, not only `difficulty=hard`; fail on missing/duplicate IDs or private-location leakage; verify 80/20/5 prompt/controller agreement |
| Runtime files in §6 | Close the scoped safety/parity/training failure-handling defects and add regression coverage |
| `air/115`, `air/116a`, `air/116b` (NEW recipes) | Smoke and two full half-jobs; versioned bank/probe/config inputs, worker ownership, measured resource/time limits; no automatic training continuation |

Required tests, not an arbitrary target test count:
- **Validity:** real 313/318 reprint conflict; partial series cannot certify invariance; explicit
  edition scopes; same label/different hierarchy or footnote; fiscal/calendar and missing/duplicate
  months; unknown units/OCR suffixes; positive/zero-base template constraints; rounding/annual-total
  distinctions; independent cell readback; hint removal that introduces a second interpretation.
  Assert both accepted controls and rejection reasons.
- **Main-family math/public interface:** independent expected results for all three compositions;
  ratio-of-sums versus mean-of-ratios, shares versus ratios, percent versus percentage points,
  signed growth gaps, tied maxima and one final rounding step. Validate contextual denominator
  columns. Main prompts must omit table/page/line locators, navigation titles/paths, intermediate
  values and winning periods while retaining all necessary concept/vintage/precision constraints.
  Retrieval results may expose normal source locations AFTER the actor finds them; that is not a
  hint leak. No pre-retrieved gold context or gold-file-only corpus view is allowed.
- **Experimental:** shared probe/reward answers including malformed/non-submit, signs, units,
  decimal precision and multiple-number strings; nonzero reward proxy limitations; incomplete
  groups; no retry-until-success; fixed follow-up batches; related-derivation split and UID0003
  exclusion. Controls cannot enter main yield/selection/export even if labelled `hard` with mixed
  outcomes. Test four-series grouping/shared denominators and main/control paired-task separation.
- **Identity/durability:** deterministic qids and conflicting-gold rejection; independent-process
  sharding under different hash seeds; changed checkpoint/context/corpus cannot resume an old
  probe; corrupt/truncated/key-mismatched files; crash during publication; concurrent duplicate
  attempts; frozen outcome-independent winners; no global worker-output races; bounded memory.
- **SQL/recovery:** stubbed pending/failed/timeout API responses, idempotent replay after unknown
  acknowledgement, warehouse outage, invalid file exclusion, explicit JSON casts/schema, all chunks
  read, source-hash validation, and exact manifest/table key, URI and projected-value parity. Test
  actual Volume publication and warehouse ingestion separately: mocks cannot prove FUSE/UC behavior.
- **End to end before GPUs:** small bank → expanded episodes → scripted captures → finalization →
  score → selected CSV → actual training/test Parquet round-trip. Assert qids, role-separated counts,
  main-only train membership, frozen split, contracts and public-only filled controller prompts.
  Missing rows are errors, not a quietly empty `--difficulty easy` selection. Keep all historical
  artifacts unchanged.
- **Runtime:** sandbox outside-process denial, post-truncation delivery/chronology, final-turn
  submit/no-submit parity, clean-node tools, judge outage without parameter updates, and failure
  propagation despite cleanup/partial checkpoints. Existing helper-only tests are not sufficient.

## 8. Execution order and go/no-go gates

1. **Validity and main-task construction (CPU):** implement the supported source families and
   three compositions; audit final no-location-hint questions and their private derivations. Emit
   main/control/rejected/unresolved reports and real-source regressions. Revalidate old inputs,
   not merely old wording; no historical pass-rate ports or simple-template quota filling.
2. **Contracts and recovery (CPU):** implement keys, shared scoring, capture ownership, qid-aware
   export, group splits, and the complete scripted bank-to-Parquet test. Close probe safety/parity
   gates from §6. Freeze the intended probe protocol and starting actor checkpoint.
3. **Warehouse-only smoke:** with explicit profile/permissions, validate JSON projection, schema,
   file selection and replay against the warehouse without allocating GPUs. SQL statements remain
   unvalidated until this passes. No table/warehouse provisioning is authorized by this document.
4. **GPU smoke (`air/115`, separately authorized):** representative ~24 valid MAIN questions × 8
   samples across the three compositions; any location-hinted controls get a separate declared
   budget and report. Run preflights, controller checks, TP/concurrency/context comparison,
   interruption/resume and post-run ingestion. Simulate A finishing before B in CPU tests.
   Freeze measured settings and resource/time acceptance limits from MAIN-task performance.
5. **Scale bank (CPU):** target ~950 VALID MAIN candidates with meaningful compositions, no physical
   location hints, minimal vintage constraints, and independent groups/audited source families.
   Publish immutable bank/probe manifests; no smoke-result reuse across changed identities. If
   yield is insufficient, choose more audited main families or a lower target—not extra controls,
   cosmetic complexity, or weaker validation.
6. **Full probe (`air/116a` + `air/116b`, separately authorized):** 16 GPUs total, initial shards
   0–3 / 4–7 of eight. Persist attempts independently, resume missing eligible work, stop GPU jobs
   promptly. Bounded retries cannot turn unresolved infrastructure errors into model failures.
7. **Finalize/ingest/score (CPU + warehouse):** freeze exact membership; require Gate C and
   table/files scoring parity. Emit descriptive bands and uncertainty; perform optional predeclared
   follow-up sampling under the manifest policy, not ad hoc selective top-ups.
8. **Bounded semantic audit and export:** verify proxy-to-grounded-reward usefulness and source-
   requirement enforcement on real rollouts (judge use separately authorized, sequential if needed).
   Fix systematic defects/revalidate affected families; do not relabel silently. Export by qid
   with explicit main + training membership, the fixed validation set, `--difficulty hard`, a
   run-specific output directory, and verified 80/20/5 controller configuration. Assert controls
   and private solution locations cannot enter the main training prompt or its reported yield.
9. **Stop for review:** deliver counts, limitations, manifests, costs and Parquet artifacts.
   Phase-2 GRPO requires a separate plan, explicit authorization and ALL training gates in §6.

The smoke determines numeric throughput/error/time thresholds BEFORE a full launch. Completion
and safety gates have no acceptable missing-key/corrupt-data waiver. No live job or SQL statement
has been run as part of this document revision.

## 9. Scope limits and references

**Not v1:** universal table/proof engines; global reconciliation of every financial concept;
unscoped questions without validated invariance/source-selection rules; table-localized or
simple-aggregation quota fillers in the main pool; online dynamic sampling; a new storage service;
result Parquet/dual-write queues; required background SQL ingestion; or automatic training.

Implementation must verify pinned versions and use primary platform documentation. Relevant
references (no claim of a live workspace check):
- COPY INTO JSON/projection, file tracking, concurrency, and file-list limits:
  `https://docs.databricks.com/aws/en/sql/language-manual/delta-copy-into`
- Statement Execution polling, failures and result chunks:
  `https://docs.databricks.com/api/workspace/statementexecution/executestatement`
- Pinned vLLM memory allocation (not a promise that halving context doubles cache):
  `https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/worker/gpu_worker.py`
- Bubblewrap boundary is determined by invocation options:
  `https://raw.githubusercontent.com/containers/bubblewrap/main/README.md`

Simplicity constraint: narrow question contracts, one JSON capture path, one post-run ingester,
shared checks, and measured serving—not a new platform or universal proof system.
