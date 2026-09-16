# Final R-gate review and evidence archive

**Review cut-off:** 2026-09-14 10:58:22 UTC. **Verdict: NO-GO for grounded-policy optimization.**

This directory is an additive archive, not a replacement for earlier pilot records.
The current implementation is still the unsafe v2.1 implementation; this review did
not repair production reward/training code. The recovery implementation is specified
in `docs/officeqa_rgate_recovery_plan.md`; start an agent handoff with
`docs/officeqa_rgate_handoff.md`. Paths in this paragraph are repository-relative.

## 1. Final job status

Jobs API results were rechecked at the review cut-off. All five runs below are terminal;
there is no pending run among these reviewed jobs. This is not a claim about unrelated
workspace jobs.

| Run | Purpose | Execution/result |
|---|---|---|
| 878119239646411 | air/82 attempt 1 | FAILED: scorer import-path bug after judge startup; no completed suite |
| 657777974855885 | air/82 attempt 2, bare audit | Completed scoring 317 cases; FAILED thresholds |
| 591895497446045 | air/82 attempt 3, v2 checker | Completed scoring 317 cases; FAILED thresholds |
| 543029753940588 | air/82 attempt 4, v2.1 checker | Completed scoring 339 cases; FAILED thresholds |
| 616024033111491 | air/77 answer-mode smoke | SUCCESS; two optimizer steps, not a grounded-reward or learning validation |

The workspace labels completed failed gates as INTERNAL_ERROR/FAILED because the head
exits nonzero and peer nodes are terminated. Reports, score-file hashes and driver
logs distinguish those deliberate threshold failures from an infrastructure crash.

### Completed air/82 comparison

These metrics use the generator's **unverified labels**. They are NOT independently
established false-accept/true-accept rates for real grounded answers.

| Metric | Attempt 2 | Attempt 3 | Attempt 4 |
|---|---:|---:|---:|
| Cases scored | 317 | 317 | 339 |
| Labeled positives accepted | 78/80 (97.5%) | 63/80 (78.75%) | **16/80 (20%)** |
| Labeled negatives accepted | 121/220 (55.0%) | 72/220 (32.73%) | **5/242 (2.066%)** |
| Reported case-level one-sided 95% upper | 60.668% | 38.312% | **4.295%** |
| Reported base-ID upper | 54.084% | 38.010% | **7.414%** |
| Unknown rewards | 12/317 | 13/317 | **11/339 (3.245%)** |
| Wrong-answer gate nonzero | 0/17 | 0/17 | 0/17 |
| Overall gate | FAIL | FAIL | **FAIL** |

Attempt 4's report was generated at **2026-09-14T05:54:57Z**; scoring took 1,580.7
seconds. Its bundle SHA matches the 339-case bundle below, and its complete score-file
SHA matches the report. This is a completed result, not the preceding CPU prediction.

### Why attempt 4 is a regression, not near-clearance

Of its 80 labeled controls, 16 were accepted, 62 were scored zero and two were unknown.
The 62 scored rejections break down as:

- **55:** verified quotes do not literally contain the final gold string;
- **4:** period-label heuristic;
- **2:** judge says unsupported;
- **1:** anachronism heuristic.

Only **1/26** alternative-source controls was accepted (one more was unknown).
Earlier, a CPU ablation with saved attempt-3 verdicts and unchanged control trajectories
predicted the same failure direction: current v2.1 accepted 18/80 instead of 63/80.
The live result is 16/80, not 18/80. The old and new negative bundles differ, so aggregate
negative-accept improvements do not isolate the verifier change.

For the 24 anachronism-labeled cases, 20 zero scores came from literal-value absence,
only one from the anachronism checker, one from a negative judge verdict, and two were
unknown. Therefore even zero positive rewards in that family does not establish that
anachronism reasoning was fixed.

## 2. Review of all five remaining accepted negatives

### UID0002_N2 — surviving valid evidence, not a clean negative

The generator swaps an early gold-cell occurrence. Step 11 still contains the original
FY1934 Veterans' Administration 507 row and its column header, unchanged. The judge
quotes that surviving row. A blanket negative label is unjustified; do not punish
an answer for finding valid evidence after an earlier misleading observation.

### UID0148_T — benchmark/format mismatch is not wrong retrieval

The original prediction was `28,2444.28`; gold is `[28, 2444.28]`. The current strict
numeric/list parser accepts the values while the legacy benchmark selector returns 0.
The original compute outputs already include count 28 and geometric mean
2444.2808148794134, rounded to 2444.28. The transplant primarily fixes the commitment's
format. Correctness/formatting errors must be labeled separately from retrieval errors.
Full source-view certification is still required before promoting this as a positive.

### UID0204_T — a valid numerical route remains

The unchanged retrieval contains the specified September 2010 and September 2011
bulletins' reported trillion-dollar figures, 1.47 and 1.3. Decimal recomputation gives
`abs(1.47 - 1.30) = 0.17`. The original 0.18 used a different intermediate table/rounding
route. A wrong original answer does not mean the remaining evidence cannot support the
corrected answer. The task's reporting-vintage and rounding convention must be explicit.

### UID0189_R2 — the checker fails to establish full period coverage

The judge credits two 13-month medians while quoting only July 1969 through January
1970. The later monthly inputs occur in policy-authored compute arguments/stdout, not
in the corresponding visible source rows in the archived record. The checker tests two
numeric-looking lines, not required month coverage. A valid median certificate needs
all required inputs or a separately supported order-statistic proof; two lines do not
suffice. Because the underlying archive is historically truncated, this does **not**
prove the original actor never saw the missing months.

### UID0122_T — shortcut and hierarchy/arithmetic must be separated

The source outputs contain euro, yen, accounts receivable and total assets for all
six balance-sheet dates. A common CPI adjustment cancels within each same-date
numerator/denominator share, so requiring CPI retrieval mechanically would be wrong.
But the judge's prose groups accounts receivable with foreign exchange and securities:

- euro + yen yields **0.952882893112... percentage points**, rounded **0.953**;
- euro + yen + accounts receivable yields **0.946188486556...**, rounded **0.946**.

The final answer has a numerically valid candidate route in the displayed cells, but
which rows belong to the requested category requires source-hierarchy adjudication.
The judge's explanation is not a verified derivation. Preserve this as a semantic/
shortcut diagnostic until the original table hierarchy is independently certified;
do not automatically relabel it based on gold agreement.

**Conclusion:** the five accepts mix label defects, unresolved source semantics and a
real completeness-check weakness. There is no defensible corrected aggregate error
rate without independent relabeling. Do not call all five genuine judge false accepts,
and do not simply relabel them to make the gate green.

## 3. Findings across the whole approach

### F01 — mutations were incorrectly used as label oracles

U removed the literal answer, not its operands or alternative representations.
UID0023_U retains the requested 1938 $2,237,000,000, supporting 2.24 billion.
UID0043_U retains 302,665,000 and 47,976,000, supporting net 254,689,000.
UID0020_U retains source inputs AND the unrounded KL result. The stronger U/F blanker
still leaves supporting search snippets in some cases. R/N only mutate an early
occurrence; D often renames a file other than the supporting one. T conflates wrong
answers/abstentions/formatting with wrong evidence.

### F02 — alternate-source controls contradict their own date policy

`_alt_filename` moves two years forward OR backward. Nine A filenames predate the
minimum explicit question year; six of attempt 3's scored control rejections are these
impossible-date A fixtures. Renaming content does not create a real alternative source.
The earlier diagnosis blaming arbitrary preceding-filename association for the whole
acceptance collapse was not supported by the actual rejection reasons.

### F03 — answer correctness was mistaken for proof validity

G controls are selected by benchmark correctness, not independent support labels.
All 246 original traces hit the historical 3,000-character recording cap somewhere;
1,987/3,335 tool outputs are exactly 3,000 characters. All 51 G substrates are affected.
Changing today's trace writer and judge context cannot recover those lost bytes.

### F04 — string heuristics are not semantic or numeric verification

Attempt 3's other nine scored control rejections were period-check failures: missing
interior years of ranges, footnote dates mistaken for observation dates, line numbers
mistaken for years, and a set-based removal that deletes a real year because the same
year occurs in a filename. In v2.1, literal gold matching rejects calculations, units,
rounding, lists and currency formatting. Neither matching a number nor counting two
numeric-looking lines proves support. Source metadata even contributes to the numeric
cell counter. Single source does not mean single operand; UID0042_F got 0.875 in
attempt 3 despite `retrieved_all_components=false` because it used one bulletin.

### F05 — authority is lost at the training boundary

Offline verification searches ALL tool outputs, including compute stdout. Training
searches the entire flat solution. Pinned RateLimitedRewardManager strips special
role tokens. Printable `[tool_output:...]` markers are not authentication, and the
normal flat path does not recognize successful read-only evidence consistently.

Executed CPU probes with an affirmative stub judge (not live GLM attack trials):

- fabricated compute evidence after an unrelated genuine read earns **1.0**;
- the same read-only episode earns **1.0 structured / 0.0 flattened**;
- fake flat tool markers with zero actual calls earn **0.8**;
- a tool-only final tag in role-stripped text earns answer-mode **1.0**;
- a one-character quote expands into a whole evidence line;
- reverse containment accepts a longer fabricated quote containing a short real line;
- `targeted_right_category=false` can still produce positive eligibility;
- synthetic `(no matches found)` read results are counted as successful retrieval.

### F06 — the strict answer gate still lacks task-declared units/types

Percent metadata is parsed but not compared. Bare gold allows arbitrary added scales.
Examples accepted by current code: gold `0.5` vs prediction `0.5%`; gold `507` vs
prediction `507 billion`. Answer shape, reporting format, numerical value and proof
support must be separate fields, not inferred from a permissive benchmark selector.

### F07 — quarantine is not whole-group training exclusion

Executing the pinned manager's actual method with lightweight test doubles shows an
outer timeout returns ordinary 0.0 even when sentinel quarantine is enabled.
Zeroing advantages/returns does not remove KL loss or global denominator effects.
The smoke uses KL loss coefficient 0.01 and did not exercise unknown groups. A group
must be resolved/admitted before batch construction, with bounded refill and backpressure
credit handling; simply dropping groups can deadlock the 16-group async sync contract.

### F08 — the gate runner can false-pass invalid input

Reproduced overall PASS with a missing wrong-gate score, with 150 duplicates of one
negative score and 149 cases missing, and with NaN negative rewards. Exact UID-set,
uniqueness, schema, finite-number, label and coverage assertions are missing. Unknown
is not a demonstrated rejection. A green runner must also prove the evaluation was
complete and the labels/experimental scope were certified.

### F09 — independence and an untouched evaluation are absent

Both bundles have 139 nominal negative base IDs (some handbuilt IDs share facts), not
220/242 independent negative questions. Even 0/139 has an exact one-sided upper of
2.132%. The enforceable bound must use a declared independent unit. The current bundle
contains 63 hard UIDs (111 cases): separate from gradient training is not untouched
for reward development. Repeated prompt tuning on this bundle makes it development
data. With only 113 easy questions, an independent >=150-question/fact validation
claim needs additional suitable tasks or must remain blocked; repeated variants do
not manufacture new independent questions.

### F10 — provenance and completion reporting need explicit contracts

GPU runs overwrite the same Volume paths; reports identify the model only as `judge`
and lack full raw request/response and model/prompt/reward revisions. Current parser
output truncates quotes/reasons and cannot reconstruct every judge response. Scalar
threshold failures, infrastructure failures, incomplete evaluation and label-integrity
failures need separate report statuses while retaining fail-closed process exits.
The old optimizer OOM also showed why log-pattern exit guards alone are insufficient.

### F11 — data/view repairs remain necessary

The HTML parser loses parent colspan/header context and can confuse body `<th>` rows
with column headers. Character clipping can cut rows and give misleading read cursors.
Training and eval termination differ; training's second truncation layer can hide data
seen by eval. The smoke prep duplicates its final train row as validation despite its
held-out comment. Repair these without modifying historical baselines, and re-baseline
every arm under the new controller/corpus/view contract.

## 4. What remains useful

- Real easy questions and tiny learning/integration smokes, not mass synthesis.
- The actual bubblewrap boundary, scientific Python availability and sandbox probes.
- Strict-vs-benchmark metric separation, though the strict task schema still needs work.
- A real two-step answer-only training integration proof, not a learning result.
- Full GLM-5.3 TP16 infrastructure and fail-closed gate exits.
- The owner's **graded reward** requirement and GRPO std-normalization OFF.

The recovery plan keeps these and replaces the label, proof, authority and batch-admission
contracts. More prompt tweaking or changing exit 1 to success does not repair them.

## 5. Reproduction and checksums

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 officeqa_pilot_records/rgate_review_2026_09_14/tools/verify_archive.py
PYTHONDONTWRITEBYTECODE=1 python3 officeqa_pilot_records/rgate_review_2026_09_14/tools/replay_saved.py
PYTHONDONTWRITEBYTECODE=1 python3 officeqa_pilot_records/rgate_review_2026_09_14/tools/probe_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 officeqa_pilot_records/rgate_review_2026_09_14/tools/review_latest.py
```

The replay/probe tools import the archived v2.1 code, not future modified production
code; they do not contact a judge or perform optimizer updates. New outputs go to
fresh `/tmp` directories. **Successful reproduction means the known defects were
reproduced; it does not mean the production reward passes a safety gate.** Original
raw traces remain at `officeqa_pilot_records/officeqa_traces.jsonl` and are hash-checked.

The CPU replay's nonzero-error interval may use the historical local Wilson fallback;
use its exact counts only, not that interval as a new Clopper-Pearson gate result.
Existing tests were green (12/12 v2, 43/43 reward, 18/18 quarantine/generator/gate)
despite these reproduced defects. Additional regression coverage is required.

| Artifact | SHA-256 of original uncompressed bytes |
|---|---|
| Original 317-case bundle | `675561186dce6175bf6e28d5139047b8c6bb34dcf6b9897d64b85030f8694645` |
| v2.1 339-case bundle | `dd0e003d17df301a9aa1a2b6256478c32c35daf7817f619907ded783b8184de4` |
| Attempt 2 scores | `9064eec46e06b614ab5ca751400f204df0b9950c1b917f58a0ce65599a7c848b` |
| Attempt 3 scores | `873f51f87d4d665c7dc0114fcb4987d9f49d6c4642548fcd5e0170dbf0b3567b` |
| Attempt 4 scores | `98aa0077f5f55cb617f691ce897694385c5d2bba2f76ecc74be6f8dc8dfe5a4c` |
| Attempt 4 report | `b63f5f337f8ad67bd354962663de7e28ff2d38e5b4d0cb894d3423a0fa0673a5` |
| Original 246 traces | `9f1e93c633fd043f0e1f607c6e5f545b16a5bdb13e2175ce885b15c09bfb858a` |

`manifest.json` inventories every archived file plus decompressed bundle hashes.
`run_statuses/summary.json` records the latest API check. `upstream_snapshot` contains
only reviewed files from verl v0.9.0 commit
`483b8a009ba3a97563edee3a19887e4862b8094a`, with its license. Code source inspection is
not a substitute for the future installed-image integration tests.
