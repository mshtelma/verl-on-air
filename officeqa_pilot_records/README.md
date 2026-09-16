# OfficeQA experiment records — history and the next isolated test

**Current direction, 2026-09-14 UTC:** a small **actor path-report feasibility test**,
using real easy questions, actual tool traces and offline full-GLM judging. It is
planned, not implemented or run. No new reward is approved for policy optimization.

- Active plan: `docs/officeqa_path_report_pilot.md`.
- Next-agent entry point: `docs/officeqa_rgate_handoff.md`.
- Conditional roadmap: `docs/officeqa_rl_plan.md`.

Paths above are repository-relative. The old universal table/proof-engine implementation
sequence is superseded. Exact pre-pivot documents remain under
`docs/history/officeqa_pre_path_report_2026_09_14/`.

## 1. New experiment — separate from these historical scores

The actor will return an answer plus references/claims describing its supporting path.
The runtime, not the actor, will record actual calls, delivered source text and executed
Python. Simple reference checks and the full GLM-5.3 TP16 judge will compare the report
to that history. No universal cell/header ontology or proof interpreter is required.

Confirmed fabricated or unsupported paths receive zero candidate reward even if the
answer is correct. Supported correct answers retain graded scores. Judge failure or
uncertainty remains UNKNOWN. These are offline previews; **there is no optimizer**.

Future outputs belong in a **new** `path_report_pilot/<run-id>/` directory with selected
UIDs, raw observations, actor reports, independent labels, judge requests/replies,
verdicts and versions. That directory/result does not exist yet. Do not reuse or
relabel old outputs to make it appear the path-report test has already run.

Distinguish answer correctness, report faithfulness and answer support. A truthful
report can describe a wrong calculation; a valid source elsewhere in a trace does not
rescue a materially false submitted claim. Earlier failed exploration is not misconduct.

## 2. Sealed final review and completed jobs

`rgate_review_2026_09_14/` contains the final review, reports, score files, logs,
exact-hash compressed bundles, code snapshots and portable reproducers. **Keep it
byte-for-byte unchanged.** Its implementation recommendations reflect the earlier
direction; use the active plan above for next work. Its measured evidence remains valid
within the qualifications stated in the review.

The last archived Jobs API check was **2026-09-14T10:58:22Z**. This documentation update
did not poll jobs again. All five reviewed runs were terminal at that check:

| Run | Historical result |
|---|---|
| `878119239646411` | air/82 attempt 1: scorer import failure |
| `657777974855885` | attempt 2: 317 scored; controls 78/80, labeled-negative accepts 121/220; gate failed |
| `591895497446045` | attempt 3: 317 scored; controls 63/80, negatives 72/220; gate failed |
| **`543029753940588`** | **attempt 4: 339 scored; controls 16/80, negatives 5/242; gate failed** |
| `616024033111491` | answer-mode smoke: two optimizer steps; successful integration smoke, not learning |

Attempt 4's report was generated at **2026-09-14T05:54:57Z**. It has case upper95
4.295%, nominal base-ID upper95 7.414%, UNKNOWN 11/339 and wrong-answer nonzero 0/17.
**55 of 62 scored control rejections came from literal gold-in-quote matching**; only
1/26 alternate-source controls was accepted. The earlier frozen-verdict CPU replay
accepted 18/80 controls; the actual live result is 16/80.

These counts use **uncertified, sometimes defective labels**, not independently
established grounding error rates. Completed gate failures are distinct from scorer/
service crashes. None is approval for grounded-policy optimization.

### What the five residual accepted negatives taught us

- **UID0002_N2:** a later correct 507 row survives an early mutation.
- **UID0148_T:** correct numeric/list values were benchmark-selected as wrong; format
  and answer correctness must not be used as retrieval labels.
- **UID0204_T:** reported 1.47 and 1.3 trillion inputs support a 0.17 difference even
  though the original answer was 0.18.
- **UID0189_R2:** two 13-month medians were credited without establishing source-month
  coverage; historical logging loss limits claims about the actor's original view.
- **UID0122_T:** common CPI scaling cancels, but category hierarchy remains unresolved:
  euro+yen gives 0.953 pp; including receivables gives 0.946 pp. Gold agreement cannot
  decide which components belong in the requested category.

Do not automatically relabel these cases or copy them into a new certified suite.
The detailed source-qualified review is in the sealed archive's README.

## 3. Original pilot artifacts — preserve unchanged

The original judge pilot sampled 100 trajectories: all 51 legacy-scored correct
answers plus 49 wrong answers. It was not a representative correctness sample and
not a set of GRPO sibling groups. Higher average reward did not prove a safer judge.

| File | Meaning |
|---|---|
| `officeqa_traces.jsonl` | 246 historical air/72 episodes; tool outputs were clipped during recording |
| `officeqa_grounded_scores_v1.jsonl` | 100 historical v1 reward records |
| `officeqa_grounded_scores_v2.jsonl` | 100 historical v2 rerun records; not a training-ready reward |
| `officeqa_grounded_scores.jsonl` | Same as v1, retained for history |
| `validation_report.txt` | Historical v1 narrative report |
| `validation_report_v2.txt` | Historical v2 narrative; optimistic conclusions superseded, not human labels |
| `rgate_scores.jsonl` | air/79 smoke: 8 resolved + 1 unknown, not 9/9 clearance |
| `officeqa_base_air71.json` | Original per-question baseline artifact |
| `air71_vs_air72.reconcile.json` | Per-UID comparison and scorer reconciliation |
| `officeqa_traces.measure.json` | Coverage/termination/scoring measurements of saved traces |
| `rgate_review_2026_09_14/manifest.json` | Sealed review file hashes and external historical-file hashes |

Historical v1→v2 parsing increased from 62/100 to 99/100, and mean reward among the
51 legacy-correct records increased from 0.688 to 0.956. That is not independent
validation of evidence support or the model's reasoning behavior.

## 4. Historical measurement caveats that still matter

- The saved trace baseline is **13/133 hard, 38/113 easy, 51/246 overall** with the
  legacy scorer. The pinned upstream scorer gives 14/133 and 52/246 by changing one
  list-parsing decision. This is scorer drift, not learning.
- There are 158 explicit abstentions, 12 missing final answers and 179 episodes that
  reached the 16-turn cap. Do not describe the baseline as mostly wrong-but-substantive
  numerical attempts or confuse reached-cap with missing-answer counts.
- Every original episode contains at least one output at the historical 3k recording
  cap; 1,987/3,335 outputs hit it. Today's full-string writer cannot recover lost bytes.
- air/71 and air/72 disagree on 76/246 predictions despite similar aggregate accuracy.
  Comparisons need the same frozen inputs/controller/scorer and paired analysis.
- The old mutation suite mixed wrong answers with wrong retrieval, retained alternate
  valid evidence and fabricated some alternative filenames. Mutations are not labels.
- Compute/prose/flat-marker authority, units, period heuristics, runner integrity and
  actual training failure handling had reproduced defects. Existing green unit tests
  were not sufficient validation.
- The prior sentinel/advantage patch does not establish whole-group exclusion from
  KL and other loss/denominator effects. The answer-only smoke did not test this.
- Repeated mutations are not independent questions; hard-133 has evaluator-development
  exposure. The new small pilot is not a <=2% false-accept or generalization claim.

The new test must retain trustworthy actual-tool capture and reviewed labels without
turning these lessons into a universal document-model project.

## 5. Reproduction and preservation

From repository root, without a judge/GPU:

```bash
A=officeqa_pilot_records/rgate_review_2026_09_14
PYTHONDONTWRITEBYTECODE=1 python3 "$A/tools/verify_archive.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$A/tools/replay_saved.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$A/tools/probe_contract.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$A/tools/review_latest.py"
```

The tools import frozen archived code and write fresh scratch outputs. Successful
reproduction demonstrates historical defects; it is not a passing production gate.
Checksums and exact associations are in the archive manifest, not dependent on old
`/tmp` inputs or mutable shared Volume output filenames.

Never overwrite raw historical JSONL, reports, frozen source snapshots or sealed
manifests. Store new labels as additive adjudications and new experiments in unique
directories. Preserve unrelated uncommitted work and do not commit without permission.
