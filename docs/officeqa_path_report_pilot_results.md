# OfficeQA path-report pilot — results & case report (owner review)

**Date:** 2026-09-14 UTC · **Status:** both deliverables complete; **stop for owner review** (no RL/SFT run).
**Authoritative contract:** `docs/officeqa_path_report_pilot.md`. Running status: `docs/officeqa_rgate_handoff.md` §8.

This is a feasibility pilot. It asks two decoupled questions and answers each with machinery only (no
human reviewer, which is unavailable): **(A)** does the verifier soundly separate supported from
unsupported reasoning paths? **(B)** is the terminal-JSON path-report contract usable by a real actor?
All scores here are **offline previews** — never optimizer inputs. Three labels stay **separate**:
answer-correctness / report-faithfulness / semantic-support.

---

## Bottom line

- **Deliverable A (verifier soundness): PASS, decisively.** GLM-5.3 discriminates 34/34 on a
  truth-by-construction battery — 0 false-accepts, 0 false-rejects, 0 unknowns — and does so **for the
  right reasons**, including report-faithfulness (the one weak spot found and fixed earlier).
- **Deliverable B (actor usability): the contract works and the verifier holds on REAL rollouts, but
  actor report-yield is low.** Only 10/30 rollouts produced a valid structured report (13/30 exhausted
  the 16-turn budget with no report). On the 10 that did, the machinery behaved exactly to contract —
  most importantly it **zeroed a correct answer whose declared path was unsupported** (UID0011), the
  core anti-reward-hacking property, observed live rather than on an authored fixture.
- **Recommendation:** the verifier is sound enough to proceed, but **address actor report-yield (turn
  budget / report-prompting) before any learning run** — otherwise ~⅔ of rollouts carry no candidate
  signal. Details and options below. **No RL/SFT should start until an owner reviews this.**

---

## Deliverable A — verifier soundness (truth-by-construction, no human)

**Method.** We author (question, answer, golden-path) triples grounded in **real** Treasury-Bulletin
table bytes, so the accept/reject label is ground truth *by construction*. The judge sees only
question / answer / report / tool-history — never the label, family, or answer-correctness. Battery =
**34 cases** across three real table families (1941-01 monthly; 1939-01 Federal Expenditures-General;
1995-03 claims-by-country): 15 genuinely-supported, 14 must-reject (semantic), 5 deterministic defects.

**Result** (`air/83` v3, run `528897888311555`, GLM-5.3 TP16, 16×H100; `…/judge_validation_v3/`):

| | supported→supported | unsupported→unsupported | deterministic caught pre-judge | FA | FR | UNK |
|---|---|---|---|---|---|---|
| **34/34 match** | 15 / 15 | 14 / 14 | 5 / 5 | **0** | **0** | **0** |

Every must-reject case rejected **for the right reason** (from the judge's own per-case issues):
faithfulness ("287 is the November 1940 value, December is 375"), `units_mismatch` (caught the
"millions of dollars" header), `compute_mismatch` (caught a hardcoded `print(2253)`); `undelivered_ref`
was independently flagged by **both** the reference and value gates (defense in depth). The CRITICAL
FAITHFULNESS RULE generalized from the earlier 12-case set to 34 with **zero** over-rejection of the 15
supported cases.

## Deliverable B — actor usability on real rollouts (machine-labeled)

**Method.** Base actor (Qwen3.5-35B-A3B) runs 30 real easy OfficeQA questions through the collector
(`air/85`, run `599030212287940`; runtime observation IDs + delivery + chronology recorded). Those
captures are then scored offline against the validated judge (`air/86`, run `1097882710253790`;
`…/scored_v1/`): deterministic reference + value gates → answer-correctness auto-labeled from the
OfficeQA answer key → semantic support from GLM-5.3, **only** on reports that pass both gates.

### The funnel (30 rollouts)

| stage | count | note |
|---|---|---|
| rollouts | 30 | easy questions |
| emitted a terminal report | 17 | the other **13 hit the 16-turn budget still calling tools** (no report) |
| → valid structured report | **10** | reach the semantic judge |
| → explicit abstention ("DATA NOT AVAILABLE") | 3 | valid, not judged for support |
| → malformed JSON | 4 | (part of the 17 malformed total incl. the 13 turn-cap) |
| deterministic gates on the 10 valid | 0 bad refs, 0 fabricated values | actor cites real delivered obs, invents no values |

### The 10 judged reports

| uid | answer-correct | support (judge) | candidate | score | why |
|---|---|---|---|---|---|
| UID0020 | ✅ | supported | **scored** | 0.95 | |
| UID0023 | ✅ | supported | **scored** | 1.0 | |
| UID0043 | ✅ | supported | **scored** | 1.0 | |
| UID0048 | ✅ | supported | **scored** | 1.0 | |
| UID0051 | ✅ | supported | **scored** | 1.0 | |
| UID0067 | ✅ | supported | **scored** | 0.95 | |
| **UID0011** | ✅ | **unsupported** | zero | 0.0 | **correct answer, but the path never establishes the question's concept → zero** |
| UID0047 | ❌ | unsupported | zero | 0.0 | wrong period (used Aug **1980**, not 1981) **and** wrong answer |
| UID0024 | ❌ | supported | zero | 0.0 | wrong answer (support retained separately) |
| UID0066 | ❌ | supported | zero | 0.0 | wrong answer; judge noted a *minor* citation slip but correctly kept it supported |

**6 scored** (correct answer **and** supported path), mean path-score **0.98**. **0 unknowns** — the
judge parsed cleanly on every real report (the dialect + faithfulness fixes generalize off-fixture).

### Contract invariants — demonstrated live (not just on fixtures)

- **Correct answer + unsupported path → zero reward (UID0011).** The answer "42" is right, but the
  judge found the declared path never establishes the "payroll savings plans" concept the question
  asks about (cited table is "Sales of Series E Savings Bonds by States", and a grep confirms no
  "payroll" content in the bulletin). This is the central anti-reward-hacking property, observed on a
  real rollout.
- **Faithfulness catches a period misattribution (UID0047):** "line 5156 'Aug. | 22,691' is August
  1980, not August 1981 … the August 1981 value is not present in this bulletin."
- **Labels stay separate:** UID0024/UID0066 are wrong-answer → candidate zero, but their support
  verdict is recorded independently (both "supported"). UID0066 shows the judge distinguishing a
  *minor attribution slip* (value correct and delivered within cited evidence → still supported) from a
  *material* misstatement (→ unsupported) — it did not over-reject.
- **Answer-correctness auto-labeled, no human:** 7 correct / 3 wrong across the 10 valid reports
  (`score_answer` vs the answer key); 20 unlabeled = 17 malformed + 3 abstention.

---

## Honest limitations

1. **Support is machine-judged, not human-audited.** Trust rests on Deliverable A's 34/34
   construct-validity result, not on independent human review of these specific 10 reports.
2. **Small scored set (n=6).** This is a feasibility signal, not a training-scale statistic; reward
   density estimates from it are noisy.
3. **Low report-yield is the headline usability gap.** 13/30 rollouts produce no report at all
   (16-turn budget exhausted mid-exploration) and 4/30 emit malformed JSON — so ~⅔ of easy-question
   rollouts carry **no candidate signal** as-is.
4. **Easy split only.** Harder questions (longer retrieval chains) will likely worsen (3).

## Recommendation → owner decision

The verifier is sound; the blocker for a learning run is **actor report-yield**, not verifier trust.
Before any RL/SFT, consider (in rough order of leverage): (a) raise / restructure the turn budget or
add an explicit "you must now emit the terminal report" checkpoint before the cap; (b) strengthen the
report-format prompt to cut the 4/30 malformed-JSON rate; (c) re-collect and re-measure yield. **These
are actor/prompt changes and out of scope for this pilot — flagged for the owner, not executed.**

**Provenance.** air/83 v3 `528897888311555`; air/85 `599030212287940`; air/86 `1097882710253790`.
Immutable outputs on the Volume: `…/eval/path_report/{judge_validation_v3, deliverable_b/collect_v1,
deliverable_b/scored_v1}/` (each with a manifest carrying episode + code SHA-256). Reproduce the CPU
verifier: `python3 scripts/officeqa/judge_validation.py --self-check` and the test suites
(69 / 25 / 9 + reward 43 / 12 / 18, all green).
