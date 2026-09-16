# OfficeQA path reporting — replacement for the certificate proposal

**Direction changed on 2026-09-14 UTC.** The old filename remains for compatibility.
**Authoritative specification:** `docs/officeqa_path_report_pilot.md`.
**Handoff:** `docs/officeqa_rgate_handoff.md`.

## Decision

Test **actor-authored answer + path reports now**, in an isolated inference/offline-
judging experiment. Do not first build a universal document model or have the judge
construct a certificate on the actor's behalf.

| Former proposal — superseded | Current isolated test |
|---|---|
| Model all table/header/cell semantics | Leave layouts intact; model/judge interpret actual source context |
| TaskSpec slots and allowed proof plans | Question plus short free-text claims and observation references |
| Universal deterministic operator engine | Inspect actual sandboxed Python, inputs and outputs; semantic audit |
| Judge-proposed certificate first, actor reporting later | Actor writes the report from the first pilot |
| `cite(...)` plus a new terminal tool protocol | One terminal assistant JSON object initially |
| Large training integration prerequisite | Standalone inference capture and offline evaluation; no optimizer |

The report is an **auditable claim, not a soundness guarantee**. Its references resolve
against controller-owned records of actual calls and delivered results. The judge checks
material source interpretation, calculation linkage and completeness. Deterministic
reference validity does not by itself establish correct table hierarchy or data use.

Confirmed fabricated actions/results invalidate the candidate reward even if the answer
is correct. Honest wrong paths and reporting mismatches are distinct diagnostics; do not
infer intent. Valid supported answers retain graded scores; verifier failure remains
UNKNOWN. No large negative penalty or live RL reward is introduced by this test.

A report may omit failed exploration and show a later supported correction or a genuine
alternative derivation. It does not prove what internally caused the model's answer.

There is no new `officeqa_proof_v1`/`officeqa_proof_v2` implementation or promotion in
this direction change. Existing source definitions using those names are pre-existing
partial work, not evidence that the old design is deployed or enforced.

Do not maintain a second report schema here. The sole schema, case-label distinctions,
penalty semantics, minimal file map and stop criteria are in the pilot plan.

## History

The exact earlier certificate proposal and related plans are preserved in
`docs/history/officeqa_pre_path_report_2026_09_14/`. Its claims that a ledger/engine
would make arbitrary-document verification mechanical or sound are not current claims.
Historical judge acceptance counts used uncertified labels; they must not be described
as independently established semantic false-accept rates.
