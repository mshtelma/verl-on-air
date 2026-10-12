# A new use case: financial-report research

OfficeQA provides a working pattern for answering numerical questions from
documents with a checkable evidence path. A useful next task is **financial-report
reconciliation**: compare figures across annual or quarterly reports, calculate
changes or ratios, and return the value with citations to the retrieved tables.
This page is an implementation blueprint; that new dataset and experiment have
not been built or evaluated.

For example: “How much did operating cash flow change between these two annual
reports, in millions?” The agent must find the correct period and units in each
report, calculate the difference, and submit the answer with dependencies on
both retrieved observations. The same pattern suits budget reports and public
agency statistics.

## Reuse the proven pieces

Start from [OfficeQA](../usecases/officeqa/README.md) and keep the shared engine.
Separate the new dataset adapter from the evidence/runtime code so reviewers
can see which behavior changed.

| Piece | Reuse | Adapt for the new task |
|---|---|---|
| Agent loop | Continuous token history, observation IDs, delivery tracking, terminal report | Domain prompt and budgets after qualification |
| Retrieval | Search, grep, read, list; BM25 chunk interface | Document names, metadata and table-aware text extraction |
| Compute | Isolated arithmetic and limits | Source-directory masking and arithmetic calibration for the new corpus |
| Scoring | Answer check before evidence judgment; strict verdicts | Explicit units, rounding, aliases and required calculation dependencies |
| Preparation | Prompt/gold separation and hashed manifests | Domain CSV parser and a new train/held-out split |
| Evaluation | Same actor loop and scorer; four samples and question bootstrap | New held-out IDs and a predetermined checkpoint |
| Launch | Frozen snapshots, receipts, idempotency keys and budget gates | Job paths, experiment namespace and domain audit expectations |

OfficeQA's `prep_data.py` deliberately requires its official 246-question CSV,
and `final_run.py` audits the 203/43 split and 100-update budget. A copied
controller must use the new manifest's question and episode counts. These files
are a reference implementation, not a generic arbitrary-CSV loader.

## First experiment

1. Choose a licensed or public report collection with reliable table text. Store
   document IDs, fiscal period and units with the extracted content; inspect
   a small sample before generating questions.
2. Create manually checked lookup, subtraction and ratio questions. Gold values
   and reference hints stay in scorer metadata and never enter policy prompts.
   State the accepted rounding and units explicitly.
3. Hold out entire issuers or report periods before training, rather than only
   new question wording. Record the split rule and hashes. OfficeQA's existing
   result holds out questions over a shared corpus; document generalization is
   a new measurement that needs its own baseline.
4. Calibrate correct/incorrect values, wrong periods, wrong scale, unsupported
   citations and calculation dependencies. Judge outages remain infrastructure
   failures. Check that several training questions yield mixed reward groups.
5. Qualify the loop, train briefly, then freeze the final update count and
   before/after policy. Start with the measured 35B actor / 16-H100 layout and
   hosted judge to reuse the validated topology. A 2B actor on one 8-H100 node
   is a separate qualification, likely beginning with shorter lookup episodes.
6. Report supported pass@1 as the primary metric, with answer-only accuracy,
   submission rate, tools/tokens and judge validity beside it. Use question-level
   paired intervals, inspect improvements and regressions, then repeat with
   another training seed before making a general learning claim.

## Keep the example easy to use

Provide one README with a question, a tool/evidence diagram, prerequisites,
three run commands and a code map. Put protocol details in a separate run guide
and experimental history in results. Every run should use a unique data and
checkpoint namespace, and its evaluation should reuse the prompt, loop, tool
limits and scoring policy from the baseline.

Keep original Miles provenance in [SOURCE.json](../usecases/officeqa/SOURCE.json)
and record adaptations separately. Test domain data/units and evidence delivery
on CPU before changing the training topology. Use the [new-usecase guide](new-usecase.md)
for the engine hooks and [showcase guide](showcase.md) for presenting the result.
