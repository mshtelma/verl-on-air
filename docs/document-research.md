# OfficeQA: grounded reasoning over document archives

OfficeQA tests whether an agent can find and reason over information in a large
collection of real documents. The creators call this **end-to-end grounded
reasoning**: parse the documents, retrieve the relevant evidence, and carry out
the analysis needed to produce a precise, verifiable answer. The U.S. Treasury
Bulletins provide a public proxy for the messy document collections used by
enterprise analysts.

Questions range from direct lookups to following revisions across bulletins,
interpreting nested table
headers and footnotes, reconciling units and periods, fitting a regression, or
finding an external value such as an inflation index. Some require visual
information. The [technical report](https://arxiv.org/abs/2603.08655) describes
how the questions were written and independently verified, with feedback from
USAFacts to reflect the work of real analysts.

## Use the creators' datasets

The [official repository](https://github.com/databricks/officeqa) describes three
benchmarks:

| Benchmark | Questions | Documents | Intended use |
|---|---:|---|---|
| OfficeQA Pro | 133 hard | 697 Treasury Bulletins, 1939–2025 | Headline evaluation of frontier agents |
| OfficeQA Full | 246: all of Pro plus 113 easy | Same Treasury Bulletin corpus | System development and evaluation of less capable models |
| OfficeQA Pro V2 | 90 | 1,435 Federal Accounts of Receipts and Expenditures documents, 1793–2024 | Generalization to a different, harder archive |

Our [OfficeQA experiment](../usecases/officeqa/README.md) uses Full. It trains
on 113 easy and 90 hard questions and holds out
43 hard questions. That is our experiment's split; the creators' headline Pro
benchmark contains all 133 hard questions. Our result measures new questions
over a shared corpus.

The creators score final-answer correctness, normally at zero numerical error
tolerance with answer-format normalization. Our answer checker differs from
the creators' scorer, and our reward additionally requires a supported path citing observations
delivered to the agent. Our answer accuracy counts correct final answers in
well-formed reports, regardless of citation validity or the support verdict;
malformed reports count as incorrect. State the question subset and tool policy
when presenting results.

## Next experiment: OfficeQA Pro V2

Pro V2 is the creators' generalization benchmark: 90 hard questions over 1,435
federal receipts and expenditures documents from 1793–2024, most requiring
several sources. Start by comparing the original and trained OfficeQA actors
on all 90 V2 questions, before training or tuning on V2. This tests whether the
learned behavior transfers to a new corpus.

This is a proposed evaluation. The current loader, corpus tools and final-run
controller are specific to the Treasury Bulletin experiment; V2 support has
not been implemented.

| Component | Work needed for V2 |
|---|---|
| Data access | Obtain access to the creators' gated [V2 dataset](https://huggingface.co/datasets/databricks/officeqa-pro-v2); pin the dataset revision |
| Corpus | Convert the supplied parsed JSON to searchable text, preserving table headers, units, footnotes and document provenance |
| Question loader | Read the V2 schema: no `difficulty` column and semicolon-separated `source_files`; keep answers and source hints in scorer metadata |
| Prompt and tools | Replace Treasury-specific names and date filters; check that retrieval can reach every document in the V2 archive |
| Answer scoring | Pin and qualify the creators' `reward.py` at zero tolerance; check differences from our current answer checker |
| Evidence scoring | Reuse observation delivery tracking and the support judge; calibrate against V2 table and calculation examples |
| Evaluation | Fix the baseline and trained comparison, sampling budget and full-corpus access before running either model |

Use the complete 1,435-document corpus for an end-to-end retrieval evaluation.
Providing only the files named in each question's `source_files` creates an
oracle-source evaluation and must be labeled separately. Likewise, the current
agent has no web-search or image tool; questions needing those capabilities
remain in the evaluation and their limitations must be reported.

Report answer accuracy, supported pass@1, completion rate, latency, tool calls
and tokens together. Pair results by question, retaining repeated samples together
when estimating uncertainty. Repeat with additional training seeds to assess
how consistently the improvement transfers. If V2 later becomes training data,
reserve a new evaluation set before tuning.

## Adapting the implementation

The reusable pieces are the agent loop, delivered-observation tracking,
calculation tools and scoring pipeline. The dataset reader, corpus metadata,
prompt and experiment checks need their own adapters. In particular,
`prep_data.py` expects the 246-question Full CSV, and `final_run.py` checks the
203/43 split and 100-update horizon.

See the [code map](../usecases/officeqa/README.md#find-the-code),
[new-usecase guide](new-usecase.md) and
[presentation guide](showcase.md). [SOURCE.json](../usecases/officeqa/SOURCE.json)
records the Miles code used by the port.
