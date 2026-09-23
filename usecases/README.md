# usecases — RL tasks, one folder each

← [verl-on-air](../README.md) · [build your own use case](../docs/new-usecase.md) · [running-jobs](../docs/running-jobs.md) · [tuning](../docs/tuning.md)

Each use case is a **thin** layer over the shared [`../engine/`](../engine): a reward, an
optional tool, a data-prep, an eval, and a handful of `air/` jobs. Copy one to start your
own — step by step in [`../docs/new-usecase.md`](../docs/new-usecase.md).

| use case | task | reward | tool | topology | headline |
|---|---|---|---|---|---|
| [`agentic-search/`](agentic-search) ⭐ | multi-hop QA as a retrieval **agent** over Databricks Vector Search | **rule-based EM** (no judge) | `vector_search` / `keyword_search` / `read_article` | 16×H100 (2 nodes) | the flagship demo — [`../RESULTS.md`](../RESULTS.md) |
| [`math/`](math) | competition math as a calculator-using agent (MATH-500) | **LLM judge** (GLM-5.3, served by the same job) | `calculator` | 32×H100 (2 train + 2 judge) | the judge-reward pattern |

The two differ deliberately **on the reward axis**, because that is the axis that decides
what infrastructure you need:

| | rule-based (agentic-search) | LLM judge (math) |
|---|---|---|
| cost | free, instant, deterministic | a served model — extra nodes, extra latency |
| when | the answer is a short span you can match | open-ended output, LaTeX, prose, "is this reasoning sound?" |
| reward manager | `naive` | `rate_limited` (async + concurrent; verl's default concurrency is **1**) |
| extra moving parts | none | judge staging, co-located serving, rendezvous, `NORM_ADV_BY_STD_IN_GRPO=False` |
| failure mode to watch | a rule that scores 0 for correct-but-misformatted answers | a judge outage silently scoring everything 0 — **fail closed** |

Start with the rule-based one if your task allows it. It is the cheapest agentic RL loop
in the repo and has half the failure modes.

## The contract

A new use case = these files, wired to the engine **by env var only**:

| file | engine hook |
|---|---|
| `reward.py` | `CUSTOM_REWARD_PATH` (+ `CUSTOM_REWARD_NAME`) |
| `tool.py` | `FUNCTION_TOOL_PATH` |
| `prep_data.py` → parquet | `parameters.train_files` / `val_files` |
| `eval.py` (imports `reward.py`) | `EVAL_SCRIPT` |
| `air/{1_prep,…,4_train,5_eval}.yaml` | the jobs |

Both use cases follow the same job numbering, so the shape is recognisable across tasks:
**prep → (stage/index) → baseline eval → train → eval (→ deploy)**. Always run the
baseline *before* training: it is the only thing that makes the trained number mean
anything, and it validates the eval harness for one node-hour.

`eval.py` importing `reward.py` is not a convention, it is the point — the *answer scoring*
is the same code in training and eval. It does not make the two runs identical: the eval
drives its own agent loop under a recorded policy (turn budget, per-request token caps, a
forced final answer for search), and the math use case optimises a judge's score while
the eval measures exact equivalence — a surrogate objective against an independent target.

## Where to go next

- [`../docs/running-jobs.md`](../docs/running-jobs.md) — run either use case end to end
- [`../docs/configuration.md`](../docs/configuration.md) — every setting, including the
  per-use-case `QA_*` / `MATH_*` / `JUDGE_*` variables
- [`../docs/tuning.md`](../docs/tuning.md) — which knobs move the number, in what order
- [`../docs/training-modes.md`](../docs/training-modes.md) — sync vs fully-async
- [`../docs/new-usecase.md`](../docs/new-usecase.md) — build your own
