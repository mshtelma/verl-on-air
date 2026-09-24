# Use cases

Each use case is a thin layer over [`engine/`](../engine): a reward, optional tools, a data-prep
script, an eval, and a few `air/` jobs. To start your own, copy one and follow
[docs/new-usecase.md](../docs/new-usecase.md).

| use case | task | reward | tools | GPUs |
|---|---|---|---|---|
| [`agentic-search/`](agentic-search) | multi-hop QA as a retrieval agent over Databricks Vector Search | exact match, no judge | `vector_search`, `keyword_search`, `read_article` | 16 H100 (2 nodes) |
| [`math/`](math) | competition math with a calculator (MATH-500) | LLM judge (GLM-5.3), served by the same job | `calculator` | 32 H100 (2 train + 2 judge nodes) |

They differ in the reward on purpose, because the reward decides how much infrastructure you
need:

| | rule (agentic-search) | LLM judge (math) |
|---|---|---|
| cost | free, instant, deterministic | a served model: extra nodes and latency |
| fits | short answers you can match | free-form output, LaTeX, reasoning |
| reward manager | `naive` | `rate_limited` (async and concurrent; verl's default concurrency is 1) |
| extra parts | none | judge staging, co-located serving, rendezvous, a failure budget |
| watch for | a rule that scores correct but misformatted answers as 0 | a judge outage; the reward has to fail closed |

Start with a rule if your task allows it.

## The contract

| file | engine hook |
|---|---|
| `reward.py` | `CUSTOM_REWARD_PATH` (and `CUSTOM_REWARD_NAME`) |
| `tool.py` | `FUNCTION_TOOL_PATH` |
| `prep_data.py` (writes parquet) | `parameters.train_files` / `val_files` |
| `eval.py` (imports `reward.py`) | `EVAL_SCRIPT` |
| `air/*.yaml` | the jobs |

Both use cases number their jobs the same way: prep, then staging or indexing, baseline eval,
train, eval. Run the baseline before training. It is the reference for the trained number, and
it checks the eval harness for about one node-hour.

`eval.py` imports `reward.py`, so answers are scored by the same code in training and in eval.
The loops still differ: the eval runs its own agent loop under a recorded policy (turn budget,
per-request token cap, a forced final answer for search), and math trains on a judge's score
while its eval measures exact equivalence.
