# Use cases

Each use case is a thin layer over [`engine/`](../engine): a reward, optional tools, a data-prep
script, an eval, and a few `air/` jobs. To start your own, copy one and follow
[docs/new-usecase.md](../docs/new-usecase.md).

| use case | task | reward | tools | GPUs |
|---|---|---|---|---|
| [`agentic-search/`](agentic-search) | multi-hop QA as a retrieval agent over Databricks Vector Search | exact match, no judge | `vector_search`, `keyword_search`, `read_article` | 16 H100 (2 nodes) |
| [`officeqa/`](officeqa) | table/document research with a cited evidence path | binary answer check + hosted GLM-5.3-Flash support judge | search, grep, read, list, isolated Python, `submit_report` | 16 H100 (2 training/rollout nodes); hosted judge uses no allocated GPUs |
| [`math/`](math) | competition math with a calculator (MATH-500) | LLM judge (GLM-5.3-Flash), served by the same job | `calculator` | 24 H100 (2 train + 1 judge node) |

They differ in the reward on purpose, because the reward decides how much infrastructure you
need:

| | Rule (agentic-search) | Hosted evidence judge (OfficeQA) | Self-hosted judge (math) |
|---|---|---|---|
| Cost | deterministic scoring | API tokens and latency | judge GPUs and latency |
| Fits | short answers you can match | answers whose cited sources must support them | free-form output, LaTeX, reasoning |
| Reward manager | `naive` | `rate_limited` | `rate_limited` |
| Extra parts | none | delivered-evidence ledger, support calibration, hosted credentials | judge staging, serving and rendezvous |
| Failure handling | malformed answers score zero | unresolved support or service errors stop the run | bounded judge failure budget |

Start with a rule if your task allows it.

## The contract

| file | engine hook |
|---|---|
| `reward.py` | `CUSTOM_REWARD_PATH` (and `CUSTOM_REWARD_NAME`) |
| `tool.py` | `FUNCTION_TOOL_PATH` |
| `prep_data.py` (writes parquet) | `parameters.train_files` / `val_files` |
| `eval.py` (imports `reward.py`) | `EVAL_SCRIPT` |
| `air/*.yaml` | the jobs |

The use cases number their jobs the same way: prep, then staging or indexing, baseline eval,
train, eval. Run the baseline before training. It is the reference for the trained number, and
it checks the eval harness for about one node-hour.

`eval.py` imports `reward.py`, so answers are scored by the same code in training and in eval.
For search and math the loops still differ: eval runs its own loop under a recorded policy (turn budget,
per-request token cap, a forced final answer for search), and math trains on a judge's score
while its eval measures exact equivalence.

OfficeQA uses the same continuous-token loop and answer/evidence scorer in
training and evaluation. Its [final recipe](officeqa/README.md) is the entry
point for reproducing the measured run; the smaller pilot remains documented
separately. [Results](../RESULTS.md) compares OfficeQA and MuSiQue under their
respective held-out protocols. [Document research](../docs/document-research.md)
is a blueprint for a new task using this pattern.
