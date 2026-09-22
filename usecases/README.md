# usecases — RL tasks, one folder each

Each use case is a **thin** layer over the shared [`../engine/`](../engine): a reward, an
optional tool, a data-prep, an eval, and a handful of `air/` jobs. Copy one to start your
own.

| use case | task | reward | tool | headline |
|---|---|---|---|---|
| [`agentic-search/`](agentic-search) ⭐ | multi-hop RAG over Databricks Vector Search | **rule-based EM** (no judge) | search / read | the flagship demo — see [`../RESULTS.md`](../RESULTS.md) |
| [`math/`](math) | MATH-500, agentic | **LLM judge** (GLM-5.3) | calculator | the judge-reward pattern |

**The contract** — a new use case = these files, wired to the engine by env var:

| file | engine hook |
|---|---|
| `reward.py` | `CUSTOM_REWARD_PATH` |
| `tool.py` | `FUNCTION_TOOL_PATH` |
| `prep_data.py` → parquet | `train_files` / `val_files` |
| `eval.py` (reuses `reward.py`) | `EVAL_SCRIPT` |
| `air/{1_prep,…,4_train,5_eval}.yaml` | the jobs |

The two use cases deliberately differ on the reward axis — rule-based vs LLM-judge — so
you have both patterns to copy. The knobs to tune are in
[`../docs/tuning.md`](../docs/tuning.md); the training modes in
[`../docs/training-modes.md`](../docs/training-modes.md).
