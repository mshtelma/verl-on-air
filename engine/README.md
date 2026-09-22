# engine — the shared RL platform (write once, reuse everywhere)

All the genuinely hard infrastructure lives here and is **use-case-agnostic**. A use case
never edits the engine; it plugs in via a few env vars.

```
engine/
├─ train/
│   ├─ dispatch_agentic.sh        rank dispatcher: some nodes train, some serve the judge;
│   │                             resolves the use case's tool/reward and sets PYTHONPATH
│   ├─ run_grpo_fully_async.sh    fully-async GRPO (disjoint rollout/trainer GPUs) — agentic
│   └─ run_grpo_megatron.sh       synchronous GRPO (co-located) — the geo3k ladder
├─ serve/
│   ├─ serve_judge.sh             serve an LLM-as-judge as an OpenAI endpoint (math)
│   └─ serve_and_eval.sh          serve a model + run a use case's eval.py against it
├─ lib/                           hparams.sh (air params → shell), ray_cluster.sh (multi-node Ray)
├─ stage_model.py                 HF → Unity Catalog Volume model staging
└─ _site/sitecustomize.py         opt-in GRPO advantage hook (quarantine unknown-reward groups)
```

## The plugin seam

The dispatcher and launchers read these env vars, so a use case supplies only files +
values — never engine edits:

| env var | what the use case supplies |
|---|---|
| `CUSTOM_REWARD_PATH` | path to `reward.py` (verl imports it; its dir goes on `PYTHONPATH`) |
| `FUNCTION_TOOL_PATH` | path to `tool.py` (the agent's `@function_tool` defs) |
| `EVAL_SCRIPT` | path to `eval.py` (for `serve_and_eval.sh`) |
| `MULTI_TURN` / `MAX_TURNS` / `TOOL_FORMAT` | agent-loop shape |
| `REWARD_MANAGER` / `TRAINING_NODES` | rule reward (`naive`) vs judge reward (`rate_limited` + co-located judge nodes) |
| topology (`TP`/`EP`/`GEN_TP`/`ROLLOUT_NNODES`/…) | fixed per model+GPU — see [`../docs/tuning.md`](../docs/tuning.md) |

`dispatch_agentic.sh` derives `PYTHONPATH` from the resolved tool/reward paths, so `reward.py`
and `tool.py` import each other by bare name (`import reward`, `import tool`) and the same
holds in `eval.py` — training and eval share the exact scorer.

Training modes (sync vs fully-async) and how to switch: [`../docs/training-modes.md`](../docs/training-modes.md).
