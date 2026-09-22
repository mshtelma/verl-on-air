# Results — agentic-search

**This is a showcase and a template, not a benchmark claim.** The numbers below are an
illustrative, reproducible example of what you get from the `usecases/agentic-search`
recipe on one dataset. Swap in your own corpus, questions, reward, or tool and re-run
the same jobs — the point is the *machinery*, not the leaderboard.

## Setup

| | |
|---|---|
| model | `Qwen3.5-35B-A3B` (MoE), trained with GRPO (verl, fully-async, 16×H100) |
| task | multi-hop question answering as an **agent**: the model runs a multi-turn tool loop (`vector_search` / `keyword_search` / `read_article`) over a **Databricks Vector Search** index and commits an answer in `<answer>…</answer>` |
| data | MuSiQue (multi-hop) questions; a Wikipedia passage corpus indexed in Vector Search |
| reward | **rule-based exact match** — no LLM judge, no reward model (`usecases/agentic-search/reward.py`) |
| eval | 200 held-out dev questions, EM, **matched 12-turn budget** for base and trained (`3_baseline_eval.yaml` / `5_eval.yaml`) |

The reward and the eval scorer are the **same code**, so "what we optimise" and "what we
measure" cannot drift apart.

## Headline

At a matched 12-turn eval budget, over the 200 held-out questions:

| model | EM |
|---|---|
| base `Qwen3.5-35B-A3B` | **54%** |
| GRPO-trained (best checkpoint, step 20) | **58.5%** |
| GRPO-trained (plateau, steps 30–40) | ~56–57% |

So roughly **+3 to +4.5 EM** from GRPO, on top of the base model, from a rule-based
reward and no judge. (Turns matter independently: extending the *base* model's budget
from 8→12 turns alone lifts it 52→54 — see below.)

Treat single-run deltas cautiously: n=200 gives a standard error around ±3.5 points, and
these are individual runs, not multi-seed means.

## How we found what to move: EM = recall × conversion

The useful diagnostic (`usecases/agentic-search/analyze_traces.py`) decomposes every
eval trace into two independent factors:

```
EM  =  recall            ×  conversion
       (did a retrieved      (given the gold was retrieved,
        passage contain       did the model answer correctly?)
        the gold answer?)
```

- **recall** is the EM ceiling — the model cannot answer what it never retrieved.
- **conversion** = EM given the gold was in a retrieved passage.

Measured on the traces, recall sat around **79–81%** and conversion was the gap. That
told us where to push: GRPO improved **conversion** (the model got better at *using*
what it retrieved), while the **12-turn budget** protected **recall** (more hops =
more chances to surface the gold).

## What moved the number — and what didn't

| lever | effect | kept? |
|---|---|---|
| **turn budget 8 → 12** | +2 EM on the base, recall-safe | ✅ yes |
| **GRPO (pure EM reward)** | improved conversion → the trained delta | ✅ yes |
| retrieval-shaped reward (bonus for surfacing gold) | **inert** — recall was already ~80%, so the bonus fired on nearly every sample in a group and cancelled under GRPO's within-group mean-subtraction | ❌ dropped |
| deeper/longer training run | flat — no late surge past the plateau | ❌ |
| denser GRPO groups (`rollout_n` 16 → 32) | no gain, same conversion | ❌ |

The retrieval-reward result is a nice GRPO lesson: **an additive bonus that fires on
almost every rollout in a group contributes ~nothing**, because advantage is
`(reward − group_mean)/group_std` and the bonus lands in `group_mean` too.

## The honest ceiling

Both compute levers (depth, group density) and the reward-shaping lever were ruled out,
so the ~57% plateau is a **capability ceiling of pure-EM GRPO on this setup**, not a
training-dynamics artifact you can spend your way past. Reaching materially higher would
need a *signal or inference* change — e.g. self-consistency at eval, a recall-weighted
advantage, or a harder-negative retrieval curriculum — not more of the same knob. None
of those are implemented here; they're the honest "what next".

## Reproduce

```bash
# 1. data + corpus, then the Vector Search index (wait until ONLINE)
air run --file usecases/agentic-search/air/1_prep_data.yaml   -p df1 --watch
air run --file usecases/agentic-search/air/2_build_index.yaml -p df1 --watch

# 2. baseline (base model) and 4. train, then 5. eval the checkpoint
air run --file usecases/agentic-search/air/3_baseline_eval.yaml -p df1 --watch
air run --file usecases/agentic-search/air/4_train.yaml          -p df1 --watch
air run --file usecases/agentic-search/air/5_eval.yaml           -p df1 \
  --override env_variables.MODEL_PATH=<…/global_step_20/actor/model/huggingface> \
            env_variables.EVAL_MODEL_PATH=<same> \
            env_variables.EVAL_OUT=/Volumes/main/mshtelma/verl/eval/agentic_search_trained_step20.json
```

Then decompose the traces:

```bash
python3 usecases/agentic-search/analyze_traces.py <base_traces.jsonl> <trained_traces.jsonl>
```
