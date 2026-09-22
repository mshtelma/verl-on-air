# Does it actually learn?

Short answer: yes, modestly, on the one example task shipped here. This page is the
evidence — enough to show the loop works end to end, not a benchmark claim.

← [README](README.md) · run it yourself: [docs/running-jobs.md](docs/running-jobs.md) §4

## The setup

[`usecases/agentic-search`](usecases/agentic-search) trains `Qwen3.5-35B-A3B` with GRPO to
answer multi-hop questions as a **retrieval agent**: it searches and reads over a
Databricks Vector Search index, then commits a short span in `<answer>…</answer>`. The
reward is **rule-based exact match** — no LLM judge, no reward model, no labels beyond the
dataset's own gold answers. Data is MuSiQue (built so single-hop shortcuts fail); the
corpus is a union of three datasets' passages, so retrieval is a real decision rather than
"read the top hit".

Two properties make the number trustworthy:

- **The reward and the eval scorer are the same module** (`eval.py` does `import reward`),
  so what we optimise and what we measure cannot drift apart.
- **Base and trained are evaluated by the same job file**, with only
  `EVAL_MODEL_PATH` swapped — same 200 held-out questions, same 12-turn budget, same
  temperature.

## The result

| model | exact match |
|---|---|
| base `Qwen3.5-35B-A3B` | **54%** |
| GRPO-trained, best checkpoint (step 20) | **58.5%** |
| GRPO-trained, plateau (steps 30–40) | ~56–57% |

**+3 to +4.5 EM.** Caveats that belong right here, not in a footnote:

- The **turn budget moves the number on its own**: the *base* model goes 52 → 54 when
  given 12 turns instead of 8. An 8-turn baseline against a 12-turn trained model would
  have manufactured two extra points. Both eval jobs ship `EVAL_MAX_TURNS: '12'`.
- **n=200 → standard error ≈ ±3.5 points**, and these are single runs, not multi-seed
  means. The direction is real; the decimal is not.
- Different corpus, different absolute numbers. Bring your own and re-measure.

## How we knew which knob to turn

Worth more than the number itself. A single EM figure says *whether* something moved, never
*what to try next*, so [`analyze_traces.py`](usecases/agentic-search/analyze_traces.py)
factors each eval trace into two independent stages:

```
EM  =  recall                      ×  conversion
       did a retrieved passage        given the gold WAS retrieved,
       contain the gold answer?       did the model answer correctly?
```

Measured: recall **79–81%**, with conversion the gap. That settled the agenda — had recall
been the bottleneck the work would have been retrieval engineering (better index,
reranking); because conversion was, the work was policy improvement, which is what GRPO
does. It also explains the split we ended up with: **more turns protect recall, GRPO
improves conversion.**

## What worked, and what didn't

| lever | effect |
|---|---|
| turn budget 8 → 12 (`MAX_TURNS`) | ✅ +2 EM on the base model, recall-safe |
| GRPO with the pure EM reward | ✅ improved conversion — the trained delta |
| retrieval bonus (credit for surfacing the gold) | ❌ inert |
| deeper run · `rollout_n` 16 → 32 | ❌ flat |

The dead lever is the useful lesson. GRPO's advantage is
`(reward − group_mean) / group_std`, computed **within** each group of `rollout_n` samples
for one prompt. Recall was already ~80%, so a retrieval bonus fired on nearly every sample
in the group — landing in `group_mean` too, where it cancels itself out.

> **A reward term only teaches if it discriminates *within* the group.** One that almost
> always fires, or almost never does, is decoration. Same failure mode as a saturated
> dataset, one level down — which is why [docs/tuning.md](docs/tuning.md) puts "measure
> reward variance" above every other knob.

It survives as `QA_RETRIEVAL_BONUS` (default `0.0`) for corpora where recall is genuinely
low, the regime where it should help.

## The ceiling

Both compute levers and the reward-shaping lever were ruled out, so ~57% looks like a
**capability ceiling for pure-EM GRPO on this setup**, not something more GPU hours fixes.
Going materially higher would need a different *signal or inference* — self-consistency at
eval, a recall-weighted advantage, a harder-negative curriculum. None are implemented here;
that is the honest "what next", not a roadmap.

## Reproduce

Full walkthrough, including prerequisites: [docs/running-jobs.md](docs/running-jobs.md) §4.

```bash
make search-prep          # MuSiQue questions + passage corpus
make search-index         # Vector Search index (returns early; wait for status.ready)
make search-baseline      # the "before" number — never skip this
make search-train         # GRPO, fully-async, 16xH100
make search-eval CKPT=<…/global_step_20/actor/model/huggingface>

# then decompose, because the number alone won't tell you what to do next
python3 usecases/agentic-search/analyze_traces.py <base_traces.jsonl> <trained_traces.jsonl>
```

The exact settings are in the job files —
[`4_train.yaml`](usecases/agentic-search/air/4_train.yaml),
[`3_baseline_eval.yaml`](usecases/agentic-search/air/3_baseline_eval.yaml),
[`5_eval.yaml`](usecases/agentic-search/air/5_eval.yaml) — and what each one does is in
[docs/configuration.md](docs/configuration.md). The load-bearing ones: `MAX_TURNS=12`,
`rollout_n=16`, `actor_lr=2e-6`, `TRAIN_MODE=async` with a 1:1 rollout:trainer split,
`EP=8`/`GEN_TP=8`, pure-EM reward.

**→ Build the same thing for your own task: [docs/new-usecase.md](docs/new-usecase.md)**
