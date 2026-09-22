# Results — a case study in agentic RL

**This is a showcase and a template, not a benchmark claim.** What follows is one
reproducible example of the [`usecases/agentic-search`](usecases/agentic-search) recipe on
one dataset, written up honestly — including the two levers that did nothing and the ceiling
we could not get past. Swap in your own corpus, questions, reward or tool and the same jobs
run unchanged; the transferable part is the *method*, not the digits.

← back to the [README](README.md) · run it yourself:
[docs/running-jobs.md](docs/running-jobs.md) §4

---

## 1. The task, and why it is hard

Multi-hop question answering, as an **agent**. The model gets a question and three tools over
a **Databricks Vector Search** index — `vector_search`, `keyword_search`, `read_article` —
and must commit a short span in `<answer>…</answer>`.

The questions come from **MuSiQue**, chosen deliberately: it is constructed so that
single-hop retrieval shortcuts fail. No one passage holds the answer, so the model has to
find an entity, read about it, discover the next entity, and search again. And the corpus is
a **union** of three datasets' contexts rather than only the gold passages for the questions
being asked — otherwise retrieval degenerates into "read the top hit".

Both are dataset decisions made for one reason: keep the task inside the band where **the
reward can still separate good rollouts from bad ones**. A version of this demo on HotpotQA
scored ~62% EM before any training, which leaves GRPO very little to work with. That is the
reward-variance idea from [docs/tuning.md](docs/tuning.md) applied to data, and ignoring it
is the most common way an RL project wastes a week.

| | |
|---|---|
| model | `Qwen3.5-35B-A3B` (MoE), GRPO via verl, fully-async on 16×H100 (1 node generating, 1 training) |
| reward | **rule-based exact match** — no LLM judge, no reward model ([`reward.py`](usecases/agentic-search/reward.py)) |
| eval | 200 held-out dev questions, EM, **matched 12-turn budget** for base and trained |

The reward and the eval scorer are **the same module** — `eval.py` does `import reward` — so
"what we optimise" and "what we measure" cannot drift apart. That invariant is part of the
template: [docs/new-usecase.md](docs/new-usecase.md) §4.

## 2. The result

At a matched 12-turn eval budget, over the 200 held-out questions:

| model | EM |
|---|---|
| base `Qwen3.5-35B-A3B` | **54%** |
| GRPO-trained, best checkpoint (step 20) | **58.5%** |
| GRPO-trained, plateau (steps 30–40) | ~56–57% |

So roughly **+3 to +4.5 EM** from GRPO on top of the base model, from a rule-based reward
with no judge and no labels beyond the dataset's own gold answers.

Two caveats that belong beside the number, not in a footnote:

- **The turn budget moves it independently.** Giving the *base* model 12 turns instead of 8
  lifts it from 52 to 54 on its own. An 8-turn baseline against a 12-turn trained model would
  have manufactured two extra points of "learning". Both eval jobs therefore ship
  `EVAL_MAX_TURNS: '12'`.
- **n=200 means a standard error of roughly ±3.5 points**, and these are single runs, not
  multi-seed means. Treat the direction as real and the decimal as noise.

## 3. How we found what to move: EM = recall × conversion

This is the part worth copying. A single EM number tells you *whether* something changed; it
never tells you *what to try next*. So
[`analyze_traces.py`](usecases/agentic-search/analyze_traces.py) decomposes every eval trace
into two independent factors:

```
EM  =  recall                      ×  conversion
       did a retrieved passage        given the gold WAS retrieved,
       contain the gold answer?       did the model answer correctly?
```

- **recall** is the hard ceiling — the model cannot answer what it never retrieved.
- **conversion** is everything after that: reading, reasoning, committing the right span.

Measured on the traces, recall sat at **79–81%** and conversion was the gap. That one fact
set the whole agenda:

- Had **recall** been the bottleneck, the work would have been retrieval engineering — a
  better index, more hops, hybrid search, reranking.
- Because **conversion** was the bottleneck, the work was policy improvement on the
  reasoning-and-committing half — which is exactly what GRPO optimises.

It also explains the division of labour we ended up with: **more turns protected recall**
(more hops = more chances to surface the gold), while **GRPO improved conversion** (the model
got better at *using* what it had already retrieved). Without the decomposition we would have
been guessing across a hundred knobs.

## 4. Four levers, two of which did nothing

| lever | effect | kept? |
|---|---|---|
| **turn budget 8 → 12** (`MAX_TURNS`) | +2 EM on the base model, recall-safe | ✅ |
| **GRPO with the pure EM reward** | improved conversion → the trained delta | ✅ |
| retrieval-shaped reward (bonus for surfacing the gold) | **inert** — no measurable effect | ❌ |
| deeper run · denser groups (`rollout_n` 16 → 32) | flat; no late surge, same conversion | ❌ |

The dead lever is the instructive one, because it looks so reasonable on paper. The idea: give
partial credit when a retrieved passage contained the gold answer, so the model learns to
retrieve well even when it fumbles the final span. It changed nothing — and GRPO explains
why.

GRPO's advantage is computed **within a group** of `rollout_n` samples for the same prompt:

```
advantage = (reward − group_mean) / group_std
```

Recall was already ~80%, so the retrieval bonus fired on *nearly every sample in the group* —
and therefore landed in `group_mean` too, where it subtracts itself out. The gradient sees
almost nothing.

> **The transferable rule: a reward term only teaches if it *discriminates within the
> group*.** A bonus that almost always fires — or almost never does — is decoration. It is
> the same failure mode as a saturated dataset, one level down, and it is why
> [docs/tuning.md](docs/tuning.md) puts "measure reward variance" above every other knob.

The knob survives as `QA_RETRIEVAL_BONUS` (default `0.0`) so it can be re-tested on a corpus
where recall is genuinely low — the regime where it *should* help.

## 5. The ceiling, honestly

Both compute levers (a deeper run, denser groups) and the reward-shaping lever were ruled
out. That makes the ~57% plateau a **capability ceiling of pure-EM GRPO on this setup** — not
a training-dynamics artifact you can spend your way past.

Going materially higher would need a change of *signal or inference*, not more of the same
knob:

- **self-consistency at eval** — sample several trajectories and vote on the answer;
- **a recall-weighted advantage** — make the reward discriminate *within* the group on
  retrieval quality, which is the fix the flat bonus failed to be;
- **a harder-negative retrieval curriculum** — train where recall is actually contested.

None of these are implemented here. They are the honest "what next", not a roadmap.

## 6. What transfers to your task

Even if you never touch MuSiQue, five things from this run generalise:

1. **Run the baseline first, at final settings.** One node-hour, and it is the only thing
   that makes a trained number mean anything. It also exercises the whole eval path before
   you buy a multi-node training job. → [docs/running-jobs.md](docs/running-jobs.md) §4.3
2. **Match every eval setting between base and trained** — the turn budget especially. One
   job file, `EVAL_MODEL_PATH` swapped, nothing else.
   → [docs/configuration.md](docs/configuration.md) §13
3. **Make the reward and the eval scorer the same code**, so they cannot diverge.
   → [docs/new-usecase.md](docs/new-usecase.md) §2
4. **Build a decomposition, not just a metric.** Factor your number into independent stages
   and measure each; that is what tells you which knob to reach for.
   → [docs/tuning.md](docs/tuning.md), "the procedure"
5. **Evaluate several checkpoints.** The best held-out checkpoint here was step 20, with a
   plateau afterwards. The last checkpoint is not automatically the one you want, and
   training reward is not the deliverable.

One operational finding deserves its own line: the **rollout:trainer node ratio is not
learning-neutral.** A 2:1 split (2 rollout nodes, 1 trainer) raised effective staleness enough
to cost ~2–3 EM versus 1:1 at the same step count. More generation capacity than the trainer
can consume just produces staler samples — so if you scale a deep run, scale both sides.
→ [docs/training-modes.md](docs/training-modes.md) §4

## 7. The exact configuration

So this is reproducible rather than anecdotal. All of it lives in the job files; what each
setting does is in [docs/configuration.md](docs/configuration.md).

| | |
|---|---|
| mode / topology | `TRAIN_MODE=async`, `ROLLOUT_NNODES=1` (1:1 rollout:trainer), `STALENESS=0.1`, `TRIGGER_SYNC_STEP=1` |
| parallelism | `TP=2 EP=8 ETP=1 GEN_TP=8`, `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE=True`, prefix caching on |
| agent loop | `MULTI_TURN=True`, `MAX_TURNS=12`, `TOOL_FORMAT=qwen3_coder`, `MAX_TOOL_RESPONSE_LEN=4000` |
| GRPO | `rollout_n=16`, `ppo_mini_batch_size=32`, `actor_lr=2e-6`, `kl_loss_coef=0.01` (`low_var_kl`), `total_rollout_steps=3200` |
| reward | pure EM, `QA_RETRIEVAL_BONUS=0.0`, `REWARD_MANAGER=naive`, no judge |
| eval | `EVAL_LIMIT=200`, `EVAL_MAX_TURNS=12`, `EVAL_TEMPERATURE=0` — **identical for base and trained** |

Job files: [`4_train.yaml`](usecases/agentic-search/air/4_train.yaml) ·
[`3_baseline_eval.yaml`](usecases/agentic-search/air/3_baseline_eval.yaml) ·
[`5_eval.yaml`](usecases/agentic-search/air/5_eval.yaml). What the use case itself contains:
[`usecases/agentic-search/`](usecases/agentic-search).

## 8. Reproduce it

Prerequisites: image registered, Volume created, base model staged, and a Vector Search
endpoint to hold the index — [docs/running-jobs.md](docs/running-jobs.md) §1.

```bash
# 1. data + corpus, then the index. The index job RETURNS BEFORE the index is ready:
#    it keeps provisioning server-side, so wait for status.ready before evaluating.
air run --file usecases/agentic-search/air/1_prep_data.yaml   -p df1 --watch
air run --file usecases/agentic-search/air/2_build_index.yaml -p df1 --watch
databricks vector-search-indexes get-index main.mshtelma.wiki_qa_big_corpus_index \
  -p df1 --output json | python3 -c 'import json,sys; s=json.load(sys.stdin)["status"]; \
  print(s["ready"], s["indexed_row_count"])'

# 2. the baseline — the "before" number
air run --file usecases/agentic-search/air/3_baseline_eval.yaml -p df1 --watch

# 3. train
air run --file usecases/agentic-search/air/4_train.yaml -p df1 --watch

# 4. eval EACH saved checkpoint at the SAME settings (step 20 was best here, not the last)
air run --file usecases/agentic-search/air/5_eval.yaml -p df1 --watch \
  --override env_variables.MODEL_PATH=<…/global_step_20/actor/model/huggingface> \
            env_variables.EVAL_MODEL_PATH=<same> \
            env_variables.EVAL_OUT=/Volumes/main/mshtelma/verl/eval/trained_step20.json \
            env_variables.EVAL_TRACE_OUT=/Volumes/main/mshtelma/verl/eval/trained_step20_traces.jsonl

# 5. decompose, because the number alone will not tell you what to do next
python3 usecases/agentic-search/analyze_traces.py <base_traces.jsonl> <trained_traces.jsonl>
```

Expect run-to-run variation, and expect different absolute numbers on a different corpus.
What should reproduce is the *shape*: a matched-turn baseline, recall in the low 80s,
conversion as the bottleneck, and a flat retrieval bonus.

**→ Build the same thing for your own task: [docs/new-usecase.md](docs/new-usecase.md)**
