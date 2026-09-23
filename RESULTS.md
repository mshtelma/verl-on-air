# What one training run shows — and what it does not

Short answer: in one run, the trained agent scored a little higher than the base model on a
200-question development set, and the paired evidence is **suggestive, not conclusive**. This
page is that evidence, including what it cannot establish. It is an illustration that the loop
runs end to end on a real task — not a benchmark result.

← [README](README.md) · run it yourself: [docs/running-jobs.md](docs/running-jobs.md) §4 ·
evidence file: [`results/agentic-search/2026-09-dev-paired.json`](results/agentic-search/2026-09-dev-paired.json)

## The setup

[`usecases/agentic-search`](usecases/agentic-search) trains `Qwen3.5-35B-A3B` with GRPO to
answer MuSiQue questions as a **retrieval agent**: it searches and reads over a Databricks
Vector Search index, then commits a short span in `<answer>…</answer>`. The reward is
**rule-based exact match** against the dataset's gold answers — no LLM judge, no learned
reward model, no annotation beyond those gold answers.

- **Corpus:** 603,607 passages — the union of MuSiQue's and HotpotQA's own train + validation
  contexts, de-duplicated by content. The index every eval below used holds exactly the passage
  count that a rebuild at the pinned dataset revisions produces (checked 2026-09-23).
  2WikiMultihopQA is not included: its loader script cannot run under `datasets` ≥ 4.
  Validation-question passages are in the retrieval corpus — a curated, transductive setting,
  held fixed across every eval below.
- **Eval:** the training reward's own scorer (`score_segments` in `reward.py`), driven by the
  eval's own agent loop under a fixed policy: up to 12 turns, ≤512 tokens per request (+2
  continuations), a forced final answer on the last turn, greedy decoding. Base and
  checkpoints were scored under identical settings. These artifacts predate the recorded
  `eval_policy` (v2): they rendered hand-copied, shorter tool descriptions than training's, and
  their tool-call parser also accepted a call without `<tool_call>` tags. A v2 eval matches
  training on both, so it is a different measurement — `paired_eval.py` pairs only artifacts
  with equal policies.
- **The 200 questions** are the first 200 rows of the prepared MuSiQue validation split — which
  turn out to be **all 2-hop** — and every checkpoint below was scored on the same 200. A rebuild
  at the pinned MuSiQue revision reproduces all 200 questions and gold-answer lists exactly. Because
  those same questions were used to compare checkpoints and pick the best, this is a
  **development set**, not an untouched test set.

## The result — 12-turn eval, 200 development questions

| model | correct | EM | gained / lost vs base | exact McNemar p |
|---|---|---|---|---|
| base `Qwen3.5-35B-A3B` | 108 | 54.0% | — | — |
| pure-EM run, step 10 | 112 | 56.0% | +11 / −7 | 0.48 |
| **pure-EM run, step 20** | **117** | **58.5%** | **+13 / −4** | **0.049** |
| pure-EM run, step 30 | 113 | 56.5% | +13 / −8 | 0.38 |
| pure-EM run, step 40 | 114 | 57.0% | +10 / −4 | 0.18 |
| retrieval-bonus run, step 10 | 103 | 51.5% | +6 / −11 | 0.33 |
| retrieval-bonus run, step 20 | 106 | 53.0% | +11 / −13 | 0.84 |
| retrieval-bonus run, step 30 | 107 | 53.5% | +9 / −10 | 1.00 |
| longer run, step 10 | 102 | 51.0% | +6 / −12 | 0.24 |
| longer run, step 20 | 111 | 55.5% | +9 / −6 | 0.61 |
| longer run, step 30 | 112 | 56.0% | +14 / −10 | 0.54 |
| longer run, step 40 | 111 | 55.5% | +12 / −9 | 0.66 |
| longer run, step 50 | 111 | 55.5% | +13 / −10 | 0.68 |
| `rollout_n` 32 run, step 10 | 108 | 54.0% | +8 / −8 | 1.00 |

How to read it:

- **The headline pair.** 54.0% → 58.5% is 13 questions gained and 4 lost — 17 of 200 changed
  outcome. On its own that gives an exact McNemar p of 0.049 and a paired-bootstrap 95% CI for
  the gain of +0.5 to +8.5 points.
- **But it is the best of 13.** Step 20 of the pure-EM run was chosen by looking at this table.
  Accounting for that choice — a max-statistic sign-flip permutation test across all 13
  checkpoints — gives **p = 0.31**: a best-of-13 gain this large is well within what chance
  alone produces. The headline number is not statistically established.
- **What is suggestive.** All four checkpoints of the pure-EM run score at or above the base
  (+2 to +4.5 points), while all three of the retrieval-bonus run score below it. That pattern
  is worth a proper test; it is not one.
- **Single runs.** One training run per configuration, no repeated seeds: run-to-run variance
  is unmeasured.
- **Provenance.** These are the historical eval artifacts, produced before the eval contract
  (`engine/serve/eval_contract.py`) existed; none of the 14 contains a tool error or a
  swallowed inference error. The evidence file records each artifact's SHA-256 and the
  question-id digest; regenerate it with `scripts/paired_eval.py` (the per-question artifacts
  themselves live on the workspace Volume, not in this repository).

## What would settle it

1. **A held-out test split, used once.** Choose the checkpoint on a development split, then
   report it on questions no decision ever looked at — with the question ids published.
2. **Repeated seeds** of the training run, compared paired, question by question.
3. **More than 2-hop.** This development set is entirely 2-hop, so it says nothing yet about
   3- or 4-hop chains; report by hop count.
4. **Controls** that separate retrieval from recall: a no-tools (closed-book) eval, and
   supporting-passage coverage rather than answer-string matches.

## Other observations — single runs, none significant

| lever | observation (same 200 questions) |
|---|---|
| eval turn budget 8 → 12 | base 104 → 108 (+8 / −4, p = 0.39). The 8-turn artifact predates the eval contract and does not record its own settings |
| retrieval-bonus reward (`QA_RETRIEVAL_BONUS`) | 51.5–53.5%, all three checkpoints below the base |
| longer run | 51.0–56.0% across five checkpoints |
| `rollout_n` 16 → 32 | 54.0% at step 10, level with the base |

A **hypothesis** for the retrieval bonus, not a measurement: with the gold answer string
appearing in retrieved text for roughly 80% of questions, the bonus would often fire for most
samples of a GRPO group, shifting the group's mean rather than separating good rollouts from
bad ones. Within-group firing rates were not measured, so this is an explanation to test,
not a finding. `QA_RETRIEVAL_BONUS` stays available (default `0.0`).

## The EM decomposition diagnostic (a hypothesis generator)

[`analyze_traces.py`](usecases/agentic-search/analyze_traces.py) splits each eval trace by
whether a retrieved passage contained a gold answer string:

```
EM = P(retrieved) × P(correct | retrieved)  +  P(not retrieved) × P(correct | not retrieved)
```

"Retrieved" here is an **answer-string proxy** — the gold string appeared in some tool output
— not proof that the supporting passages of a multi-hop chain were found. On these runs the
proxy was 79–81%, which *suggested* that most misses happened after the answer had surfaced:
in how the model used what it found, which a policy update can act on. It does not show which
component limits EM — better queries and tool choice raise recall too — so treat it as a
hypothesis generator for the next experiment, not a causal account.

## Reproduce

Full walkthrough, including prerequisites: [docs/running-jobs.md](docs/running-jobs.md) §4.

```bash
make search-prep          # MuSiQue questions + passage corpus
make search-index WAREHOUSE_ID=<id>   # Vector Search index (returns early; wait until ready)
make search-baseline      # the "before" number — never skip this
make search-train         # GRPO, fully-async, 16xH100
make search-eval CKPT=<run>/global_step_20

# compare, question by question (every artifact must have scored the same questions)
python3 scripts/paired_eval.py <base.json> <ckpt_step10.json> <ckpt_step20.json> ...
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
