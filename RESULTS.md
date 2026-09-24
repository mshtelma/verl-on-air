# Results

One training run of the agentic-search use case, scored on a development set and then on a
held-out test set. The trained agent beat the base model by 4.5 points on the 200 development
questions used to pick it. On 500 held-out questions, scored once, the gain was 2.8 points and not
statistically significant (p = 0.15). Read it as a demonstration that the loop trains on a real
task, not as a benchmark result.

Evidence files: [`results/agentic-search/2026-09-dev-paired.json`](results/agentic-search/2026-09-dev-paired.json)
(dev) and [`results/agentic-search/2026-09-heldout-test.json`](results/agentic-search/2026-09-heldout-test.json)
(test). To run it yourself, see [docs/running-jobs.md](docs/running-jobs.md).

## Setup

[`usecases/agentic-search`](usecases/agentic-search) trains `Qwen3.5-35B-A3B` with GRPO to answer
MuSiQue questions as a retrieval agent. It searches and reads over a Databricks Vector Search
index, then gives a short answer in `<answer>…</answer>`. The reward is exact match against the
dataset's gold answers.

- Corpus: 603,607 passages, the union of MuSiQue's and HotpotQA's train and validation contexts,
  de-duplicated. The validation questions' passages are in the corpus, so the setting is
  transductive. It is the same for every eval below.
- Eval: the reward's own scorer (`score_segments` in `reward.py`), driven by the eval's agent
  loop: up to 12 turns, at most 512 tokens per request with 2 continuations, a forced final
  answer on the last turn, greedy decoding. Base and checkpoints use identical settings.
- Development set: the first 200 rows of the prepared MuSiQue validation split, which are all
  2-hop. Every checkpoint below was scored on them and the best was picked from them, so they
  are a development set, not a test set.

The dev-set artifacts come from an earlier version of the eval (shorter tool descriptions, a more
lenient tool-call parser). They are only compared with each other; `paired_eval.py` refuses to
pair artifacts whose eval policies differ.

## Development set: 200 questions

| model | correct | EM | gained / lost vs base | exact McNemar p |
|---|---|---|---|---|
| base `Qwen3.5-35B-A3B` | 108 | 54.0% | | |
| pure-EM run, step 10 | 112 | 56.0% | +11 / −7 | 0.48 |
| pure-EM run, step 20 | 117 | 58.5% | +13 / −4 | 0.049 |
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

Step 20 of the pure-EM run was the best of these 13 checkpoints. Only 17 of the 200 questions
changed outcome: 13 gained and 4 lost, an exact McNemar p of 0.049 on its own, and a
paired-bootstrap 95% interval for the gain of +0.5 to +8.5 points. But it was chosen by looking at
this table. A max-statistic sign-flip permutation test over all 13 checkpoints, which accounts for
that choice, gives p = 0.31.

All four pure-EM checkpoints score at or above the base (+2 to +4.5 points), and all three
retrieval-bonus checkpoints score below it. That pattern deserves a proper test; on its own it is
not evidence. Each configuration was trained once, so the table says nothing about run-to-run
variance.

None of the 14 artifacts has a tool error or an inference error. The evidence file records each
artifact's SHA-256; `scripts/paired_eval.py` regenerates it from the artifacts on the Volume.

## Held-out test: 500 questions, scored once

Since the dev set picked the checkpoint, we fixed a test split afterwards and scored each model
on it once. It holds 500 MuSiQue validation questions from row 500 onward, past anything an
earlier eval touched, stratified by hop count: 196 2-hop, 198 3-hop and 106 4-hop. The IDs and
the rule that drew them (`make_splits.py`, seed 20260923) are in
[`usecases/agentic-search/splits/`](usecases/agentic-search/splits/). All runs below used the
current eval (`eval_policy` v2) and are valid, with no infrastructure errors.

| model | correct | EM | gained / lost vs base | exact McNemar p | 95% CI of the gain |
|---|---|---|---|---|---|
| base, with tools | 174 | 34.8% | | | |
| pure-EM step 20 (chosen on dev), with tools | 188 | 37.6% | +47 / −33 | 0.146 | −0.6 to +6.4 pts |
| replicate step 20 (seed 7, fixed in advance), with tools | running | | | | |
| base, closed-book | 22 | 4.4% | | | |
| pure-EM step 20, closed-book | 26 | 5.2% | +8 / −4 | 0.388 | −0.6 to +2.2 pts |
| replicate step 20, closed-book | running | | | | |

| hops | n | base | pure-EM step 20 |
|---|---|---|---|
| 2 | 196 | 42.9% | 48.0% |
| 3 | 198 | 33.8% | 32.8% |
| 4 | 106 | 21.7% | 27.4% |

The chosen checkpoint still beats the base, by 2.8 points, but the 95% interval of the gain
includes zero. A smaller gain than on dev is expected, since the dev number was the best of 13 on
the same questions.

Both models score lower here than on dev. The test adds 3- and 4-hop questions, which are
harder, and its 2-hop questions come from later rows of a file that is not in random order (the
base scores 42.9% on them against 54.0% on dev). Compare the gains between the two sets, not the
levels.

Without tools the base scores 4.4% and the trained model 5.2% (p = 0.39). Nearly all of the score
comes from retrieval, and training did not measurably change what the model answers from memory.

The gain is in the 2- and 4-hop questions, with 3-hop flat. With 100 to 200 questions per group,
that is too few to read as a pattern.

The trained model answers more often (493 vs 474 of 500) and makes fewer tool calls (7.1 vs 8.2
per question). That fits the model learning when to stop and answer, which EM rewards, but it is
an observation, not a demonstrated mechanism.

The replicate is a second run of the same configuration with seed 7. Its step 20 was named as the
checkpoint to test before it trained. It is still training; its rows will be filled in once it has
been scored.

## GRPO signal

GRPO learns only from groups whose rewards differ. A group of samples that are all right or all
wrong has zero advantage. We sampled the base model 8 times on each of 64 dev questions at T = 1.0
(`EVAL_N_SAMPLES=8`): 31% of the groups were mixed (95% CI 21–43%), 38% all correct and 31% all
wrong. Training uses 16 samples per group, which can only raise the mixed share. The same probe on
geo3k (5 samples) gives 30% (CI 20–42%).

## What would settle it

The held-out split, the closed-book control, the per-hop breakdown and the variance probe were
added after the first version of this page. Still missing:

1. More seeds. One replicate is not a variance estimate. A claim about the configuration needs
   several runs, compared paired on the test split.
2. Supporting-passage coverage instead of answer-string matches
   (`analyze_traces.py --supporting-from-musique`), to separate finding the evidence from using it.
3. A larger test split. At +2.8 points with 16% of questions changing outcome, 500 paired questions
   give about a one-in-three chance of p < 0.05; about 1,600 give 80%. The unused validation pool
   has 1,917 questions, so this needs no new data.

## Other observations

Single runs on the same 200 dev questions. None is significant.

| change | result |
|---|---|
| eval turn budget 8 → 12 | base 104 → 108 (+8 / −4, p = 0.39); the 8-turn artifact does not record its settings |
| retrieval bonus in the reward (`QA_RETRIEVAL_BONUS`) | 51.5–53.5%, all three checkpoints below the base |
| longer run | 51.0–56.0% across five checkpoints |
| `rollout_n` 16 → 32 | 54.0% at step 10, level with the base |

A possible reason the retrieval bonus did not help, not measured: the gold answer string shows up
in retrieved text for about 80% of questions, so the bonus would fire for most samples in a group
and shift the group mean instead of separating good rollouts from bad ones. `QA_RETRIEVAL_BONUS`
is still available (default `0.0`).

## EM decomposition

[`analyze_traces.py`](usecases/agentic-search/analyze_traces.py) splits EM by whether a retrieved
passage contained a gold answer string:

```
EM = P(retrieved) × P(correct | retrieved) + P(not retrieved) × P(correct | not retrieved)
```

"Retrieved" is a proxy: the gold string appeared in some tool output, which doesn't mean the
supporting passages of the chain were found. On these runs it was 79–81%, which suggested that
most misses happen after the answer has appeared, in how the model uses what it found. It doesn't
show which part limits EM (better queries raise recall too), so use it to pick the next
experiment.

## Reproduce

```bash
make search-prep                        # MuSiQue questions and passage corpus
make search-index WAREHOUSE_ID=<id>     # Vector Search index (returns early; wait until ready)
make search-baseline                    # the base model's score
make search-train BUDGET_OK=1           # GRPO, fully-async, 16xH100
make search-eval CKPT=<run>/global_step_20

# compare question by question (every artifact must have scored the same questions)
python3 scripts/paired_eval.py <base.json> <ckpt_step10.json> <ckpt_step20.json> ...
python3 usecases/agentic-search/analyze_traces.py <base_traces.jsonl> <trained_traces.jsonl>
```

The settings are in [`4_train.yaml`](usecases/agentic-search/air/4_train.yaml),
[`3_baseline_eval.yaml`](usecases/agentic-search/air/3_baseline_eval.yaml) and
[`5_eval.yaml`](usecases/agentic-search/air/5_eval.yaml). The main ones: `MAX_TURNS=12`,
`rollout_n=16`, `actor_lr=2e-6`, fully-async with one rollout node and one trainer node, `EP=8`,
`GEN_TP=8`, and the pure exact-match reward.
