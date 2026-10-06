# OfficeQA pilot results

Data, isolated compute and the complete controller have passed the qualifications
linked in [README.md](README.md). The first baseline run
[5796847728497](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/5796847728497)
failed during judge calibration from commit `4fa7f71` with the qualified v12 trial
image. Nine of ten calibration calls passed; one concurrent cold tokenizer load
failed in Transformers' lazy import. No policy question was evaluated and no
training was started. The reward raised the abort channel; this is an infrastructure
failure, not a negative answer or evidence judgment.

The corrected calibration passed 10/10 on run
[867023020948845](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/867023020948845).
The compact baseline was then canceled after 44 episodes finished without a
terminal submission: 41 reached the 12-action limit and three stopped without a
tool call. All 44 had resolved zero rewards and no infrastructure errors. This
was an incomplete diagnostic run, not a benchmark result, and training was not
launched. Traces showed keyword-overlap retrieval returning irrelevant periods,
and a calculation cut off at the 1,024-token generation limit.

The revised pilot adds pinned BM25 retrieval, 16 actions, 4,096 tokens per
generation and 8,000-character tool observations. It uses a new immutable
16-action prompt snapshot and preserves the official question split and pilot
IDs. Its Qwen server uses CUDA graphs with custom all-reduce disabled; baseline,
training and checkpoint evaluation retain identical controller/reward budgets.

This file will record the terminal baseline, training and paired checkpoint
evaluation results. No accuracy or learning claim is made while those runs are
in progress.
