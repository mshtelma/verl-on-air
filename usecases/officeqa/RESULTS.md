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

The v13 image passed its exact dependency-lock check and a real BM25 ranking
probe, then was pushed and registered on `df1`; its digest is in
[IMAGE.lock](../../docker/IMAGE.lock). Revised preparation
[285706995986882](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/285706995986882)
passed isolated arithmetic and produced a maximum prompt length of 1,907 tokens.
Qualification
[720095605769482](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/720095605769482)
verified unchanged Miles source hashes, corpus/chunk hashes, official split and
pilot question IDs. It indexed 697 bulletins / 131,113 chunks with BM25, passed
compute staging, and completed the real continuous-token controller with 16
requests, 15 delivered observations, a final-turn submission and 31 role spans.

The revised 96-episode baseline is
[747485359876719](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/747485359876719),
submitted from `92dce54` with logical ID
`officeqa-baseline3-20261006T022752Z-92dce54`. Terminal baseline, training and
paired checkpoint measurements will be recorded here. No accuracy or learning
claim is made while those runs are in progress.
