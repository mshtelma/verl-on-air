# OfficeQA pilot results

Data, isolated compute and the complete controller have passed the qualifications
linked in [README.md](README.md). The first baseline run
[5796847728497](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/5796847728497)
failed during judge calibration from commit `4fa7f71` with the qualified v12 trial
image. Nine of ten calibration calls passed; one concurrent cold tokenizer load
failed in Transformers' lazy import. No policy question was evaluated and no
training was started. The reward raised the abort channel; this is an infrastructure
failure, not a negative answer or evidence judgment.

This file will record the terminal baseline, training and paired checkpoint
evaluation results. No accuracy or learning claim is made while those runs are
in progress.
