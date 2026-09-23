# Deploying a trained checkpoint — NOT IMPLEMENTED

← [verl-on-air](../README.md) · [running-jobs](running-jobs.md)

**Status: not implemented, not tested.** This repo trains and evaluates; it does not ship a
deployment. What follows is the shape of the work, so you can plan it — none of it has been
run end to end, and no command here is part of the tested path. (`make deploy-recipe` prints
this page; it used to be an 8-GPU job that only printed text.)

An *agent* deployment is two things, and both are yours to build:

1. **a model endpoint** serving the checkpoint's HF export
   (`<output_dir>/<RUN_ID>/global_step_N/actor/model/huggingface/`), and
2. **the agent loop** in your application: the prompt the model was trained on
   (`usecases/agentic-search/prep_data.py`'s `SYSTEM_PROMPT`), the `qwen3_coder` tool-call
   format, the three tools of `usecases/agentic-search/tool.py` against your Vector Search
   index, and the turn budget. `usecases/agentic-search/eval.py` is a working reference for
   that loop against an OpenAI-compatible `/v1/completions` endpoint — it is what produced
   every number in [RESULTS.md](../RESULTS.md).

## Path A — Databricks Model Serving (untested for this architecture)

`mlflow.register_model()` on a raw HF directory does **not** work: registration needs an
MLflow model artifact. The steps are roughly:

1. Log the HF export as an MLflow model with the `transformers` flavor (a task, a signature,
   and the tokenizer — Qwen3.5 needs `trust_remote_code`), then register it in Unity Catalog.
2. Check that the architecture (`Qwen3_5MoeForConditionalGeneration`, 35B total / 3B active)
   is eligible for provisioned throughput in your workspace — eligibility is per model
   family, and a custom MoE may not be.
3. Create the serving endpoint; call it from your agent loop with the endpoint's
   OpenAI-compatible route and a workspace token.

## Path B — a self-hosted vLLM server (a demo, not a deployment)

`engine/serve/serve_and_eval.sh` already brings a checkpoint up with vLLM (TP=8, one
`GPU_8xH100` node) for the evals. The same server could be left running in a job — but a job
has **no ingress**: nothing outside the job's private network can reach it, there is no
authentication (see [security.md](security.md)), and it stops at the job's timeout. Use it to
inspect a model interactively from inside a job; do not call it a deployment.

## What "done" would mean

A tool-using client outside the job, calling an authenticated endpoint, reproducing the eval's
EM on a sample of the test split, with the endpoint and index cleaned up afterwards. None of
that exists here yet.
