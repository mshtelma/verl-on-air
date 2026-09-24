# Deploying a trained checkpoint (not implemented)

This repo trains and evaluates; it has no deployment, and nothing here has been run end to end.
The page describes the work so you can plan it (`make deploy-recipe` prints it).

An agent deployment has two parts, both yours to build:

1. A model endpoint serving the checkpoint's Hugging Face export
   (`<output_dir>/<RUN_ID>/global_step_N/actor/model/huggingface/`).
2. The agent loop in your application: the training prompt (`SYSTEM_PROMPT` in
   `usecases/agentic-search/prep_data.py`), the `qwen3_coder` tool-call format, the tools from
   `usecases/agentic-search/tool.py` pointed at your index, and the turn budget.
   `usecases/agentic-search/eval.py` implements that loop against an OpenAI-compatible
   `/v1/completions` endpoint.

## Option A: Databricks Model Serving (untested for this architecture)

1. Log the HF export as an MLflow model with the `transformers` flavor (task, signature,
   tokenizer; Qwen3.5 needs `trust_remote_code`) and register it in Unity Catalog.
   `mlflow.register_model()` won't take a raw HF directory.
2. Check that `Qwen3_5MoeForConditionalGeneration` (35B total, 3B active) is eligible for
   provisioned throughput in your workspace. Eligibility is per model family.
3. Create the endpoint and call its OpenAI-compatible route from your agent loop.

## Option B: a vLLM server inside a job

`engine/serve/serve_and_eval.sh` already serves a checkpoint on one `GPU_8xH100` node. You could
leave it running, but a job has no ingress, the server has no authentication
([security.md](security.md)), and it stops at the job's timeout. It's a way to try a model from
inside a job, not a deployment.
