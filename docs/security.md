# Security model

The template assumes one team, its own Databricks workspace, and its own models and data. It is
not hardened for multi-tenant or internet-facing use.

## What is trusted

| what | how | if that doesn't hold |
|---|---|---|
| model repositories | every loader passes `trust_remote_code=True` (training, vLLM, eval tokenizers), which Qwen3.5 needs, so a model repo can run arbitrary Python on every node that loads it | stage only models you trust, at a pinned revision (`infra/air/stage_model.yaml` records the commit in `STAGED.json`) |
| datasets | pinned by revision (`engine/lib/data_manifest.py`), content-checked when a mirror stands in, no script-based loaders | pin any new source the same way |
| the image | base by digest, `docker/requirements.lock`, wheels by sha256 and sources by commit (`docker/artifacts.lock`), each pushed tag's digest in `docker/IMAGE.lock` | `make register` refuses a tag whose registry digest differs from the recorded one |
| the job's code | the `code_source` snapshot of your checkout runs on every node | review what you submit; `GIT_SHA` in manifests and eval artifacts names the commit (`-dirty` if it had uncommitted changes) |

## What is exposed

Every server listens on the job's private network without authentication: the eval's vLLM
server (`0.0.0.0:8000`), the LLM judge (`0.0.0.0:${JUDGE_PORT}`, its Ray on `${JUDGE_RAY_PORT}`
with the dashboard on `0.0.0.0`), and the training Ray cluster. That's fine inside one AI Runtime
job, whose nodes are isolated from other tenants. On a shared network, add an API key
(`vllm serve --api-key`) and bind Ray's dashboard to localhost.

Rendezvous and abort files live on the Volume under `RENDEZVOUS_ROOT/<RUN_ID>/`. Anyone who can
write there can stop a run (`ABORT.json`) or point its reward workers at another judge URL, so
limit write access on that path to the team that runs the jobs.

The Docker Hub credential for `air register` is kept in a Databricks secret scope
(`SECRET_SCOPE`/`SECRET_KEY` in `config.env`), as is the optional `HF_TOKEN`; neither is logged. A
package index URL with credentials reaches the build as a BuildKit secret, so it isn't in the
image or its history, and `make build` and `make doctor` print it masked.

## FIPS mode is off

AI Runtime hosts run a FIPS-enabled kernel, and several libraries in the stack bundle non-FIPS
crypto that aborts under enforced FIPS (`FATAL FIPS SELFTEST FAILURE`, `ssl.SSLError: [CRYPTO]
unknown error`). So `docker/Dockerfile` sets `OPENSSL_FORCE_FIPS_MODE=0` and `OPENSSL_FIPS=0`, as
the AI Runtime base image does, and the four stock-environment jobs set them in their YAML.
OpenSSL then doesn't restrict itself to FIPS-validated algorithms, so a deployment that must be
FIPS-compliant end to end can't run this stack as is.

## What is recorded

| record | contents | sensitivity |
|---|---|---|
| training logs (MLflow, `logs/`) | metrics, configs, stack traces; no prompts or completions by default (`rollout_data_dir` is unset) | low |
| `run_manifest.json`, `run_result.json`, `DATA_MANIFEST.json` | resolved config, identities, hashes | low |
| eval artifacts (`EVAL_OUT`) | per question: the question, gold answers, final answer, the last 300 to 400 characters of output, counts | dataset text and model output |
| eval traces (`EVAL_TRACE_OUT`) | every turn, tool call and retrieved passage | as sensitive as your corpus and prompts. With your own data, write them where only your team can read, or unset the variable |
| `JUDGE_DEBUG=1` | the first 600 characters of each verdict, in the reward workers' logs | off in the shipped jobs |

With your own data, treat the eval artifact and trace paths like the data itself.
