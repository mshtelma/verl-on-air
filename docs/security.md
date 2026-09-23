# Security model

What this template trusts, what it exposes, and what it records. None of it is hardened for a
multi-tenant or internet-facing setting: it assumes **one team, its own Databricks workspace,
its own models and data**. Read this before pointing it at anything else.

## What is trusted

| thing | how it is trusted | if that does not hold |
|---|---|---|
| **model repositories** | every loader passes `trust_remote_code=True` (the training config, vLLM, the evals' tokenizers). Qwen3.5 needs it for its architecture code. A model repo can therefore run arbitrary Python on every node that loads it. | stage only models you trust, from a pinned revision: `infra/air/stage_model.yaml` resolves one commit and records it in `STAGED.json`; `verify_checkpoint.py` checks the files, not their intent |
| **datasets** | pinned by revision (`engine/lib/data_manifest.py`), content-checked when a mirror is used. Script-based / `trust_remote_code` dataset loaders are not used. | a new data source goes through the same `Source(..., revision)` pin |
| **the image** | the base by digest, every package by `docker/requirements.lock`, every wheel by sha256 and every source by commit (`docker/artifacts.lock`), each pushed tag's digest in `docker/IMAGE.lock` ([build-linux.md](build-linux.md)) | `make register` refuses a tag whose registry digest is not the recorded one |
| **the job's own code** | shipped as a `code_source` snapshot of your checkout -- whatever is in the tree at submit time runs, on every node | review what you submit; `GIT_SHA` in every manifest and eval artifact says which commit it was (`-dirty` if it had uncommitted changes) |

## What is exposed, and to whom

Everything listens on the job's **private network** only, with **no authentication**:

| endpoint | where | who can reach it |
|---|---|---|
| vLLM OpenAI server (evals, `serve_and_eval.sh`) | `0.0.0.0:8000` on the eval node | processes on that node / the job's network |
| the LLM judge (`serve_judge.sh`) | `0.0.0.0:${JUDGE_PORT}` on the judge head; Ray on `${JUDGE_RAY_PORT}` with its dashboard on `0.0.0.0` | the training nodes of the same job -- by design, over the rendezvous file |
| the training Ray cluster | the head's Ray port + dashboard | the job's nodes |

That is acceptable inside one AI Runtime job, whose nodes are isolated from other tenants. Do not
reuse these scripts on a shared network without adding an API key (`vllm serve --api-key`) and
binding Ray's dashboard to localhost.

**Rendezvous and abort files** live on the UC Volume (`RENDEZVOUS_ROOT/<RUN_ID>/`). Anyone with write
access to that Volume path can stop a run (`ABORT.json`) or point its reward workers at another
judge URL. Keep the Volume's write grants to the team that runs the jobs.

**Secrets**: the Docker Hub credential for `air register` lives in a Databricks secret scope
(`SECRET_SCOPE`/`SECRET_KEY` in `config.env`); an optional `HF_TOKEN` likewise. Neither is written
to logs or artifacts. The build's package index URL may carry credentials: it reaches the build as
a BuildKit secret, never a build arg, so it is not in the image or its history, and `make build`
and `make doctor` print it with the userinfo masked.

## FIPS mode is turned off

AI Runtime hosts run a FIPS-enabled kernel. Several libraries in the stack bundle non-FIPS
crypto, which aborts at SSL initialisation under enforced FIPS mode (`FATAL FIPS SELFTEST FAILURE`,
`ssl.SSLError: [CRYPTO] unknown error`). The image therefore sets `OPENSSL_FORCE_FIPS_MODE=0` and
`OPENSSL_FIPS=0` (one place: `docker/Dockerfile`'s `ENV`; the four stock-environment jobs set them
in their YAML), as the AI Runtime base image itself does.

**The trade-off:** OpenSSL in these processes does not restrict itself to FIPS-validated
algorithms. TLS to Databricks, Hugging Face and Docker Hub still uses the platform's certificates
and modern ciphers, but a deployment that must be FIPS-compliant end to end cannot run this stack
as is. Making it so would need FIPS builds of every crypto-bundling dependency (opencv, the
Python `ssl` module's OpenSSL, ...), which this template does not attempt.

## What is recorded

| record | contains | sensitivity |
|---|---|---|
| training logs (MLflow, `logs/`) | metrics, configs, stack traces; **no** prompts or completions by default (verl's `rollout_data_dir` is unset) | low |
| `run_manifest.json`, `run_result.json`, `DATA_MANIFEST.json` | resolved config, identities, hashes | low |
| eval artifacts (`EVAL_OUT`) | per question: the question, gold answers, the model's final answer, the last 300-400 characters of its output, counts | contains dataset text and model output |
| eval traces (`EVAL_TRACE_OUT`, optional) | **full trajectories**: every turn, every tool call and every retrieved passage | **sensitive** -- as sensitive as the corpus and the prompts. The shipped evals write it (`analyze_traces.py` reads it); with your own data, point it at a path only your team can read, or unset it |
| `JUDGE_DEBUG=1` | the first 600 characters of each judge verdict, in the reward workers' logs | debugging only; off in the shipped jobs |

With public benchmarks (MuSiQue, HotpotQA, MATH) none of this is private. With your own data, the
eval artifacts and traces carry it: treat their Volume paths like the data itself.
