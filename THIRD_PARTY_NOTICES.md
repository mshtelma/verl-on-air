# Third-party notices

verl-on-air is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). It adapts, calls or
depends on the work below. This file records attribution. It is not legal advice and does not
establish what any model or dataset may be used for: check each licence yourself before you
train on, evaluate with or redistribute anything.

## Code adapted into this repository

| where | what | origin | licence |
|---|---|---|---|
| `usecases/agentic-search/reward.py` (`normalize_answer`, EM / cover-EM / F1, last-`<answer>` extraction) | SQuAD-style answer normalisation and exact-match scoring, and the Search-R1 outcome-reward shape | the official SQuAD v1.1 evaluation script (Rajpurkar et al., 2016), as used by HotpotQA's evaluation and by Search-R1 (Jin et al., 2025; github.com/PeterGriffinJin/Search-R1) | see the SQuAD release (rajpurkar.github.io/SQuAD-explorer) and the Search-R1 repository |
| `engine/train/run_grpo_fully_async.sh` | the launcher's structure | verl v0.9.0 `verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh` | Apache-2.0 |
| `engine/train/run_grpo_megatron.sh`, `infra/geo3k/` | GRPO/geo3k recipe settings | verl v0.9.0 examples (`examples/grpo_trainer/`) | Apache-2.0 |
| `docker/Dockerfile` | the version set | verl v0.9.0 `docker/Dockerfile.stable.vllm` (deviations listed in the file) | Apache-2.0 |
| `usecases/math/grading.py` (`last_boxed`) | last-`\boxed{}` extraction, as in the MATH benchmark's evaluation | Hendrycks et al., 2021 (github.com/hendrycks/math) | MIT |

## Called, not copied

| what | used for | licence |
|---|---|---|
| verl (v0.9.0, commit 483b8a0), incl. `verl.utils.reward_score.prime_math` | the RL framework; the math grader | Apache-2.0 |
| vLLM, Megatron-Core, Megatron-Bridge, mbridge, TransformerEngine, apex, flash-attn, flash-linear-attention, PyTorch, Ray, Transformers | the training/inference stack in the image (`docker/requirements.lock`, `docker/artifacts.lock`) | each project's own (Apache-2.0 / BSD-3-Clause / MIT) |
| Databricks AI Runtime base image `databricksruntime/air` | the image base | Databricks' terms |

## Models (not redistributed here)

| model | used as | where to read its licence |
|---|---|---|
| Qwen3.5-35B-A3B (and -2B / -9B for the ladder) | the policy | its Hugging Face model card (Qwen) |
| GLM-5.3 | the math LLM judge | its Hugging Face model card (Z.ai) |
| `databricks-gte-large-en` | Vector Search embeddings | Databricks' model terms |

## Datasets (downloaded at a pinned revision; not redistributed)

| dataset | used for | where to read its licence |
|---|---|---|
| MuSiQue (`dgslibisey/MuSiQue`) | search training, dev/test questions, corpus | the MuSiQue release (Trivedi et al., 2022) |
| HotpotQA (`hotpotqa/hotpot_qa`, distractor) | corpus passages | the HotpotQA release (CC BY-SA 4.0) |
| MATH (Hendrycks et al.; `EleutherAI/hendrycks_math`), MATH-500 | math training / eval | the MATH release (MIT) and the MATH-500 subset's card |
| AIME / HMMT via MathArena | optional math eval sets | the MathArena dataset cards |
| geometry3k (`hiyouga/geometry3k`) | the scaling ladder | its dataset card |

The committed question-ID lists (`usecases/agentic-search/splits/`) are identifiers only.
