# Final OfficeQA experiment

Branch `experiment/verl010-officeqa-final` starts from the tested verl 0.10
OfficeQA port at `33ff660`. It uses Qwen3.5-35B-A3B and the existing v13 image:
PyTorch **2.13.0**, verl **0.10.0.dev0** at `8718ca30`, vLLM **0.30.0**, and
Transformers **5.16.1**. The image digest is
`sha256:fb2f959ad6c423040fe725e4ce8cb0babea7ba1be5ed3e808de8084710f4648d`.
Repository changes travel in the source snapshot; the image is reused unchanged.

The reference is `../miles` at `d984fc86b1`, specifically
`examples/glm5_3_flash/agent.py` and its `officeqa/` tools, official split and
path reward. The relevant vendored files still match that checkout's content;
`SOURCE.json` records their original port at `24bdbc023`. Miles' completed
official R3 trained on **113 easy + 90 hard = 203 questions** and reserved
**43 hard questions**, ordered by SHA256 of UID, for evaluation. Its audit
found no held-out evaluation in that launch, so this experiment explicitly
collects a frozen baseline and final evaluation.

The agent searches, greps, reads and lists Treasury Bulletins, uses isolated
Python for arithmetic, and ends with `submit_report(answer, path)`. Observation
IDs belong to the runtime and count as evidence only after delivery. Reward is
binary: a correct final answer with a supported cited path receives one.
Wrong answers skip the support judge. Service errors, invalid verdicts and
unresolved evidence assessments stop the run and invalidate evaluation.

This run permits **128 assistant turns**, one tool call per turn, as in Miles, with the same
**98,304-token context** already exercised by our earlier OfficeQA run. Each
generation is capped at 4,096 tokens; tool results are capped at 8,000
characters. The explicit episode budget decouples the action count from the
context allocation. Training and both evaluations use the same controller,
prompt, token limits, temperature, tools and reward.

Training starts from the original actor, using **all 203 training questions**.
It runs a fixed **100 updates**, comprising 400 prompt groups and 1,600
trajectories, with four samples per group and learning rate `1e-6`.
Questions are shuffled with seed 42. No baseline outcomes select training
questions, and held-out outcomes do not select the checkpoint or run length.
Checkpoints are saved every 25 updates; the final evaluation requires a
certified step-100 checkpoint and verifies the exact served identity.

The judge is the discovered **`system.ai.glm-5-3-flash`** service in **df1**,
called through Unity Gateway with refreshable ambient credentials, structured
verdicts, and low reasoning effort. Ten live supported/unsupported evidence
checks run before baseline and training. Judge calls have a 180-second request
timeout, two bounded retries and a 600-second overall deadline, within the
900-second reward timeout. The old continuation stopped during the step-80
save when a valid 26k-token judge request hit its 180-second overall deadline.
Local preflight also replayed that training-side request successfully: the API
resolved it as unsupported evidence, with no infrastructure error.

Training uses **16 H100s**: eight Megatron trainer/reference GPUs and eight
vLLM rollout GPUs. The tested trainer layout is TP2/CP4/EP8/DP1 with packed
sequences, fused LM head and CPU offload. Baseline and final evaluation each
use eight H100s. The workspace judge uses no GPUs from these jobs; its API
tokens are billable. The fixed timeouts cap the complete pipeline at about
641 GPU-hours, including preparation and artifact export; actual cost depends
on when the stages finish.

The immutable dataset path is
`/Volumes/main/mshtelma/verl/data/officeqa_miles/final-t128-c98304-20261011-v1`.
The preparation manifest hashes the source CSV, corpus, chunks and parquet
files and records disjoint train/held-out UIDs. It rejects exact question text
overlap. Gold answers and reference document hints are absent from policy
prompts. The corpus is shared across the splits, so the evaluation measures
new questions over the same Treasury Bulletin collection.

Baseline and final evaluation each score **236 episodes**: four samples for
every held-out hard question plus 16 fixed training probes. Report supported
pass@1, pass@3 and answer-only accuracy separately. The paired comparison
resamples questions for its 95% confidence interval, keeping each question's
four correlated samples together. Training reward alone is not evidence of
held-out improvement.

The original launch froze a clean, committed source at worker revision `9a54323`
with the controller interface available at that revision:

```bash
python usecases/officeqa/final_run.py prepare
python usecases/officeqa/final_run.py run
```

For a new experiment, use the [named-experiment Make commands](README.md#run-a-new-experiment).
The current controller requires a fresh experiment name, derives workspace paths
from the retargeted YAML and writes to `.cache/officeqa-final/<name>/`. Its `run`
and `status` commands can still read the original recorded directory with
`--work-dir .cache/officeqa-final/run`. The paths and receipts below describe
the completed 11 October experiment.

The controller uses `df1`, stable idempotency keys, exact AIR dry-runs and a
frozen Git archive. It records stage receipts and runs preparation, baseline,
training, verified training-artifact export and final evaluation in sequence.
Logs, certificates and the paired result are under `.cache/officeqa-final/run/`.
It stops on a failed stage and can resume monitoring the existing receipt
without creating another run. An AIR training failure can proceed to evaluation
only after an explicit completion review verifies the original clean final
certificate, every planned training trajectory, both nodes' clean exits, and
the platform's generic error without an additional error trace. The review
pins those evidence files by SHA256 and preserves AIR's `FAILED` status.
Durable remote artifacts are under
`/Volumes/main/mshtelma/verl/eval/officeqa-final/`, with checkpoints under
`/Volumes/main/mshtelma/verl/ckpt/qwen3_5-35b-officeqa-final/`.

The baseline completed on 11 October in AIR run
[679618243056695](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/679618243056695),
scoring all 236 episodes with no infrastructure failures. On the 43 held-out
hard questions, supported pass@1 is **8.72%**, pass@3 is **19.77%**, and
answer-only accuracy is **19.19%**. The 16 training probes have supported
pass@1 of **29.69%**, with mixed rewards on 11 questions. The
[baseline receipt and question samples](../../results/officeqa/2026-10-11-final-baseline.json)
preserve the measurement before training.

Fresh training completed all 100 updates in AIR run
[612185663086483](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/612185663086483).
The worker source remains frozen at `9a54323`; subsequent commits record
receipts and controller fixes. The original certificate verifies step 100,
all 14 weight shards, raw/final exits 0, no abort and no hard-error matches.
Both node commands exited 0, but AIR reported `FAILED` with
`CLIENT_ERROR / RUN_EXECUTION_ERROR` and no detailed error trace. The platform
cause remains unresolved. A completion review and independent trajectory
audit allowed the frozen evaluation to continue without rerunning training.
Training took about **9.5 hours**, with 400 complete groups and 1,600 unique
trajectories. The audit found all 203 training questions, including all 90 hard
questions, zero held-out questions and zero infrastructure rewards.

The final evaluation completed in AIR run
[623602877521600](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/623602877521600)
with `SUCCESS`: all 236 predetermined episodes scored, no infrastructure
errors, identical baseline policy/data/source, and served checkpoint identity
`6d6eb3c6134cc516` matching the original step-100 certificate. Both raw
236-episode files passed the question, score, observation-ID and delivery audit.

| Held-out metric | Original | Step 100 | Change | Paired 95% interval |
|---|---:|---:|---:|---:|
| Supported pass@1 | 8.72% | 20.93% | +12.21 pp | [+4.65, +20.35] pp |
| Supported pass@3 | 19.77% | 30.23% | +10.47 pp | [0.00, +21.51] pp |
| Answer accuracy | 19.19% | 26.74% | +7.56 pp | [+1.74, +13.95] pp |

The primary learning criterion passed: supported held-out pass@1 has a
positive lower confidence bound. Supported successes increased from 15/172
to 36/172, with 11 questions improving and two regressing. The pass@3 interval
includes zero. Training-probe supported pass@1 increased from 29.69% to 50.00%.
Correct held-out answers with invalid references fell from 13 to zero; ten
correct final answers still had unsupported paths. This is one seeded run on
43 new questions over the shared corpus.

The [final report and figures](../../results/officeqa/2026-10-11-final-report.md)
link the paired measurements, independent audit and raw-trajectory findings.
Cases were selected after the fixed evaluation for illustration, including a
Zipf-exponent question improving from 0/4 to 3/4 supported samples and an OCS
rents-and-royalties question regressing from 1/4 to 0/4. They did not select
the checkpoint or the reported question set. The controller's 11 tests and
`make lint` passed.

A local preflight stress check prepared eight concurrent 127-observation
evidence histories in 58.21 seconds, retaining all cited evidence and producing
identical request hashes; no scorer change was needed.
