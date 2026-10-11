# Final OfficeQA experiment

Branch `experiment/verl010-officeqa-final` starts from the tested verl 0.10
OfficeQA port at `33ff660`. It uses Qwen3.5-35B-A3B and the existing v13 image:
PyTorch **2.13.0**, verl **0.10.0.dev** at `8718ca30`, vLLM **0.30.0**, and
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

This run permits **128 assistant/tool steps**, as in Miles, with the same
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
save when a valid 26k-token judge request hit its 180-second overall deadline;
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

After committing the source, run:

```bash
python usecases/officeqa/final_run.py prepare
python usecases/officeqa/final_run.py run
```

The controller uses `df1`, stable idempotency keys, exact AIR dry-runs and a
frozen Git archive. It records stage receipts and runs preparation, baseline,
training, verified training-artifact export and final evaluation in sequence.
Logs, certificates and the paired result are under `.cache/officeqa-final/run/`.
It stops on a failed stage and can resume monitoring the existing receipt
without creating another run. Durable remote artifacts are under
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

Fresh 100-update training started in AIR run
[612185663086483](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/612185663086483).
The worker source remains frozen at `9a54323`; subsequent commits record
receipts and controller fixes. Training and final evaluation are pending, so
this baseline does not demonstrate learning. A local stress check prepared
eight concurrent 127-observation evidence histories in 58.21 seconds, retaining
all cited evidence and producing identical request hashes; no scorer change
was needed.
