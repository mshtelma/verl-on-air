# OfficeQA from Miles

This use case ports the OfficeQA task at Miles commit
`24bdbc0235e0bb2c671b261311547129552279e9` to the `trial/verl-0.10` stack.
The actor is Qwen3.5-35B-A3B; a co-located GLM-5.3-Flash FP8 judge runs at TP8.
The source repository is unchanged. [SOURCE.json](SOURCE.json) records copied files,
hashes, the original table-aware chunker, and reward semantics.

The agent searches, greps and reads Treasury Bulletins, uses isolated Python for
arithmetic, and finishes with `submit_report(answer, path)`. The reward is binary:
the final value must match gold and the judge must confirm its cited evidence path.
Wrong answers skip the support judge. Scale, sign, percent and list-length handling
come from Miles' answer checker with zero numeric tolerance. Late-input dependencies
are counted without rejecting the report, following Miles; lookup figures still
must occur in delivered evidence even when the step declares dependencies.

The runtime assigns observation IDs after clipping and records delivery only when
the result reaches the next generation's committed context. Model-written markers
cannot create evidence. A complete final-turn submission is accepted before the
turn-limit check. The judge receives all cited evidence intact; an oversize request,
unknown verdict, scorer fault or service outage aborts the pilot. There is no reward
fallback. Ten live positive/negative support calibrations run before evaluation or
training.

Training and evaluation use the same `OfficeQAToolAgentLoop`, continuous-token
builder, tool registry and reward. Evaluation passes token IDs through vLLM's
`/completions` API and requires returned token IDs. It records model/data identities,
policy, closed per-episode traces, pass@1/pass@3 and within-group reward variation.
Evaluation has fixed per-episode sampling seeds; training uses verl's seeded sampler.

The official split is 113 easy + 90 hard training questions and 43 hard held-out
questions, ordered by the SHA256 of each hard question's UID. The bounded baseline
uses 16 training probes (8 easy, 8 hard) and the first 8 held-out questions, four
samples each. Pilot training selects only training probes with mixed binary rewards;
the held-out results never influence selection. This is a small integration pilot,
not a full benchmark run or Miles' adaptive curriculum.

| Setting | Pilot value |
|---|---|
| Initial prompt budget | 2,048 tokens; measured maximum 1,907 |
| Action limit | 12 assistant turns, one tool call per turn |
| Generation per turn | 1,024 tokens |
| Episode response budget | 34,816 tokens, including tool context |
| Tool result cap | 4,000 characters including observation marker |
| Judge context / output | 16,384 / 4,096 tokens |
| Baseline / checkpoint eval | 16 H100: actor TP8 + judge TP8; 60-minute limit |
| Training | 24 H100: 8 trainer + 8 rollout + 8 judge; 120-minute limit |
| Training budget | 8 prompt groups × 4 trajectories; two updates/syncs |
| Checkpoints | Both syncs saved; final is `global_step_2` |
| Retries | Zero job retries |

The image is
`michaelshtelma587/verl-megatron-air:v12-verl010-glmflash-fix1`, digest
`sha256:7bb2fbf97763e4aae1c79ed7f6c5515a25519c868bbbce002476c722d19f388a`.
It includes bubblewrap but lacks `bm25s`, so this pilot uses Miles' keyword-overlap
fallback. Every staging/evaluation artifact records the actual retrieval backend.
The full Miles GLM actor trainer, judge pool and 64-B300 topology are not ported.

The immutable dataset is
`/Volumes/main/mshtelma/verl/data/officeqa_miles/pilot-24bdbc023`.
Its prep receipt records `sandbox_probe: PENDING` because data staging preceded
the compute repair. Separate AIR qualifications then passed on the actual image:

| Qualification | df1 AIR run | Result |
|---|---|---|
| Official data / corpus / token lengths | [780370971048556](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/780370971048556) | 203 train, 43 held out, 697 bulletins, 131,113 chunks |
| Compute mounts and benign isolation controls | [541506187572680](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/541506187572680) | Arithmetic, NumPy/pandas, hidden data/proc/environment, isolated network passed |
| Full controller on real tokenizer/tools | [831623611146158](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/831623611146158) | 11 delivered results; forged marker ignored; turn-12 submit accepted; role spans retained |

AIR forbids a fresh proc mount. The port uses an empty `/proc` tmpfs inside a new
PID namespace, keeping Miles' masked data directories, cleared environment,
network isolation and resource bounds. `OQ_COMPUTE_ALLOW_UNSANDBOXED=0` is required
for the jobs, and each actor/rollout node performs a real compute/retrieval preflight.

Run with the explicitly selected `df1` profile:

```bash
# Use the staged immutable snapshot, or choose a new OQ_DATA_DIR for fresh prep.
make officeqa-toolcheck AIR_PROFILE=df1
make officeqa-baseline AIR_PROFILE=df1 BUDGET_OK=1
# The baseline prints its eval.json path; train refuses a baseline with no signal.
make officeqa-train AIR_PROFILE=df1 BUDGET_OK=1 BASELINE=/Volumes/.../eval.json
make officeqa-eval AIR_PROFILE=df1 BUDGET_OK=1 CKPT=/Volumes/.../global_step_2
```

Each submission uses a unique `RUN_ID` and commit identity. Training artifacts and
the selected-data manifest go under `OQ_ARTIFACT_ROOT/<RUN_ID>/`; checkpoints go
under `parameters.output_dir/<RUN_ID>/`. Completed pilot measurements are recorded
in [RESULTS.md](RESULTS.md).
