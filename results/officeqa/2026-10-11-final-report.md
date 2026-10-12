# OfficeQA: 100-update results

Supported pass@1 on held-out questions improved from **8.72% to 20.93%** after
100 training updates. The gain is **+12.21 percentage points**, with a paired
95% confidence interval of **+4.65 to +20.35**.

The experiment splits OfficeQA Full into **113 easy +
90 hard = 203 training questions**, with **43 hard questions held out**.
The creators' headline OfficeQA Pro benchmark contains all 133 hard questions.
Our scores describe this 43-question subset under the policy below.

## Held-out results

| Metric | Original | Step 100 | Change | Paired 95% interval |
|---|---:|---:|---:|---:|
| Supported pass@1 | 8.72% | 20.93% | +12.21 pp | [+4.65, +20.35] pp |
| Supported pass@3 | 19.77% | 30.23% | +10.47 pp | [0.00, +21.51] pp |
| Answer accuracy | 19.19% | 26.74% | +7.56 pp | [+1.74, +13.95] pp |

Supported successes increased from **15/172 to 36/172**. Eleven questions
improved, two regressed and 30 were unchanged. Each held-out question has four
samples per model; confidence intervals resample questions with their samples
kept together. The pass@1 interval excludes zero. The pass@3 interval includes
zero, so its increase remains uncertain.

The reward requires both a correct answer and an accepted evidence path. This
is an added requirement beyond the creators' answer-correctness score, and the
answer checker differs from the creators' scorer. Answer accuracy counts correct
final answers in well-formed reports, regardless of citation validity or the
support verdict. Malformed reports count as incorrect. Our tools, prompt and
sampling policy also differ from the paper's agent evaluations.

![Held-out comparison](2026-10-11-final-heldout.png)

## Training and evaluation

Training ran for the predetermined **100 updates**, comprising **400 groups**
and **1,600 trajectories**. It used all 203 training questions, with no held-out
questions in training. No baseline or held-out result selected the questions,
run length or final checkpoint.

Both evaluations scored all **236 planned episodes** without infrastructure
errors: four samples for each held-out question and each of 16 fixed training
probes. The probes changed from 29.69% to 50.00% supported pass@1. During
training, 700 of 1,600 trajectories earned supported rewards and 195 of 400
groups had mixed rewards.

The actor is **Qwen3.5-35B-A3B**, initialized from its original weights. Training
used 16 H100s for about **9.5 hours**; each evaluation used eight H100s for about
an hour. The runtime was PyTorch 2.13.0, verl 0.10.0.dev0, vLLM 0.30.0,
Transformers 5.16.1 and CUDA 13. A hosted GLM-5.3-Flash judge used no GPUs from
these allocations. Hosted inference tokens are charged separately.

The agent searches, greps, reads and lists the Treasury Bulletins and uses
isolated Python for calculations. It finishes with `submit_report(answer,
path)`. Its policy allows 128 turns, one tool call per turn, 4,096 generated
tokens per turn, 8,000-character tool responses and a 98,304-token context.
The learning rate is `1e-6` and the training seed is 42. Training and evaluation
share the agent loop and scorer. The current tools do not include web search
or visual inputs.

![Training rewards](2026-10-11-final-training-reward.png)

## Behavior

| Held-out behavior | Original | Step 100 |
|---|---:|---:|
| Final-report submission rate | 83.72% | 84.88% |
| Correct answers with invalid references | 13 | 0 |
| Correct answers with valid references but unsupported paths | 5 | 10 |
| Episodes without a final report | 28 | 26 |
| Mean delivered observations | 28.78 | 33.94 |
| Mean generated tokens | 11,499 | 11,096 |

On **UID0025**, the public-works spending difference between 1934 and 1946,
sample 1 answered **142** in both models. The baseline cited document
descriptions and a calculation label, which were invalid observation
references. The trained sample cited delivered `obs_5`, `obs_13` and `obs_16`
and received a supported reward. The question improved from 0/4 to 1/4
supported samples.

The largest increase was **UID0042**, asking for the Zipf exponent of 2020 state
unemployment-insurance tax receipts: 0/4 → 3/4 supported samples. Both models
could produce the value **1.172**, but the trained model more often supplied an
accepted evidence path.

The largest decrease was **UID0068**, asking for the lowest absolute OCS rents
and royalties in 2016: 1/4 → 0/4 supported samples. Three trained samples gave
the correct value **56** with valid references, but their paths were judged
unsupported.

These examples were selected after the fixed evaluation. The
[behavior audit](2026-10-11-final-behavior.json) contains their four samples and
checks all raw episodes against the recorded scores, questions and delivered
observations. Ten final correct answers still had unsupported paths and 26
episodes ended without a report. Mean observations increased, so the result
does not establish lower tool-use cost.

## Completion review and limitations

AIR reported training **FAILED** with `CLIENT_ERROR / RUN_EXECUTION_ERROR`,
although both node commands exited successfully. Before export and evaluation
continued, a manual review verified the original completion certificate,
all 1,600 trajectories and the step-100 checkpoint. The failed AIR receipt is
preserved; the platform cause remains unresolved. The repository checks the
hashed evidence but does not automate assembling that review.

Checkpoint verification covers shard headers, tensor completeness and sampled
contents, rather than a full hash of all 70 GB of weights. The independent audit
also verified matching before/after data, source and evaluation policy, and that
the served checkpoint matched the certified model.

This is one training seed on new questions over a shared document corpus. The
intervals measure uncertainty across questions for this run. They do not measure
training-seed variation, performance on the full Pro benchmark, or transfer to
a different archive.

The measured BM25 index used the creators' transformed text, while grep and
document reading used a lossless rendering of the parsed JSON. Table text can
differ between these inputs, including numeric formatting. The
[corpus builder](../../usecases/officeqa/build_corpus.py) reproduces both; the
[data lock](../../usecases/officeqa/DATA.lock.json) records their identities.

## Evidence and reproduction

Use the [OfficeQA guide](../../usecases/officeqa/README.md) to run a new
experiment and the [protocol](../../usecases/officeqa/FINAL_RUN.md) for its
completion checks. The records below retain exact source, model and data
identities, run states and per-question measurements:

- [Baseline](2026-10-11-final-baseline.json)
- [Final evaluation](2026-10-11-final-eval.json)
- [Paired comparison](2026-10-11-final-comparison.json)
- [Independent audit](2026-10-11-final-audit.json)
- [Behavior audit](2026-10-11-final-behavior.json)
