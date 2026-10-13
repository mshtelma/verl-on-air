# Presenting the results

Show an analytical question, the evidence the agent retrieves, and the answer
it produces. Then show the held-out before/after comparison. OfficeQA and
MuSiQue exercise the same training engine with different tools and rewards.
Each gain is measured against its own baseline and metric, so avoid comparing
the two gains directly.

## A five-minute walkthrough

| Part | Material | Explain |
|---|---|---|
| The task | [OfficeQA workflow](../usecases/officeqa/README.md#what-the-agent-does) | Search an archive, interpret tables and carry out the requested analysis |
| One example | UID0025 below | Both models can give the right value while only one supplies accepted evidence |
| The result | [Comparison chart](../results/heldout-summary.png) | Show gains against each baseline, including uncertainty |
| Reproduce it | [OfficeQA instructions](../usecases/officeqa/README.md#run-a-new-experiment) | Configure a workspace, prepare a run, inspect its plan and submit it |
| Extend it | [OfficeQA Pro V2 plan](document-research.md#next-experiment-officeqa-pro-v2) | Test transfer to the creators' separate document archive |

The [paper](https://arxiv.org/abs/2603.08655) frames OfficeQA as end-to-end
grounded reasoning. Treasury Bulletins act as a public proxy for enterprise
archives with complex tables, changing definitions and revised values. The
benchmark covers retrieval and quantitative analysis, including statistical
calculations and some external-data or visual questions.

## Before and after

Question **UID0025** asks for the difference in public-works spending between
1934 and 1946. Sample 1 returned **142** in both models. The baseline used
descriptive references; the trained model cited observations it had received,
and the support judge accepted its path.

| Same question and sample | Value | References | Supported reward |
|---|---|---|---:|
| Baseline | 142 | Document descriptions and a calculation label | 0 |
| Step 100 | 142 | Delivered `obs_5`, `obs_13`, `obs_16` | 1 |

This is an abridged example selected after evaluation. It illustrates a change
in evidence reporting. The [behavior audit](../results/officeqa/2026-10-11-final-behavior.json)
also includes the improving Zipf-exponent question and the regressing OCS-rents
question, with all four samples. Use both improvements and regressions when
discussing the remaining failure modes.

## Held-out comparison

![OfficeQA supported pass@1 and MuSiQue exact match before and after training](../results/heldout-summary.png)

OfficeQA trained on 203 questions for 100 updates and evaluated 43 held-out hard
questions with four samples each. Supported pass@1 rose from **8.72% to 20.93%**:
**+12.21 points**, with a paired 95% interval of **+4.65 to +20.35**. Answer
accuracy (correct final answers in well-formed reports) rose from **19.19% to
26.74%**. Citation validity and the support verdict do not affect that answer
metric; malformed reports count as incorrect. The evidence reward and answer checker
are specific to our experiment, as is the 43-question held-out split.
The creators' Pro benchmark has 133 hard questions and scores answer
correctness, so label our subset and metric on every results slide.

The [report](../results/officeqa/2026-10-11-final-report.md) explains the manual
review we did before evaluation despite AIR's FAILED status, and the limits of
the result.

MuSiQue exact match was **34.8%** for the baseline, **37.6%** for run 1 and
**40.4%** for run 2. Run 1's interval includes zero. Run 2's step-20 checkpoint,
chosen before training, gained **+5.6 points [+2.0, +9.2]**, with paired p = 0.004.
The earlier **54% → 58.5%** figure is a development-set result used to select a
checkpoint. [RESULTS](../RESULTS.md) explains both comparisons.

OfficeQA uses the PyTorch 2.13 development stack; MuSiQue used the earlier
stack. OfficeQA has one training seed and MuSiQue two. The intervals describe
question uncertainty, and more seeds are needed to assess repeatability.
OfficeQA used more observations after training, so avoid claiming improved
efficiency. Math has a recipe and judge qualification, with no published
learning result.

## Demo materials

The [PNG](../results/heldout-summary.png) works in slides; the
[SVG](../results/heldout-summary.svg) scales for export. The figures and example
are recorded results and can be shown offline. To introduce the implementation,
show the code map, a prepared `plan.json` and the saved experiment status:

```bash
make help
.venv/bin/python usecases/officeqa/final_run.py --help
make officeqa-final-status EXPERIMENT=<prepared-experiment>
```

Regenerate the comparison chart from the committed evidence with:

```bash
uv run --no-project --with matplotlib==3.11.2 python scripts/summarize_results.py --plot
python3 scripts/summarize_results.py --check
```
