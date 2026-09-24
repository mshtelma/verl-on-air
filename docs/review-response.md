# Response to the review of rev `6eec6e5`

Each finding, what changed, and what shows it is fixed. "Tests" are CPU regression tests in
`tests/` and `usecases/*/tests/`, run by `make check` together with lint, the composition of every
training job against the pinned verl, and validation of every job file by the air CLI. Each of the
reviewer's reproductions was ported to a test that failed before its fix. "GPU" names a run from
the acceptance suite on H100 with image v9, listed at the end. Replaying the reviewer's
`reproduce.py` against the fixed tree (same inputs, current public APIs) gives 15 of 15 no longer
reproducing, and `reproduce_ops.py` runs unmodified with every injected failure now failing.

## P1

| ID | fix | commits | proof |
|---|---|---|---|
| R01 eval path | `resolve_code_path` shared by the dispatcher and the eval launcher; the eval script and model are checked before vLLM starts | 3512b13 | `test_eval_launcher.py`; GPU A1, A2, A6 |
| R02 false success | a completion certificate on every exit code: the tracker rewritten by this run at the planned final version, `ckpt_contents.json` and a verified HF export, no ABORT, no hard-error signature; `run_result.json` | 6c9598c, 8fc6581, 1f1a281, cb5ae8e | `test_async_guard.py`, `test_sync_guard.py`; GPU A5a (reward raises, verl exits 0, reported FAILED "stopped early"), A5b (trainer killed after its first save, FAILED), A4 (CERTIFIED 2/2) |
| R03 + N1 sync switch | the sync launcher wires the use case's reward; roles validated on every rank; the judge node count must be stated (`JUDGE_NODES`) | f36f780, a074552 | `test_dispatch.py`, `test_compose.py`; GPU A7: the recipe runs out of memory as configured (documented; the certificate was correct) |
| R04 image rewrite | `scripts/retarget.py` rewrites by field; `make bump` and `make retarget` fail when nothing changes; also covers the Volume prefix and Vector Search names | 5cc98c2, d51d44a | `test_retarget.py` |
| R05 absent checkpoint | no default checkpoint; `verify_checkpoint.py` checks the manifest and each shard's safetensors header (mbridge over-declares `total_size`) | 3512b13, cb5ae8e | `test_checkpoint.py`; GPU A6 |
| R06 judge parser | grammar-constrained JSON verdict, strict validation, the same key set on every path, bounded retries, a failure budget that aborts | a910e91 | `test_judge_reward.py` (with the self-check), `test_dispatch.py` (the pre-train gate); GPU A8 |
| R07 eval outage | eval contract: per-question status, an infrastructure error budget, a readiness gate, an identity header, atomic artifacts that are never overwritten | c189fb2, 41f5fa8 | `test_eval_contract.py` (both use cases) |
| R08 geo3k images | requests built the way verl's dataset builds them; the baseline disables vLLM's custom all-reduce | bbd40b9, 68eef71 | `test_geo3k_baseline.py`; GPU A3: decoded images reach vLLM, 29.7% of groups carry signal |
| R09 reward roles | a role-span agent loop; the reward reads the answer from assistant spans only and gives retrieval credit from tool spans only; fails closed | a193093 | `test_role_spans.py`, `test_reward.py`; GPU A4 (no ABORT on any sample) |
| R10 gates | strict shell flags, ShellCheck required, all compile errors collected, an integer size gate, doctor counters | 330e80a | `test_gates.py`; the reviewer's `reproduce_ops.py` |
| R11 headline | reworded as a selected dev-set observation; paired statistics; a held-out test split scored once | f91c71a, 7a5ff18 | `results/agentic-search/`; [RESULTS.md](../RESULTS.md) |

## P2

| ID | fix | commits | proof |
|---|---|---|---|
| R12 eval vs training loop | tool schemas from verl's registry, verl's parser through its public API, an explicit versioned `eval_policy` | bdb48cc | `test_eval_contract.py` |
| R13 math equivalence | one extractor, verl's `prime_math` grader under an explicit numeric policy, at least 30 labelled edge cases | 97e4511 | `test_grading.py` (3 documented `prime_math` limitations as xfail) |
| R14 std-normalisation | explanation corrected; `NORM_ADV_BY_STD_IN_GRPO` set explicitly | 97e4511 | docs |
| R15 eval cache | content-addressed local cache, verified before the rename | f89e7cf | `test_eval_launcher.py::test_a_reused_cache_never_serves_the_previous_models_weights` |
| R16 output dirs | a `RUN_ID` per run, explicit `RESUME`, `run_manifest.json`; `SEED` for the data order | c89ddfa, 511b368 | `test_run_identity.py` |
| R17 floating revisions | pinned model and dataset revisions, `STAGED.json`, `DATA_MANIFEST.json`; hashes computed locally | f89e7cf, 2a3b9f7, d980758 | `test_stage_model.py`, `test_data_manifest.py`; GPU A3 prep |
| R18 corpus | a missing source fails the build; provenance and coverage recorded | 2a3b9f7 | `test_data_prep.py`, `test_data_sources.py` |
| R19 index lifecycle | versioned tables and indexes, never truncated, ownership tagged; `make cleanup-vs` deletes only what it owns | 2a3b9f7, d51d44a | `test_create_vs_index.py`, `test_cleanup.py` |
| R20 deploy | the 8-GPU job that printed text is gone; [deploy.md](deploy.md) says deployment is not implemented | 2b537d0 | |
| R21 image | base by digest, `requirements.lock`, wheels by sha256, sources by commit, `IMAGE.lock`; v9 built, pushed and registered | 8a3a764 | `test_image_lock.py`, `test_vendor_artifacts.py`; every GPU run above used v9 |
| R22 build and release | content-based stale check before push, digest check before register, serial steps | 8a3a764 | `test_image_lock.py` |
| R23 calculator | resource bounds (size, depth, bits, factorial and power pre-checks); never echoes the input | 348156e | `test_calculator.py` (51 cases) |
| R24 judge outage | explicit fallback and a per-worker failure budget; agreement computed on valid verdicts only; judge input bounded in tokens | a910e91, 441425d | `test_judge_reward.py` |
| R25 topology | typed preflight: knobs, modes, geometry against the model, batch divisibility, FSDP with offload refused; `make preflight` prints the plan | 782bc68 | `test_preflight.py` |
| R26 lifecycle | traps installed before `ray start`, heartbeats, fresh rendezvous, signal-aware exit records, the driver in its own process group | 24fcda8, 8fc6581 | `test_lifecycle.py`; `air cancel` delivers no signal a job can trap (documented) |
| R27 probes | gates exit non-zero with a JSON verdict; diagnostics labelled as such | 5b073a4 | `test_probes.py` |
| R28 trace analysis | runs anywhere, the full EM identity, pairing by uid, presented as a hypothesis generator | b4b1b13 | `test_analyze_traces.py`; replay |
| R29 sizing | bytes internally, GiB displayed, evidence labels, "smallest validated configuration" | 9580144 | `test_sizing.py` |
| R30 setup and cost | permissions checklist, storage table, `BUDGET_OK=1` gate, `prune-ckpts`, `cleanup-vs`, workspace retarget, optional HF token | d51d44a | `test_cleanup.py`, `test_retarget.py` |
| R31 coverage | 552 CPU tests, a CI workflow, `make check`, the GPU acceptance suite below | 5efb99c and every later commit | `.github/workflows/check.yml` (making it a required check is a GitHub setting) |
| R32 security | credential-free builds (BuildKit secret, redaction), booleans parsed, [security.md](security.md), FIPS set in one place with its trade-off | 8a3a764, a910e91, 08f2ee8 | `test_image_lock.py`, `test_judge_reward.py` |

## P3 and the documentation table

| ID | fix | commits |
|---|---|---|
| R33 | Apache-2.0 `LICENSE`, `THIRD_PARTY_NOTICES.md`, `.idea/` untracked, `rung4_…_32gpu.yaml`, 122B notes archived, `-p <profile>` in generic docs | 2b537d0 |
| documentation corrections (15 rows) | every row applied: `no human labels`, CUDA compilation, the stack's provenance, `EVAL_SCRIPT`, the math smoke, `SAVE_FREQ`, per-turn budgets, the cache-blocks error, the A10 smoke, sync claims, one variable per rung, the build cache, the profile, the judge topology, the contamination claim | 8a3a764, 2b537d0, 9580144 and earlier Phase 1 commits |
| improvement 6 | the knob table in configuration.md is generated from the preflight schema; `make lint` fails on drift | 9d9dc97 |

## GPU acceptance suite (H100, image v9 unless noted)

| run | checks | result |
|---|---|---|
| A1 / A2 (v8) | eval readiness, valid artifacts, identity | passed (search base EM 0.50 on 20; math 0.84 on 32) |
| A3 | decoded geo3k images reach vLLM; real group variance | passed after the all-reduce fix: 19 of 64 groups mixed (29.7%, CI 19.9–41.8%) |
| A4 | tool trajectories, role spans in Ray workers, an update, weight sync, and the certificate on a real teardown | v8 trained to 2/2, but the certificate hit a Volume read error (fixed); v9 CERTIFIED 2/2 |
| A5a / A5b | an exit-0 crash and a killed trainer are reported as failures | both FAILED with the right reason |
| A6 | a trained checkpoint loads and is evaluated with its identity | passed (20/20 valid; the image's environment reaches jobs) |
| A7 | the sync search recipe | out of memory in the first actor update, twice; certificate correct; recipe marked experimental |
| A8 | the math judge path: prebuilt judge Ray, the self-check gate, one certified sync | CERTIFIED 1/1 |
| V1 | the search variance probe | 20 of 64 groups mixed at T=1.0, n=8 (31.2%, CI 21.2–43.4%) |
| held-out test | base vs the selected pure-EM step 20, with and without tools; a fresh seed | 500 questions, all valid: with tools 34.8% vs 37.6% (+47 / −33, p = 0.15, not significant); closed-book 4.4% vs 5.2%. The seed-7 replicate (step 20 fixed in advance) is still training. [RESULTS.md](../RESULTS.md) |
