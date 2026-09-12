# OfficeQA GRPO RL — Ultra-Detailed Execution Plan

**Owner:** michael.shtelma · **Infra:** Databricks AI Runtime serverless GPU (`df1`, AWS/H100), verl v0.9.0 fully-async/disaggregated, image `michaelshtelma587/verl-megatron-air:v6` · **Last updated:** 2026-09-12

---

## 0. Goal & success criteria

**Goal.** Demonstrate a *real, meaningful* accuracy improvement of GRPO-trained **Qwen3.5-35B-A3B** on a **hard, uncontaminated, genuinely-agentic held-out benchmark** — proving the fully-async agentic RL stack produces learning that transfers, not a saturated toy gain.

**Why OfficeQA.** Competition math (GSM8K/MATH-500/AIME/HMMT-2026) is **correctness-saturated** for this model (~95–100% right whenever it answers, even on post-cutoff sets). OfficeQA — Databricks' grounded-reasoning benchmark over U.S. Treasury Bulletins (1939–2025, ~89K pages) — has **large real headroom**: frontier LLMs score <34% with the corpus generally, and **Opus ~50% on the HARD split**. A 35B-A3B will start well below that → a wide band for GRPO to climb.

**Deliverable / definition of done.**
- Held-out set = the **133 HARD OfficeQA questions** (`officeqa_pro.csv`). The generator is **blind** to these; nothing derived from them enters training.
- Report **base% → RL-trained%** (exact `score_answer`, plus fuzzy 1%/5%), with the by-difficulty and among-answered breakdowns, and the full trajectories for inspection.
- A **positive, statistically-credible delta** on the hard set is success. Secondary: improved tool-use efficiency (turns, tool-error rate).

**Scientific guarantees (non-negotiable).**
1. **Held-out integrity** — training questions are synthesized without the generator ever seeing the 133 hard questions; a post-hoc contamination audit (embedding + n-gram) drops any near-duplicate.
2. **Trustworthy gold** — training answers are **correct by construction** from atoms verified by a frontier model, *not* by an LLM solving hard questions (which is unreliable: even Opus is ~50%).
3. **Identical eval harness** for base and every checkpoint — the delta is an apples-to-apples measurement.

---

## 1. Architecture at a glance

```
                    ┌─────────────────── OFFLINE (build training data) ───────────────────┐
   Treasury JSONs ─▶ extract_atoms ─▶ GPT-4.1 verify_atoms ─▶ verified_atoms.jsonl
   (clean HTML tables)                (Databricks AI gateway)          │
                                                                       ▼
                              compose_questions (Python operators) ─▶ candidate Q + Python answer
                                                                       │
                          GLM-5.3 render (phrasing, BLIND) ───────────┤
                                                                       ▼
                    validate: well-posedness + GPT-4.1 cross-check ─▶ gold / silver / discard
                                                                       │
                          base-35B pass-rate difficulty band ─────────┤
                                                                       ▼
                          contamination_audit vs 133 hard ─▶ train.parquet (tool_agent schema)
                    └──────────────────────────────────────────────────────────────────────┘
                                                                       │
   ┌──────────────────────── ONLINE (RL + eval) ────────────────────── ▼ ─────────────┐
   │ verl fully-async GRPO: disjoint trainer(Megatron) / rollout(vLLM ToolAgentLoop)   │
   │   tools = officeqa_tools (search/read/list/calculator, local BM25)                │
   │   reward = score_answer rule (+ optional judge)  · SAVE_FREQ checkpoints (HF)     │
   └───────────────────────────────────────────────────────────────────────────────── ┘
                                                                       │
   eval_officeqa_agentic.py (same harness) on BASE and each checkpoint ─▶ base% → RL%
```

**Two corpora, deliberately separated:**
- **Retrieval corpus** (what the model's tools see at train+eval time) = the *flattened* `.txt` → `chunks.jsonl` (131,113 chunks). Realistically messy (OCR, column-swapped headers) — the model must cope, exactly as at eval.
- **Gold-construction source** (offline only) = the *pre-flatten* JSONs with clean HTML `<table>` structure. Used only to build/verify atoms; never seen by the trained model.

---

## 2. Current status (2026-09-12)

| Component | State |
|---|---|
| Checkpoint save (35B → loadable HF, no OOM) | ✅ validated (run 410679584424844) |
| Fully-async agentic multi-turn + GLM judge infra | ✅ validated (air/53 run 864953242920662) |
| OfficeQA scorer (`score_answer`) + `XMLTagExtractor` | ✅ ported + offline-validated |
| Corpus tools (BM25 search + read/list/calculator) | ✅ built, offline-validated; BM25-at-scale in-job pending final confirm |
| Retrieval index (131,113 chunks) + Volume staging | ✅ built + staged |
| Eval harness `eval_officeqa_agentic.py` | ✅ built; staging+serve+tool-parse+render **confirmed in-job** (run 187346983850485) |
| **Bug: tool-registry collision** (`calculator` double-register) | ✅ fixed (`safe_eval.py`) |
| **Bug: BM25 freeze → vLLM ServerDisconnected** | ✅ fixed (pre-warm + executor + retry); **reconfirming in run 326906562878044** |
| Baseline number (base-35B on hard set) | ⬜ **next** (air/71) |
| Atom bank → synthesis → validation → train.parquet | ⬜ to build |
| RL training + trained eval + the delta | ⬜ to build |

---

## 3. Jobs (air/*.yaml) — every job, purpose, compute, key env, outputs

Submit each with `air run --file <yaml> -p df1`; status `air get run <id> -p df1`; logs `air logs <id> -p df1 --tail N`. Monitor via the Monitor tool (background bash monitors get memory-killed).

| # | Job (air file) | Purpose | Compute | Serves | Key env | Runtime | Output |
|---|---|---|---|---|---|---|---|
| J1 | `air/70_eval_officeqa_smoke.yaml` | Pipeline smoke (24 Q) | 8×H100 | base-35B (TP8) | `EVAL_LIMIT=24`, `OFFICEQA_STAGE=1`, `EVAL_SCRIPT=eval_officeqa_agentic.py` | ~13 min | `officeqa_smoke.json` |
| J2 | `air/71_eval_officeqa_base.yaml` | **BASELINE** (246 Q, by-difficulty) | 8×H100 | base-35B (TP8) | `EVAL_LIMIT=0`, `EVAL_CONCURRENCY=32` | ~30–40 min | `officeqa_base.json` |
| J3 | `air/72_officeqa_atoms.yaml` *(new)* | Extract + **GPT-4.1 verify** atoms | CPU (2–8 vCPU) | — (gateway HTTP) | `OFFICEQA_JSONS`, `VERIFIER_ENDPOINT=agentbricks-eval-gpt-41-2025-04-14`, `ATOM_TRIALS=4` | ~1–3 h (batched) | `verified_atoms.jsonl` |
| J4 | `air/73_officeqa_synth.yaml` *(new)* | Compose+render+validate questions | 16×H100 (GLM serve) | GLM-5.3 (TP16) | `SYNTH_N_PER_OP`, GLM serve (serve_judge), GPT-4.1 gateway for cross-check | ~2–4 h | `synth_candidates.jsonl` (gold/silver) |
| J5 | `air/74_officeqa_band.yaml` *(new)* | Base-35B pass-rate → difficulty band | 8×H100 | base-35B (TP8) | reuse serve loop, `M_TRIALS=8`, temp>0 | ~1–2 h | `synth_banded.jsonl` (+ p per Q) |
| J6 | `air/75_officeqa_train.yaml` *(new)* | **GRPO fully-async training** | 32×H100 (disagg) | trainer Megatron + rollout vLLM | officeqa tools + reward, `SAVE_FREQ`, `REWARD_SOURCE=rule` | hours | `global_step_N/actor/.../huggingface/` |
| J7 | `air/76_officeqa_eval_trained.yaml` *(new)* | Eval each checkpoint (== J2 on ckpt) | 8×H100 | ckpt-35B (TP8) | `EVAL_MODEL_PATH=<ckpt hf dir>` | ~30–40 min ×N | `officeqa_ckpt_<N>.json` |

Supporting/existing serve jobs referenced: `air/52c_serve_judge_glm53_tp16.yaml` (GLM-5.3 serve pattern, reused by J4), `air/57/59` (agentic-judge train configs, templates for J6), `air/60_ckpt_save_derisk.yaml` (checkpoint-save de-risk, already green).

### Job detail sheets

**J2 — Baseline (`air/71`).** Serves base 35B (TP8, `EVAL_SERVE_LEN=32768`, `--disable-custom-all-reduce`), stages OfficeQA data to `/local_disk0`, runs the agentic loop over all 246 with `EVAL_CONCURRENCY=32`, `EVAL_MAX_TURNS=10`, `EVAL_MAX_TOKENS=2048`, temp 0. **Headline = the `hard` row** of the by-difficulty report (directly comparable to Opus ~50%); `easy` row calibrates the warm-up band. **Go/no-go:** hard in a workable band (roughly 5–40% exact, with WRONG answers not just no-answers) ⇒ proceed. ~0% ⇒ ease tools/questions; already-high ⇒ rethink (unlikely given Opus 50%).

**J3 — Atom bank (`air/72`, new).** No GPU. Steps: (a) `extract_atoms.py` parses `treasury_bulletins_parsed/jsons/*.json`, keeps `type=="table"` elements, parses HTML via `html_table_parser.py` into `(bulletin_date, table_title, section, row_label, period, value, unit)` candidate triples; filters to **well-formed data tables** (period-header present, numeric grid, low `nan`). (b) `verify_atoms.py` sends each candidate + the **raw table text** to **GPT-4.1** via the gateway, `ATOM_TRIALS` times (prompt/temp ensembled), keeps only **unanimously-confirmed** atoms (catches the header/value column-swap corruption). Output `verified_atoms.jsonl` with provenance. Runs on CPU because the gateway does the model work; can also run locally from the dev box (gateway reachable via df1 auth) for a first batch.

**J4 — Synthesis (`air/73`, new).** Serves GLM-5.3 (TP16 cross-node, `serve_judge.sh` pattern). Driver: (a) `compose_questions.py` samples coherent verified-atom sets and applies a **deterministic operator** (Σ-over-months, diff, %-change, ratio, CAGR, argmax-period, threshold-year, multi-category-sum, multi-hop) → **answer computed in Python** + a computation trace; (b) `render_questions.py` calls GLM-5.3 (BLIND to hard eval) to phrase a fluent, **unambiguous** NL question that pins units / CY-vs-FY / revised-vs-unrevised; (c) `validate_questions.py` runs a **well-posedness** GPT-4.1 pass (one defensible interpretation, else discard) and a **cross-check** (GPT-4.1 solves from scratch N×: consensus==construction ⇒ *gold*; construction-sound but no consensus ⇒ *silver* [genuinely-hard, kept, flagged]; disagrees ⇒ parse/label bug ⇒ discard). Output `synth_candidates.jsonl`.

**J5 — Difficulty band (`air/74`, new).** Serves base-35B; reuses the eval agentic loop to solve each candidate `M_TRIALS` times (temp>0) and records pass-rate `p` vs the constructed answer. Keep `p≈0.1–0.6` (GRPO needs intra-group variance — always-fail/always-pass give no gradient); keep a slice of easy (high-p) for warm-up. Output `synth_banded.jsonl`.

**J6 — Training (`air/75`, new).** verl fully-async GRPO, disaggregated (Megatron trainer / vLLM `ToolAgentLoop` rollout), 32×H100 (35B validated offload-free at 32 GPU). `rollout.multi_turn.function_tool_path=scripts/tools/officeqa_tools.py` (registers all 4 tools — the safe_eval fix makes this collision-free), `TOOL_FORMAT=qwen3_coder`. Reward = `officeqa_judge_reward.compute_score` with `REWARD_SOURCE=rule` (constructed gold ⇒ `score_answer` exact is sound; judge optional/blend for text answers). `SAVE_FREQ=8` (global steps; validated save→HF export). Launcher `run_grpo_fully_async.sh` via `dispatch_agentic.sh`.

**J7 — Trained eval (`air/76`, new).** Identical to J2 with `EVAL_MODEL_PATH` pointed at `global_step_N/actor/model/huggingface/`. Run for each saved checkpoint → the **delta curve**.

---

## 4. Scripts inventory

| Script | Role | State |
|---|---|---|
| `scripts/eval_officeqa_agentic.py` | Agentic eval harness (4 tools → `<FINAL_ANSWER>` → `score_answer`; pre-warm+executor+retry) | ✅ built |
| `scripts/tools/officeqa_tools.py` | search/read/list/calculator; plain `_impl`s + `@function_tool` wrappers; local BM25 | ✅ built |
| `scripts/tools/safe_eval.py` | Decorator-free safe AST calculator (breaks the registry collision) | ✅ built |
| `scripts/serve_and_eval.sh` | Serve vLLM + stage OfficeQA data + run `EVAL_SCRIPT` | ✅ extended |
| `scripts/officeqa/build_chunks.py`,`chunker.py`,`html_table_parser.py` | Corpus → `chunks.jsonl`; HTML table parse | ✅ built/ported |
| `scripts/reward/officeqa_reward.py` | Official `score_answer` (unit-aware) | ✅ ported |
| `scripts/reward/answer_extract.py` | `XMLTagExtractor("FINAL_ANSWER")` | ✅ ported |
| `scripts/reward/judge_reward.py` | MATH LLM-judge reward (template for OfficeQA judge) | ✅ exists |
| `scripts/serve_judge.sh` | Serve GLM-5.3 (judge/generator) | ✅ exists |
| `scripts/dispatch_agentic.sh`,`run_grpo_fully_async.sh` | Agentic-train dispatcher + fully-async launcher | ✅ exists |
| `scripts/officeqa/extract_atoms.py` | JSON elements → candidate atoms | ⬜ new (J3) |
| `scripts/officeqa/verify_atoms.py` | GPT-4.1 multi-trial atom certification | ⬜ new (J3) |
| `scripts/officeqa/compose_questions.py` | Operators over verified atoms → Q + Python answer | ⬜ new (J4) |
| `scripts/officeqa/render_questions.py` | GLM-5.3 NL phrasing (blind) | ⬜ new (J4) |
| `scripts/officeqa/validate_questions.py` | Well-posedness + GPT-4.1 cross-check → gold/silver/discard | ⬜ new (J4) |
| `scripts/officeqa/band_difficulty.py` | Base-35B pass-rate banding | ⬜ new (J5) |
| `scripts/officeqa/contamination_audit.py` | Embedding + n-gram leakage check vs 133 hard | ⬜ new |
| `scripts/prep_officeqa_tool_agent.py` | Assemble `train.parquet` (tool_agent schema) | ⬜ new |
| `scripts/reward/officeqa_judge_reward.py` | `compute_score`: `score_answer` rule + optional judge | ⬜ new (J6) |
| `scripts/gateway_client.py` | Shared OpenAI-compatible Databricks-gateway client (GPT-4.1) | ⬜ new (J3/J4) |

---

## 5. Data & artifacts inventory

| Path | What |
|---|---|
| `/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B` | Base model (tokenizer + weights) |
| `/Volumes/main/mshtelma/verl/data/officeqa/officeqa_full.csv` | 246 Q (133 hard + 113 easy) |
| `/Volumes/main/mshtelma/verl/data/officeqa/officeqa_pro.csv` | 133 hard = **held-out eval** |
| `/Volumes/main/mshtelma/verl/data/officeqa/treasury_bulletins_transformed.zip` | 697 `.txt` retrieval docs |
| `/Volumes/main/mshtelma/verl/data/officeqa/chunks.jsonl` | 131,113 BM25 chunks (retrieval index source) |
| `databricks-deep-research-agent/.../treasury_bulletins_parsed/jsons/*.json` | 697 clean-HTML-table docs (gold-atom source) |
| `/Volumes/main/mshtelma/verl/eval/officeqa_*.json` | Eval outputs (base, ckpts) |
| `/Volumes/main/mshtelma/verl/data/officeqa/verified_atoms.jsonl` *(planned)* | Verified atom bank (J3) |
| `/Volumes/main/mshtelma/verl/data/officeqa/train.parquet` *(planned)* | Training set (J4/J5) |
| Gateway `agentbricks-eval-gpt-41-2025-04-14` | GPT-4.1 verifier (also gpt-4o, gpt-4o-mini) |

---

## 6. Reward design

- **Rule (primary, `REWARD_SOURCE=rule`):** `score_answer(ground_truth, extract_final_answer(trajectory), tol=0)` → 1.0/0.0. Trustworthy because gold is constructed. Guards: `None`/empty → 0.0.
- **Judge (optional, `blend`):** adapt `judge_reward.py` → `officeqa_judge_reward.py` with an OfficeQA grading prompt (given the reference answer + trajectory). Reserve for **text/date/partial-credit** answers where exact match is too brittle. Same `compute_score` contract, telemetry (`score`, `judge_score`, `acc`=rule, `judge_agree`, `judge_ok`, `n_tool_calls`, `num_turns`).
- Endpoint resolution via rendezvous file (Ray actors don't inherit driver env) — already solved in `judge_reward.py`.

---

## 7. Sequencing & decision gates

```
J1 smoke ✅fix→ J2 BASELINE ──(gate A: headroom)──▶ J3 atoms ─▶ J4 synth ─▶ J5 band ─▶ audit ─▶ prep parquet
                                                                                              │
                                                             (gate B: ≥N gold+silver, clean)  ▼
                                                                                       J6 train (SAVE_FREQ)
                                                                                              │
                                                                          (gate C: reward rising) ▼
                                                                                       J7 eval ckpts ─▶ Δ
```
- **Gate A (after J2):** base hard% in workable band ⇒ go. Also yields the pass-rate distribution that calibrates J5.
- **Gate B (after audit):** enough validated questions (target ≥ a few k, mixed gold/silver, difficulty-banded, zero contamination hits) ⇒ train.
- **Gate C (during J6):** MLflow `acc`/reward trending up, tool-error rate sane, no truncation collapse ⇒ let checkpoints accrue; else adjust (LR, curriculum, reward).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Table-parse corruption (column swaps, `nan` grids) poisons atoms | Gold from clean JSON HTML **and** GPT-4.1 atom-verify from raw table; well-formed-table filter |
| Solver can't certify hard answers (Opus ~50%) | Truth by **construction**, not consensus; silver kept via construction |
| Ambiguous questions → reward noise | Explicit well-posedness pass; discard non-unique |
| Contamination of held-out claim | Generator blind to 133 hard; embedding+n-gram audit; eval=hard-only |
| No GRPO gradient (all-pass/all-fail) | Difficulty band by base pass-rate (keep p≈0.1–0.6) |
| verl tool-registry name collision | Decorator-free `safe_eval.py`; unique tool names (**fixed**) |
| Sync CPU work freezes async eval → vLLM disconnects | Pre-warm BM25, run tools in executor, retry `_complete` (**fixed**) |
| 35B checkpoint save OOM | Validated mbridge full-gather HF export at 35B (no recurrence) |
| GLM/GPT gateway rate limits | Batch + backoff in `gateway_client.py`; cap concurrency |

---

## 9. Open items

- **Verifier endpoint auth/shape** — confirm calling `agentbricks-eval-gpt-41-2025-04-14` over the gateway (headers, model name); check for a Claude/Opus gateway endpoint too (given Opus ~50% reference).
- **Silver policy** — include silver (construction-trusted, solver-unconfirmed) in training from the start, or gold-first then add silver? (Leaning: include, flagged, difficulty-banded.)
- **Judge on/off** — start rule-only (numeric gold is exact); add judge only if text/date answers need partial credit.
- **Train scale/curriculum** — warm up on easy+medium then mix hard, vs difficulty-weighted mixed sampling.
