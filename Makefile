# =============================================================================
# verl-on-air — build / register / run
#
#   make help                                    every target, with the resolved config
#   make doctor image volume                     one-time host + image + storage setup
#   make dev-env check                           free pre-flight: lint + tests + verl composition + air schema
#   make smoke prep stage baseline               platform validation + data + model
#   make rung1 rung2 rung3 rung4                 the infra scaling ladder
#   make search-prep ... search-eval             the agentic-search use case
#   make math-prep ... math-eval                 the math use case
#   make runs logs cancel                        ops
#
# Full walkthrough: docs/running-jobs.md    Every setting: docs/configuration.md
# =============================================================================
include config.env

SHELL := /bin/bash
# Every recipe line fails on the first failing command, an unset variable, or a failing
# pipeline stage -- a gate that can print a failure and still exit 0 is not a gate.
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

AIR  := air
RUN  := $(AIR) run -p $(AIR_PROFILE) --watch --file

# ---- run identity -------------------------------------------------------------
# Every training and eval submission carries a RUN_ID (UTC time + commit) and the commit it
# was submitted from ("-dirty" if tracked files have uncommitted changes). Training writes to
# <output_dir>/<RUN_ID>/, so re-running a command starts a NEW run instead of silently
# resuming the last one (continue one with: make search-train RUN_ID=<id> RESUME=auto);
# evals are named after their checkpoint + RUN_ID, and artifacts are never overwritten.
ifndef RUN_ID
RUN_ID := $(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)
endif
GIT_SHA  := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD -- 2>/dev/null || echo -dirty)
IDENTITY  = --override env_variables.RUN_ID=$(RUN_ID) env_variables.GIT_SHA=$(GIT_SHA) \
            env_variables.VOA_IMAGE=$(IMAGE) $(if $(RESUME),env_variables.RESUME=$(RESUME),)
EVAL_DIR ?= $(VOL)/eval
# <run>-<step> of a CKPT=<...>/<run>/global_step_N path, for naming its eval artifacts
CKPT_LABEL = $(notdir $(patsubst %/,%,$(dir $(patsubst %/,%,$(CKPT)))))-$(subst global_step_,step,$(notdir $(patsubst %/,%,$(CKPT))))

# ---- corporate PyPI index auto-detection ------------------------------------
# Locked-down boxes (e.g. Databricks corp hosts) cannot reach pypi.org and use an
# internal proxy instead, configured in ~/.pip/pip.conf. The build container does
# NOT inherit that config, which is exactly how a build died with
# "dns error ... pypi.org". So detect it here and pass it as a build ARG.
#
# It reaches the build as the BuildKit SECRET `pip_index` (docker/uvi.sh), never as a
# --build-arg: an arg a RUN uses is recorded in the image history, so an index URL with
# credentials would ship inside the image. It is exported to the docker process, not
# written into the command line make echoes, and printed with any userinfo redacted.
# Detection lives in scripts/detect_pypi_index.sh (env > uv config > pip config >
# config files) because `pip config get` alone found NOTHING on a box that does
# use an internal proxy -- pip may be absent, or the setting may live in uv config.
# Override explicitly with:  make build PIP_INDEX_URL=https://.../simple
PIP_INDEX_URL ?= $(shell bash scripts/detect_pypi_index.sh 2>/dev/null)
comma := ,
INDEX_SECRET := $(if $(strip $(PIP_INDEX_URL)),--secret id=pip_index$(comma)env=PIP_INDEX_URL,)
REDACT := sed -E 's|(://)[^/@]*@|\1***@|'

# ---- what the image is built from (scripts/image_lock.py) --------------------
# Build args that change what the image contains are make variables (not BUILD_ARGS), so the
# inputs hash the build labels -- and `make stale-check` compares -- covers them.
WITH_VIDEO      ?= 0
OVERRIDE_NCCL   ?= 0
TORCH_INDEX_URL ?=
IMAGE_ENV   = IMAGE_TAG=$(IMAGE_TAG) WITH_VIDEO=$(WITH_VIDEO) OVERRIDE_NCCL=$(OVERRIDE_NCCL) \
              TORCH_INDEX_URL=$(TORCH_INDEX_URL)
IMAGE_ARGS  = --build-arg IMAGE_TAG=$(IMAGE_TAG) --build-arg WITH_VIDEO=$(WITH_VIDEO) \
              --build-arg OVERRIDE_NCCL=$(OVERRIDE_NCCL) \
              $(if $(TORCH_INDEX_URL),--build-arg TORCH_INDEX_URL=$(TORCH_INDEX_URL),) \
              --label org.verl-on-air.inputs=$$($(IMAGE_ENV) python3 scripts/image_lock.py inputs) \
              --label org.verl-on-air.tag=$(IMAGE_TAG)

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  profile=$(AIR_PROFILE)  image=$(IMAGE)  volume=$(VOL)"

# ---------------------------------------------------------------- image ------
.PHONY: certs
certs: ## Copy this host's CA bundle into ./certs (for corporate TLS interception)
	@mkdir -p certs
	@for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt; do \
	  if [ -f "$$f" ]; then cp "$$f" certs/host-ca-bundle.crt; \
	    echo "copied $$f -> certs/host-ca-bundle.crt ($$(grep -c 'BEGIN CERTIFICATE' certs/host-ca-bundle.crt) certs)"; \
	    exit 0; fi; \
	done; echo "no system CA bundle found; on macOS this step is unnecessary" >&2

.PHONY: vendor
vendor: ## Pre-fetch github artefacts on the host (TLS-blocked networks)
	@bash scripts/vendor_artifacts.sh

.PHONY: doctor
doctor: ## Preflight: can THIS machine build the image? (arch/docker/disk/auth)
	@bash scripts/doctor.sh

.PHONY: bootstrap
bootstrap: ## Fresh x86_64 Linux box -> installs tooling, builds, pushes, registers
	@bash scripts/bootstrap_linux.sh

.PHONY: build
build rebuild: export PIP_INDEX_URL := $(PIP_INDEX_URL)
build: ## Build the image (linux/amd64). Extra flags via BUILD_ARGS=...
	@if [ -n "$${PIP_INDEX_URL:-}" ]; then \
	  echo "using detected PyPI index: $$(printf '%s' "$${PIP_INDEX_URL}" | $(REDACT)) (as a BuildKit secret)"; \
	else \
	  echo "using default public PyPI (no local pip index configured)"; \
	fi
	DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
	  $(INDEX_SECRET) $(IMAGE_ARGS) $(BUILD_ARGS) \
	  -f docker/Dockerfile \
	  -t $(IMAGE) .
# Corporate networks: pass a proxy or an internal index without editing anything, e.g.
#   make build BUILD_ARGS="--build-arg HTTPS_PROXY=http://proxy:3128"
#   make build PIP_INDEX_URL=https://mirror.internal/simple    # handed over as a secret
#   make build BUILD_ARGS="--network=host"      # if container DNS is the problem

.PHONY: bump
bump: ## Bump IMAGE_TAG in config.env + every custom-image job (REQUIRED after Dockerfile changes)
	@PYTHON=$(LINT_PY) bash scripts/bump_image_tag.sh $(TAG)

.PHONY: retarget
retarget: ## Point every job file at config.env's image (after editing DOCKERHUB_USER/IMAGE_NAME)
	@$(LINT_PY) scripts/retarget.py

.PHONY: stale-check
stale-check: ## Refuse to push an image not built from the current inputs, or a tag already pushed from other ones
	@$(IMAGE_ENV) python3 scripts/image_lock.py check-local $(IMAGE_TAG) $(IMAGE)
# By CONTENT, not mtime: the image carries the sha256 of its build inputs (docker/*, certs/,
# the content build args) as a label, and docker/IMAGE.lock records the inputs each pushed tag
# was built from. air registration is PER TAG -- re-pushing changed content under a registered
# tag keeps serving the old digest, and a fix "does not work". Repository code is not an input:
# jobs ship it as a snapshot and the image carries none.

.PHONY: rebuild
rebuild: ## Build from scratch: no layer cache, re-pull the base image
	@echo "clean rebuild: --no-cache --pull (expect ~20-30 min, re-downloads ~11 GB)"
	@if [ -n "$${PIP_INDEX_URL:-}" ]; then \
	  echo "using detected PyPI index: $$(printf '%s' "$${PIP_INDEX_URL}" | $(REDACT)) (as a BuildKit secret)"; fi
	DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --no-cache --pull \
	  $(INDEX_SECRET) $(IMAGE_ARGS) $(BUILD_ARGS) \
	  -f docker/Dockerfile \
	  -t $(IMAGE) .

# Release steps run STRICTLY IN ORDER, stopping at the first failure. Sibling prerequisites
# (`release: rebuild size push register`) would not guarantee that: under `make -j` they may
# run concurrently, e.g. pushing before the size gate has passed.
.PHONY: release
release: ## Clean rebuild -> size gate -> push -> register (serial; stops at the first failure)
	$(MAKE) rebuild
	$(MAKE) size
	$(MAKE) push
	$(MAKE) register

.PHONY: size
size: ## Fail if the image is missing, unmeasurable, or over MAX_IMAGE_GB (decimal GB)
	@python3 scripts/image_size.py $(IMAGE) $(MAX_IMAGE_GB)

.PHONY: layers
layers: ## Show layer sizes, largest first (for shrinking the image)
	@docker history $(IMAGE) --human --format '{{.Size}}\t{{.CreatedBy}}' \
	  | sed 's/&&/\n\t\t&&/g' | head -40

.PHONY: push
push: stale-check ## Push to Docker Hub (current inputs + push scope checked first), then record the digest
	@bash scripts/check_dockerhub_push.sh $(DOCKERHUB_USER) $(IMAGE_NAME) \
	  || { echo ""; echo "Refusing to upload ~16 GB that would be rejected."; exit 1; }
	docker push $(IMAGE)
	@$(IMAGE_ENV) python3 scripts/image_lock.py record $(IMAGE_TAG) $(IMAGE)
	@echo "commit docker/IMAGE.lock: it is how a job file's tag maps to one digest"

.PHONY: register
register: ## Register the image with AI Runtime (2-6 min) -- only the digest docker/IMAGE.lock records
	@python3 scripts/image_lock.py check-remote $(IMAGE_TAG) $(IMAGE)
	@if [ -n "$(strip $(SECRET_SCOPE))" ] && [ -n "$(strip $(SECRET_KEY))" ]; then \
	  echo "registering with stored credentials: $(SECRET_SCOPE)/$(SECRET_KEY)"; \
	  $(AIR) register image $(IMAGE) -p $(AIR_PROFILE) \
	    --scope $(SECRET_SCOPE) --key $(SECRET_KEY); \
	else \
	  echo "no SECRET_SCOPE/SECRET_KEY in config.env -> interactive credential setup"; \
	  echo "(it will print the scope/key it creates; put them in config.env to reuse)"; \
	  $(AIR) register image $(IMAGE) -p $(AIR_PROFILE) --interactive-authenticate; \
	fi

.PHONY: image
image: ## build -> size gate -> push -> register (serial; stops at the first failure)
	$(MAKE) build
	$(MAKE) size
	$(MAKE) push
	$(MAKE) register

# ----------------------------------------------------------------- setup -----
.PHONY: volume
volume: ## Create the UC volume (OK if created or it already exists; any other error fails)
	@if out=$$(databricks volumes create $(UC_CATALOG) $(UC_SCHEMA) $(UC_VOLUME) MANAGED \
	      -p $(AIR_PROFILE) 2>&1); then \
	  echo "created volume $(VOL)"; \
	elif printf '%s' "$$out" | grep -qiE 'RESOURCE_ALREADY_EXISTS|already exists'; then \
	  echo "volume $(VOL) already exists"; \
	else \
	  printf '%s\n' "$$out" >&2; echo "FAILED to create volume $(VOL) (see the error above)" >&2; exit 1; \
	fi

.PHONY: smoke
smoke: ## STEP 0  1xA10 image pre-flight (~2 min)
	$(RUN) infra/diagnostics/air/smoke_test.yaml

.PHONY: prep
prep: ## STEP 1  geo3k -> UC volume parquet
	$(RUN) infra/geo3k/air/1_prep.yaml

.PHONY: stage
stage: ## STEP 2  Qwen3.5-35B-A3B (~70 GB) -> UC volume
	$(RUN) infra/air/stage_model.yaml

.PHONY: baseline
baseline: ## STEP 3  measure GRPO reward variance before training
	$(RUN) infra/geo3k/air/2_baseline.yaml

.PHONY: setup
setup: ## volume -> smoke -> data -> model (serial; stops at the first failure)
	$(MAKE) volume
	$(MAKE) smoke
	$(MAKE) prep
	$(MAKE) stage

# ------------------------------------------------------------- the ladder ----
.PHONY: rung1
rung1: ## Qwen3.5-2B  dense  FSDP   8xH100  (cheap full-path check)
	$(RUN) infra/geo3k/air/rung1_2b_fsdp_8gpu.yaml $(IDENTITY)

.PHONY: rung2
rung2: ## Qwen3.5-9B  dense  FSDP   8xH100
	$(RUN) infra/geo3k/air/rung2_9b_fsdp_8gpu.yaml $(IDENTITY)

.PHONY: rung3
rung3: ## Qwen3.5-35B-A3B MoE  CLASSIC+offload  8xH100 (known-good baseline)
	$(RUN) infra/geo3k/air/rung3_35b_classic_8gpu.yaml $(IDENTITY)

.PHONY: rung4
rung4: ## Qwen3.5-35B-A3B MoE  MEGATRON-FSDP no-offload  32xH100  <-- headline
	$(RUN) infra/geo3k/air/rung4_35b_fsdp_16gpu.yaml $(IDENTITY)

# -------------------------------------------------------------- use cases ----
# Same numbered shape for every use case: prep -> (stage/index) -> baseline -> train
# -> eval. Always run the baseline BEFORE training; it is what makes the trained
# number mean anything. Walkthrough: docs/running-jobs.md
UCS := usecases/agentic-search/air
UCM := usecases/math/air

.PHONY: search-prep search-index search-baseline search-train search-train-sync search-eval search-deploy
search-prep: ## agentic-search 1  MuSiQue questions + passage corpus -> Volume
	$(RUN) $(UCS)/1_prep_data.yaml
search-index: ## agentic-search 2  Vector Search index (kicks off; wait for ONLINE): WAREHOUSE_ID=<id>
	$(if $(WAREHOUSE_ID)$(shell grep -E "^ *QA_VS_WAREHOUSE_ID: *'[^']+'" $(UCS)/2_build_index.yaml),,$(error \
	  set WAREHOUSE_ID=<SQL warehouse id> (or QA_VS_WAREHOUSE_ID in $(UCS)/2_build_index.yaml) -- \
	  the table load names its warehouse; none is guessed))
	$(RUN) $(UCS)/2_build_index.yaml $(IDENTITY) \
	  $(if $(WAREHOUSE_ID),env_variables.QA_VS_WAREHOUSE_ID=$(WAREHOUSE_ID),)
search-baseline: ## agentic-search 3  EVAL base model (the "before" number)
	$(RUN) $(UCS)/3_baseline_eval.yaml $(IDENTITY) \
	  env_variables.EVAL_OUT=$(EVAL_DIR)/search_base_$(RUN_ID).json \
	  env_variables.EVAL_TRACE_OUT=$(EVAL_DIR)/search_base_$(RUN_ID)_traces.jsonl
search-train: ## agentic-search 4  GRPO, fully-async, 16xH100, rule reward
	$(RUN) $(UCS)/4_train.yaml $(IDENTITY)
search-train-sync: ## agentic-search 4  GRPO, SYNC co-located, 32xH100 (config-validated only)
	$(RUN) $(UCS)/4_train_sync.yaml $(IDENTITY)
search-eval: ## agentic-search 5  EVAL a checkpoint: make search-eval CKPT=<run>/global_step_N
	$(if $(CKPT),,$(error set CKPT=<run>/global_step_N -- the checkpoint to evaluate (no default)))
	$(RUN) $(UCS)/5_eval.yaml $(IDENTITY) env_variables.EVAL_MODEL_PATH=$(CKPT) \
	  env_variables.EVAL_OUT=$(EVAL_DIR)/search_$(CKPT_LABEL)_$(RUN_ID).json \
	  env_variables.EVAL_TRACE_OUT=$(EVAL_DIR)/search_$(CKPT_LABEL)_$(RUN_ID)_traces.jsonl
search-deploy: ## agentic-search 6  print the deployment recipe (SERVE=1 to serve)
	$(RUN) $(UCS)/6_deploy.yaml

.PHONY: math-prep math-judge math-baseline math-train math-eval
math-prep: ## math 1  Hendrycks MATH L3-5 -> tool-agent parquet
	$(RUN) $(UCM)/1_prep_data.yaml
math-judge: ## math 2  stage the LLM judge into the Volume (once, resumable)
	$(RUN) $(UCM)/2_stage_judge.yaml
math-baseline: ## math 3  EVAL base model on MATH-500 (EVAL_LIMIT=0 for all 500)
	$(RUN) $(UCM)/3_baseline_eval.yaml $(IDENTITY) \
	  env_variables.EVAL_OUT=$(EVAL_DIR)/math500_base_$(RUN_ID).json
math-train: ## math 4  GRPO + co-located judge, 32xH100 (2 train + 2 judge)
	$(RUN) $(UCM)/4_train.yaml $(IDENTITY)
math-eval: ## math 5  EVAL a checkpoint: make math-eval CKPT=<run>/global_step_N
	$(if $(CKPT),,$(error set CKPT=<run>/global_step_N -- the checkpoint to evaluate (no default)))
	$(RUN) $(UCM)/5_eval.yaml $(IDENTITY) env_variables.EVAL_MODEL_PATH=$(CKPT) \
	  env_variables.EVAL_OUT=$(EVAL_DIR)/math500_$(CKPT_LABEL)_$(RUN_ID).json

# ------------------------------------------------------------------ ops ------
.PHONY: runs
runs: ## List recent runs (active and finished)
	$(AIR) list runs -p $(AIR_PROFILE)

.PHONY: logs
logs: ## Stream logs: make logs RUN=<run_id> [NODE=0]
	$(AIR) logs $(RUN) $(if $(NODE),--node $(NODE),) -p $(AIR_PROFILE)

.PHONY: cancel
cancel: ## Cancel a run: make cancel RUN=<run_id>
	$(AIR) cancel $(RUN) -p $(AIR_PROFILE)

.PHONY: dry
dry: ## Validate ONE YAML without submitting: make dry F=usecases/math/air/4_train.yaml
	$(AIR) run --dry-run --file $(F) -p $(AIR_PROFILE)

.PHONY: config
config: ## Print the resolved verl overrides locally: make config MODE=fsdp GPUS=16
	@MODE=$${MODE:-fsdp}; GPUS=$${GPUS:-16}; \
	 DRY_RUN=1 MEGATRON_MODE=$$MODE \
	   NUM_NODES=$$(( GPUS / 8 == 0 ? 1 : GPUS / 8 )) LOCAL_WORLD_SIZE=8 \
	   NODE_RANK=0 MASTER_ADDR=127.0.0.1 \
	   bash engine/train/run_grpo_megatron.sh 2>/dev/null

.PHONY: diff-modes
diff-modes: ## Diff the fsdp vs classic override sets (what actually changes)
	@$(MAKE) -s config MODE=fsdp    GPUS=16 | grep -E '^    [+a-z]' | sort > /tmp/vo-fsdp.txt
	@$(MAKE) -s config MODE=classic GPUS=8  | grep -E '^    [+a-z]' | sort > /tmp/vo-classic.txt
	@echo "--- only in classic (8 GPU) / +++ only in fsdp (16 GPU) ---"
	@diff /tmp/vo-classic.txt /tmp/vo-fsdp.txt || true

.PHONY: validate
validate: ## Validate all air/*.yaml against the REAL air CLI (no image needed)
	@bash scripts/validate_air_yaml.sh $(AIR_PROFILE)

.PHONY: check
check: lint test compose-check validate ## lint + CPU tests + verl composition + air schema (no GPU)

# ---- local toolchain (CPU only) ---------------------------------------------
VENV ?= .venv
PY   ?= $(VENV)/bin/python

.PHONY: dev-env
dev-env: ## Create .venv with the pinned test/lint toolchain (requirements-dev.txt)
	@command -v uv >/dev/null || { echo "uv not found -- install it (https://docs.astral.sh/uv/), or:"; \
	  echo "  python3 -m venv $(VENV) && $(PY) -m pip install -r requirements-dev.txt"; exit 1; }
	uv venv $(VENV) --python 3.12 --allow-existing
	uv pip install --python $(PY) -r requirements-dev.txt

.PHONY: test
test: ## CPU test suite: regression tests for every guard, gate and reward (no GPU, no cloud)
	@test -x $(PY) || { echo "no $(PY) -- run: make dev-env"; exit 1; }
	$(PY) -m pytest

.PHONY: preflight
preflight: ## Validate ONE job locally and print its plan + GPU-hour bound: make preflight F=<job.yaml>
	$(if $(F),,$(error set F=<job.yaml>))
	@test -x $(PY) || { echo "no $(PY) -- run: make dev-env"; exit 1; }
	@RUN_ID=$(RUN_ID) $(PY) scripts/preflight_job.py $(F)

.PHONY: compose-check
compose-check: ## Compose every training job's real overrides against the pinned verl (CPU)
	@test -x $(PY) || { echo "no $(PY) -- run: make dev-env"; exit 1; }
	$(PY) scripts/compose_check.py

# A MISSING shellcheck fails the gate too (it used to print "skipping" and pass -- and so did a
# shellcheck that found problems). Opt out explicitly with ALLOW_NO_SHELLCHECK=1.
SHELLCHECK ?= $(firstword $(wildcard $(VENV)/bin/shellcheck) $(shell command -v shellcheck 2>/dev/null))
LINT_PY    ?= $(if $(wildcard $(PY)),$(PY),python3)
SH_FILES   := $(wildcard scripts/*.sh engine/*/*.sh infra/diagnostics/*.sh docker/retry.sh docker/uvi.sh)

.PHONY: lint
lint: ## Local static checks (shellcheck + python syntax + Dockerfile + yaml parse)
	@if [ -n "$(SHELLCHECK)" ]; then \
	  $(SHELLCHECK) -S warning $(SH_FILES); echo "shellcheck ok  $(words $(SH_FILES)) scripts"; \
	elif [ "$(ALLOW_NO_SHELLCHECK)" = "1" ]; then \
	  echo "WARNING: shellcheck not installed -- NOT CHECKED (ALLOW_NO_SHELLCHECK=1)"; \
	else \
	  echo "shellcheck not installed: run 'make dev-env' (or ALLOW_NO_SHELLCHECK=1 to skip)" >&2; exit 1; \
	fi
	@$(LINT_PY) scripts/lint_python.py engine infra usecases scripts docs tests conftest.py
	@$(LINT_PY) scripts/lint_dockerfile.py
	@$(LINT_PY) scripts/retarget.py --check
	@$(LINT_PY) -c "import yaml,glob; fs=sorted(glob.glob('infra/**/air/*.yaml', recursive=True)+glob.glob('usecases/*/air/*.yaml')); [yaml.safe_load(open(f)) for f in fs]; print(f'yaml ok  {len(fs)} job files')"
