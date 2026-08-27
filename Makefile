# =============================================================================
# verl-on-air — build / register / run
#
#   make help
#   make build size push register     # one-time image setup
#   make smoke prep stage             # one-time data + model setup
#   make baseline                     # measure GRPO signal before training
#   make rung1 rung2 rung3 rung4      # the validation ladder
# =============================================================================
include config.env

SHELL := /bin/bash
.DEFAULT_GOAL := help

AIR  := air
RUN  := $(AIR) run -p $(AIR_PROFILE) --watch --file

# ---- corporate PyPI index auto-detection ------------------------------------
# Locked-down boxes (e.g. Databricks corp hosts) cannot reach pypi.org and use an
# internal proxy instead, configured in ~/.pip/pip.conf. The build container does
# NOT inherit that config, which is exactly how a build died with
# "dns error ... pypi.org". So detect it here and pass it as a build ARG.
#
# Deliberately a build ARG, never ENV: the proxy is a BUILD-time concern. Training
# nodes have different egress and must not inherit it.
# Detection lives in scripts/detect_pypi_index.sh (env > uv config > pip config >
# config files) because `pip config get` alone found NOTHING on a box that does
# use an internal proxy -- pip may be absent, or the setting may live in uv config.
# Override explicitly with:  make build PIP_INDEX_URL=https://.../simple
PIP_INDEX_URL ?= $(shell bash scripts/detect_pypi_index.sh 2>/dev/null)
ifneq ($(strip $(PIP_INDEX_URL)),)
INDEX_ARGS := --build-arg PIP_INDEX_URL=$(PIP_INDEX_URL)
endif

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
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
build: ## Build the image (linux/amd64). Extra flags via BUILD_ARGS=...
	@if [ -n "$(strip $(PIP_INDEX_URL))" ]; then \
	  echo "using detected PyPI index: $(PIP_INDEX_URL)"; \
	else \
	  echo "using default public PyPI (no local pip index configured)"; \
	fi
	docker build --platform linux/amd64 \
	  $(INDEX_ARGS) $(BUILD_ARGS) \
	  -f docker/Dockerfile \
	  -t $(IMAGE) .
# Corporate networks: pass a proxy or an internal index without editing anything, e.g.
#   make build BUILD_ARGS="--build-arg HTTPS_PROXY=http://proxy:3128"
#   make build BUILD_ARGS="--build-arg PIP_INDEX_URL=https://mirror.internal/simple"
#   make build BUILD_ARGS="--network=host"      # if container DNS is the problem

.PHONY: rebuild
rebuild: ## Build from scratch: no layer cache, re-pull the base image
	@echo "clean rebuild: --no-cache --pull (expect ~20-30 min, re-downloads ~11 GB)"
	@if [ -n "$(strip $(PIP_INDEX_URL))" ]; then echo "using detected PyPI index: $(PIP_INDEX_URL)"; fi
	docker build --platform linux/amd64 --no-cache --pull \
	  $(INDEX_ARGS) $(BUILD_ARGS) \
	  -f docker/Dockerfile \
	  -t $(IMAGE) .

.PHONY: release
release: rebuild size push register ## Clean rebuild -> size gate -> push -> register

.PHONY: size
size: ## Fail if the image exceeds the DCS limit (see MAX_IMAGE_GB)
	@bytes=$$(docker image inspect $(IMAGE) --format '{{.Size}}'); \
	gb=$$(echo "scale=2; $$bytes/1024/1024/1024" | bc); \
	echo "image size: $${gb} GB (DCS hard limit 20 GB, gate $(MAX_IMAGE_GB) GB)"; \
	if (( $$(echo "$$gb > $(MAX_IMAGE_GB)" | bc -l) )); then \
	  echo ""; \
	  echo "TOO LARGE. Registration will time out replicating this image."; \
	  echo "Size levers, cheapest first:"; \
	  echo "  * confirm UV_NO_CACHE=1 took effect (the uv cache is ~11 GB)"; \
	  echo "  * --build-arg WITH_VIDEO=0 (drops ffmpeg + torchcodec)"; \
	  echo "  * drop nvidia-modelopt if Megatron-Bridge tolerates it"; \
	  echo "  * docker history $(IMAGE) --human --format '{{.Size}}\t{{.CreatedBy}}'"; \
	  exit 1; \
	fi; \
	echo "OK — under the gate."

.PHONY: layers
layers: ## Show layer sizes, largest first (for shrinking the image)
	@docker history $(IMAGE) --human --format '{{.Size}}\t{{.CreatedBy}}' \
	  | sed 's/&&/\n\t\t&&/g' | head -40

.PHONY: push
push: ## Push to Docker Hub (verifies push scope first, then uploads)
	@bash scripts/check_dockerhub_push.sh $(DOCKERHUB_USER) $(IMAGE_NAME) \
	  || { echo ""; echo "Refusing to upload ~16 GB that would be rejected."; exit 1; }
	docker push $(IMAGE)

.PHONY: register
register: ## Register the image with AI Runtime (2-6 min). Uses SECRET_SCOPE/SECRET_KEY if set.
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
image: build size push register ## build + size gate + push + register

# ----------------------------------------------------------------- setup -----
.PHONY: volume
volume: ## Create the UC volume (idempotent)
	databricks volumes create $(UC_CATALOG) $(UC_SCHEMA) $(UC_VOLUME) MANAGED \
	  -p $(AIR_PROFILE) || echo "(volume probably already exists)"

.PHONY: smoke
smoke: ## STEP 0  1xA10 image pre-flight (~2 min)
	$(RUN) air/00_smoke_test.yaml

.PHONY: prep
prep: ## STEP 1  geo3k -> UC volume parquet
	$(RUN) air/01_prep_geo3k.yaml

.PHONY: stage
stage: ## STEP 2  Qwen3.5-35B-A3B (~70 GB) -> UC volume
	$(RUN) air/02_stage_model.yaml

.PHONY: baseline
baseline: ## STEP 3  measure GRPO reward variance before training
	$(RUN) air/03_baseline_eval.yaml

.PHONY: setup
setup: volume smoke prep stage ## volume + smoke + data + model

# ------------------------------------------------------------- the ladder ----
.PHONY: rung1
rung1: ## Qwen3.5-2B  dense  FSDP   8xH100  (cheap full-path check)
	$(RUN) air/10_qwen3_5_2b_fsdp_8gpu.yaml

.PHONY: rung2
rung2: ## Qwen3.5-9B  dense  FSDP   8xH100
	$(RUN) air/11_qwen3_5_9b_fsdp_8gpu.yaml

.PHONY: rung3
rung3: ## Qwen3.5-35B-A3B MoE  CLASSIC+offload  8xH100 (known-good baseline)
	$(RUN) air/20_qwen3_5_35b_classic_8gpu.yaml

.PHONY: rung4
rung4: ## Qwen3.5-35B-A3B MoE  MEGATRON-FSDP no-offload  16xH100  <-- headline
	$(RUN) air/21_qwen3_5_35b_fsdp_16gpu.yaml

# ------------------------------------------------------------------ ops ------
.PHONY: runs
runs: ## List active runs
	$(AIR) list runs --active -p $(AIR_PROFILE)

.PHONY: logs
logs: ## Stream logs: make logs RUN=<run_id> [NODE=0]
	$(AIR) logs $(RUN) $(if $(NODE),--node $(NODE),) -p $(AIR_PROFILE)

.PHONY: cancel
cancel: ## Cancel a run: make cancel RUN=<run_id>
	$(AIR) cancel $(RUN) -p $(AIR_PROFILE)

.PHONY: dry
dry: ## Validate a YAML without submitting: make dry F=air/21_...yaml
	$(AIR) run --dry-run --file $(F) -p $(AIR_PROFILE)

.PHONY: config
config: ## Print the resolved verl overrides locally: make config MODE=fsdp GPUS=16
	@MODE=$${MODE:-fsdp}; GPUS=$${GPUS:-16}; \
	 DRY_RUN=1 MEGATRON_MODE=$$MODE \
	   NUM_NODES=$$(( GPUS / 8 == 0 ? 1 : GPUS / 8 )) LOCAL_WORLD_SIZE=8 \
	   NODE_RANK=0 MASTER_ADDR=127.0.0.1 \
	   bash scripts/run_grpo_megatron.sh 2>/dev/null

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
check: lint validate ## lint + air schema validation

.PHONY: lint
lint: ## Local static checks (shellcheck + python syntax + yaml parse)
	@command -v shellcheck >/dev/null && shellcheck -S warning scripts/*.sh scripts/lib/*.sh || \
	  echo "(shellcheck not installed — skipping)"
	@for f in scripts/*.py scripts/reward/*.py; do python3 -m py_compile "$$f" && echo "py ok  $$f"; done
	@python3 scripts/lint_dockerfile.py
	@python3 -c "import yaml,glob,sys; [yaml.safe_load(open(f)) for f in glob.glob('air/*.yaml')]; print('yaml ok  air/*.yaml')"
