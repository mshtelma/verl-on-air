# Building the image on a Linux box

The image **must** be `linux/amd64`. Everything else in this repo runs fine from
a laptop — only the Docker build needs a specific host.

## TL;DR

```bash
git clone https://github.com/mshtelma/verl-on-air.git
cd verl-on-air
bash scripts/bootstrap_linux.sh      # or: make bootstrap
```

That installs Docker / `uv` / the `databricks` and `air` CLIs if missing, runs
preflight, then builds → size-gates → pushes → registers.

Check first whether a box is suitable at all:

```bash
make doctor
```

## If your box uses an internal PyPI proxy

Locked-down hosts (Databricks corp boxes included) cannot reach `pypi.org` and go
through a proxy instead. **The build container does not inherit your pip config**,
so this must be passed in explicitly — it is the single most likely reason a build
fails in the first 45 seconds:

```
error: Failed to fetch: `https://pypi.org/simple/pybind11/`
  Caused by: dns error: failed to lookup address information
```

`make build` detects the index automatically via
`scripts/detect_pypi_index.sh`, which checks, in order:

1. `$PIP_INDEX_URL`, `$UV_INDEX_URL`, `$UV_DEFAULT_INDEX`
2. `~/.config/uv/uv.toml`, `/etc/uv/uv.toml`
3. `pip config get global.index-url` (via `python3 -m pip`, `pip3`, `pip`)
4. `~/.config/pip/pip.conf`, `~/.pip/pip.conf`, `/etc/pip.conf`

Confirm what it found:

```bash
bash scripts/detect_pypi_index.sh --source
# https://pypi-proxy.dev.databricks.com/simple    /home/you/.config/uv/uv.toml
```

If that prints nothing but `pypi.org` is unreachable, pass it yourself:

```bash
export PIP_INDEX_URL=https://<your-proxy>/simple      # or per-build:
make build PIP_INDEX_URL=https://<your-proxy>/simple
```

It is passed as a build **ARG, never `ENV`** — the proxy is a build-time concern
and the training nodes have entirely different egress.

Hosts the build needs (`make doctor` probes all of them **from inside a
container**, since host resolution proves nothing):

| host | why |
|---|---|
| your PyPI index | every Python package, including torch |
| `github.com`, `objects.githubusercontent.com` | verl wheelhouse binaries (TE, apex, flash-attn) + git-sourced `megatron-core`/`mbridge` |

`download.pytorch.org` is **not** needed: PyPI's own `torch==2.11.0` is already
the CUDA 13 build (its metadata requires `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`,
…), which is the ABI the wheelhouse binaries were compiled against. The build
hard-fails if `torch.version.cuda` is not `13.x`.

## Requirements

| | requirement | why |
|---|---|---|
| arch | **x86_64** | the image is `linux/amd64`; the AI Runtime base images publish no arm64 variant |
| OS | any modern Linux | Docker with BuildKit |
| Docker | with **buildx** | the Dockerfile declares `# syntax=docker/dockerfile:1.7` for its `RUN` heredoc |
| disk | **≥ 60 GiB** free on Docker's data root, 100 GiB comfortable | ~4.7 GB base + ~11 GB of wheels + layer churn |
| network | good egress to PyPI, `download.pytorch.org`, GitHub releases | torch ~4 GB, TransformerEngine ~1.5 GB |
| auth | `docker login`; `databricks auth login --profile df1` | push, then register |

Build time on a decent native box: **~15–25 min**, almost all of it downloads.
**Measured image size: 15.95 GB** (gate 19.5 GB, DCS hard limit 20 GB).
Nothing CUDA compiles — TransformerEngine, apex and flash-attn come prebuilt from
verl's wheelhouse.

## Why not just build on a Mac?

Apple Silicon is arm64, so `--platform linux/amd64` runs the whole build under
QEMU. Two consequences:

1. **Slow.** Every `pip install` is emulated. Hours, not minutes.
2. **Flaky.** The upstream reference implementation documents QEMU dropping
   mid-stream on large wheels, which is why its Dockerfile carries retry loops —
   and why ours makes retry exhaustion a *hard* build failure rather than letting
   a half-installed image through.

If you have no Linux box, prefer a remote builder over local emulation:

```bash
docker buildx create --name amd --driver docker-container \
  --platform linux/amd64 <ssh://user@host  or  tcp://host:2376>
docker buildx use amd
make build
```

A cloud build service (Depot, GitHub Actions `ubuntu-latest`, etc.) works too —
it just needs Docker Hub credentials and enough disk.

## Step by step (if you prefer not to use the bootstrap)

```bash
# 1. Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" && newgrp docker
sudo systemctl enable --now docker

# 2. CLIs
curl -fsSL https://astral.sh/uv/install.sh | sh
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sudo sh
uv tool install --force databricks-air --python 3.12

# 3. Auth
docker login
databricks auth login --host https://dbc-559ffd80-2bfc.cloud.databricks.com --profile df1

# 4. Preflight, then build
make doctor
make build
make size          # HARD GATE: fails above 19.5 GB, before wasting a registration
make push
make register      # 2-6 min, blocks
```

> A pre-existing `~/.databrickscfg` is **not** sufficient. Recent CLI versions
> reject the old token cache with *"stored credentials from older CLI versions
> are no longer used"*, and `air` shares that auth — so `air register image`
> fails until you re-run `databricks auth login`.

## The size gate matters

AI Runtime DCS rejects images over **20 GB** — registration hangs, then times out
after several minutes. `make size` fails locally at 19.5 GB so you find out in
one second instead.

If it trips:

```bash
make layers        # biggest layers first
```

Levers, cheapest first:

1. confirm `UV_NO_CACHE=1` took effect — uv's download cache is ~11 GB on its own
   and previously pushed a comparable image to 31 GB
2. `--build-arg WITH_VIDEO=0` (default) keeps ffmpeg + torchcodec out
3. drop `nvidia-modelopt` if Megatron-Bridge tolerates its absence

## What you do *not* need this box for

Only the image build is host-constrained. Steps 01 and 02 deliberately run on a
**stock** AI Runtime environment, so data prep and the 67 GiB model stage are
independent of Docker and can be driven from any machine with the `air` CLI:

```bash
make volume prep stage      # already done on df1
```

Also host-independent: `make check` (lint + YAML validation against the live CLI)
and every `make rung*` submission once the image is registered.

## Changed `DOCKERHUB_USER` after building?

No rebuild needed — the image contents are identical, only the tag differs:

```bash
make retag      # re-tags the existing local image to the new user
make push register
```

`make build` would also work (all layers cache-hit, seconds), but `make retag` is
explicit about the fact that nothing is being rebuilt.

## After the build

```bash
make smoke     # 1xA10, ~2 min. Read: cpu ram, EFA devices, AutoBridge, Python.h
make rung1     # Qwen3.5-2B  dense  FSDP  8xH100
make rung4     # 35B-A3B MoE  Megatron-FSDP  16xH100   <- headline
```

`make smoke` is the highest-value two minutes available: it reports host CPU RAM
(which decides whether `OFFLOAD=1` is viable), whether EFA is present, and
whether Megatron-Bridge can resolve Qwen3.5 MoE — the one genuinely unproven link
in the stack, since Megatron-FSDP + Qwen3.5 MoE is not an upstream-tested combo.
