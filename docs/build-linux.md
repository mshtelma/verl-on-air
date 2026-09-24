# Building the image

The image must be `linux/amd64`, so the build needs an x86_64 Linux host. Nothing else does:
data prep, training and evaluation are AI Runtime jobs you can submit from a laptop.

## Quick start

```bash
git clone https://github.com/mshtelma/verl-on-air.git
cd verl-on-air
bash scripts/bootstrap_linux.sh      # or: make bootstrap
```

The bootstrap installs whatever is missing (Docker with buildx, make, `uv`, the `databricks`
and `air` CLIs, the `.venv` from `make dev-env`), logs in to Docker Hub, runs `make doctor`,
then builds, checks the size, pushes and registers. It skips steps that are already done, so
re-running after a failure is safe. `CLEAN=1` builds without the layer cache; `SKIP_INSTALL=1`
installs nothing (tools present, no sudo). `make doctor` alone tells you whether a host can build.

| | requirement | why |
|---|---|---|
| arch | x86_64 | the AI Runtime base images have no arm64 variant |
| Docker | with buildx | the Dockerfile uses BuildKit secrets, bind mounts and a `RUN` heredoc |
| disk | 60 GiB free on Docker's data root, 100 GiB to be comfortable | ~4.7 GB base, ~11 GB of wheels, layer churn |
| network | your PyPI index, `github.com`, `objects.githubusercontent.com` | Python packages including torch; verl's prebuilt wheels and the git-pinned sources |
| auth | `docker login`, `databricks auth login --profile <profile>` | push, then register |

A build takes 15-25 minutes, almost all of it downloads. The image is 17.2 GB; `make size`
fails above 19.5 GB, and the platform rejects anything over 20 GB. TransformerEngine, apex and
flash-attn come prebuilt from verl's wheelhouse, each checked against its sha256 in
`docker/artifacts.lock`, so the build only compiles megatron-core's pybind11 extension and a
small CUDA probe.

## PyPI behind a proxy

The build container does not inherit your pip or uv configuration. On a host that reaches PyPI
through a proxy, a build without it fails in the first minute with
`Failed to fetch: https://pypi.org/simple/pybind11/ ... dns error`.

`make build` finds the index with `scripts/detect_pypi_index.sh`. It looks at
`$PIP_INDEX_URL`, `$UV_INDEX_URL` and `$UV_DEFAULT_INDEX`, then `uv.toml`, then
`pip config get global.index-url`, then the `pip.conf` files.
`bash scripts/detect_pypi_index.sh --source` shows what it picked and where from. If it finds
nothing and `pypi.org` is unreachable, pass it yourself:

```bash
make build PIP_INDEX_URL=https://<your-proxy>/simple
```

The index goes to the build as a BuildKit secret, so neither the URL nor any credentials in it
end up in the image or its history, and `make build` and `make doctor` print it masked. torch
comes from the same index (PyPI's `torch==2.11.0` is the CUDA 13 build; the build fails if
`torch.version.cuda` is not 13.x). If the build then fails on TLS with
`invalid peer certificate: UnknownIssuer`, use `make certs` or `make vendor`
([troubleshooting.md](troubleshooting.md)).

## Building from a Mac

Apple Silicon is arm64, so the build would run under QEMU: hours, with dropped downloads. Use a
remote amd64 builder (or a CI runner such as GitHub Actions `ubuntu-latest`):

```bash
docker buildx create --name amd --driver docker-container \
  --platform linux/amd64 <ssh://user@host  or  tcp://host:2376>
docker buildx use amd
make build
```

## Step by step, without the bootstrap

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" && newgrp docker
sudo systemctl enable --now docker

curl -fsSL https://astral.sh/uv/install.sh | sh
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sudo sh
uv tool install --force databricks-air --python 3.12

docker login
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile <profile>

make doctor
make build
make size          # fails above 19.5 GB, before a registration is wasted
make push
make register      # 2-6 min
```

Run `databricks auth login` even if `~/.databrickscfg` has the profile: recent CLI versions
reject the old token cache, and `air register image` fails until you log in again.

If `make size` fails, `make layers` lists the largest layers. Check that `UV_NO_CACHE=1` took
effect (uv's cache alone is ~11 GB) and that `WITH_VIDEO=0` (the default) kept ffmpeg and
torchcodec out.

## After changing the Dockerfile

```bash
make bump && make release
```

`air register image` caches per tag, so new content under an old tag leaves jobs on the
previously registered digest. `make bump` increments `IMAGE_TAG` in `config.env` and every
custom-image job file. The checks go by content:

| step | what it checks |
|---|---|
| `make build` | labels the image with a hash of its build inputs: `docker/Dockerfile`, `retry.sh`, `uvi.sh`, `cccl_probe.cu`, both lock files, `certs/`, and the build args that change content |
| `make push` | runs `make stale-check`: the local image was built from the current inputs, and the tag was not pushed before from other ones. Then it records the digest in `docker/IMAGE.lock` |
| `make register` | the registry serves the digest `docker/IMAGE.lock` records for the tag |

Commit `docker/IMAGE.lock` after a push; it maps each tag in the job files to one digest.
Repository code is not a build input, because every job uploads it as a `code_source` snapshot.

`make release` rebuilds with `--no-cache --pull`, then runs the size gate, push and register
(20-30 minutes, ~11 GB of downloads). Cached layers are exact for everything pinned (the base
digest, `docker/requirements.lock`, `docker/artifacts.lock`); only the apt packages can differ
between a cached and a fresh build. Once the image is registered, continue with `make smoke` in
[setup.md](setup.md).
