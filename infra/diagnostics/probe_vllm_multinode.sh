#!/usr/bin/env bash
# =============================================================================
# Cheap 1xA10 probe to pick the multi-node vLLM serving fix for the GLM-5.3
# judge, after air/52c hit vllm#45318 (ActorHandleNotFoundError: our image's
# ray 2.58.0 is too new for vLLM 0.24's ray executor; vLLM CI pins ray 2.48.0).
#
# Determines, without burning 16xH100:
#   (A) does vLLM 0.24 support NATIVE multiprocessing multi-node (--headless /
#       --nnodes / --node-rank), which avoids Ray (and the bug) entirely?
#   (B) is PyPI reachable from a df1 job, so we could runtime-downgrade ray to
#       2.48.0 on the judge nodes only (training keeps 2.58)?
# =============================================================================
set -uo pipefail
export OPENSSL_FORCE_FIPS_MODE=0 OPENSSL_FIPS=0

echo "=== versions ==="
python3 -c "import ray, vllm; print('ray', ray.__version__, '| vllm', vllm.__version__)" 2>&1 || true

echo "=== vLLM's declared 'ray' dependency constraint ==="
python3 -c "import importlib.metadata as m; print([r for r in (m.requires('vllm') or []) if 'ray' in r.lower()])" 2>&1 || true

echo "=== vLLM 'serve' multi-node / distributed flags present in 0.24 ==="
vllm serve --help 2>&1 | grep -iE -- '--headless|--nnodes|--node-rank|--data-parallel|--distributed-executor-backend|--pipeline-parallel-size|--tensor-parallel-size' | sed 's/^[[:space:]]*/  /' | head -60 || true

echo "=== VERDICT: native mp multi-node available? ==="
if vllm serve --help 2>&1 | grep -qiE -- '--headless'; then
    echo "  YES -- '--headless' present => vLLM native mp multi-node (no Ray) is an option (fix A)"
else
    echo "  NO  -- '--headless' absent  => must use ray executor; need the ray downgrade (fix B)"
fi

echo "=== PyPI reachable for a ray downgrade (fix B)? ==="
if timeout 90 pip download 'ray==2.48.0' --no-deps -d /tmp/raydl >/tmp/pipout 2>&1; then
    echo "  PyPI OK -- ray==2.48.0 wheel fetched:"; ls -1 /tmp/raydl 2>/dev/null | head
else
    echo "  PyPI FAIL (last lines):"; tail -6 /tmp/pipout
fi
echo "=== done ==="
