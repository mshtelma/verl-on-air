#!/usr/bin/env bash
# =============================================================================
# Can we actually PUSH to <user>/<repo> on Docker Hub?
#
#   bash scripts/check_dockerhub_push.sh <dockerhub_user> <repo>
#
# Why this exists: doctor.sh used to report "Docker Hub credentials present"
# merely because ~/.docker/config.json had an index.docker.io entry. A 16 GB
# build then completed and `docker push` failed with
#   denied: requested access to the resource is denied
# Presence of a credential says nothing about WHICH account it is or whether it
# has write scope.
#
# Method: ask Docker Hub's auth service for a token scoped to
# `repository:<user>/<repo>:push,pull` and inspect the scopes actually GRANTED in
# the returned JWT. Cheap, read-only, and definitive -- no 16 GB upload required.
#
# Exit 0 = push granted. Exit 1 = not granted (reason printed).
# =============================================================================
set -uo pipefail

USER_NAME="${1:?usage: check_dockerhub_push.sh <user> <repo>}"
REPO="${2:?usage: check_dockerhub_push.sh <user> <repo>}"
CFG="${HOME}/.docker/config.json"

# ---- recover credentials, whether stored inline or via a credential helper ----
creds_json=""
if [ -f "${CFG}" ]; then
    helper=$(python3 -c "
import json,sys
try: c=json.load(open('${CFG}'))
except Exception: sys.exit()
print(c.get('credsStore') or (c.get('credHelpers') or {}).get('https://index.docker.io/v1/','') or '')
" 2>/dev/null)
    if [ -n "${helper}" ] && command -v "docker-credential-${helper}" >/dev/null 2>&1; then
        creds_json=$(echo "https://index.docker.io/v1/" \
            | "docker-credential-${helper}" get 2>/dev/null || true)
    fi
fi

USER_FROM_CFG=""; SECRET=""
if [ -n "${creds_json}" ]; then
    USER_FROM_CFG=$(printf '%s' "${creds_json}" | python3 -c "import json,sys;print(json.load(sys.stdin).get('Username',''))" 2>/dev/null)
    SECRET=$(printf '%s' "${creds_json}" | python3 -c "import json,sys;print(json.load(sys.stdin).get('Secret',''))" 2>/dev/null)
elif [ -f "${CFG}" ]; then
    pair=$(python3 -c "
import base64, json, sys
try: c=json.load(open('${CFG}'))
except Exception: sys.exit()
a=(c.get('auths') or {})
for k in ('https://index.docker.io/v1/','index.docker.io','registry-1.docker.io'):
    e=a.get(k) or {}
    if e.get('auth'):
        try: print(base64.b64decode(e['auth']).decode()); break
        except Exception: pass
" 2>/dev/null)
    USER_FROM_CFG=${pair%%:*}
    SECRET=${pair#*:}
fi

if [ -z "${SECRET}" ]; then
    echo "no usable Docker Hub credential found (checked ${CFG} and credential helpers)"
    echo "  fix: docker login -u ${USER_NAME}"
    exit 1
fi

if [ -n "${USER_FROM_CFG}" ] && [ "${USER_FROM_CFG}" != "${USER_NAME}" ]; then
    echo "logged in as '${USER_FROM_CFG}' but the image targets '${USER_NAME}'"
    echo "  fix: docker logout && docker login -u ${USER_NAME}"
    echo "  or:  set DOCKERHUB_USER=${USER_FROM_CFG} in config.env"
    exit 1
fi

# ---- ask for a push-scoped token and read back the GRANTED scopes ------------
granted=$(curl -sS --max-time 20 \
    -u "${USER_FROM_CFG:-$USER_NAME}:${SECRET}" \
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${USER_NAME}/${REPO}:push,pull" \
    2>/dev/null | python3 -c "
import base64, json, sys
try:
    tok = json.load(sys.stdin).get('token','')
except Exception:
    sys.exit()
if not tok: sys.exit()
try:
    payload = tok.split('.')[1]
    payload += '=' * (-len(payload) % 4)
    claims = json.loads(base64.urlsafe_b64decode(payload))
except Exception:
    sys.exit()
for a in claims.get('access', []):
    print(f\"{a.get('name')} {','.join(a.get('actions', []))}\")
" 2>/dev/null)

if [ -z "${granted}" ]; then
    echo "could not obtain a Docker Hub token (bad/expired credential?)"
    echo "  fix: docker login -u ${USER_NAME}   # use a PAT with Read & Write"
    exit 1
fi

if printf '%s' "${granted}" | grep -q "push"; then
    echo "push granted for ${USER_NAME}/${REPO} (as ${USER_FROM_CFG:-$USER_NAME})"
    exit 0
fi

echo "push NOT granted for ${USER_NAME}/${REPO} (as ${USER_FROM_CFG:-$USER_NAME}); granted: ${granted}"
echo "  likely causes:"
echo "    * the PAT is Read-only  -> create one with Read & Write at"
echo "      https://app.docker.io/settings/personal-access-tokens, then docker login -u ${USER_NAME}"
echo "    * the repo belongs to another account/org"
echo "    * private-repo quota exhausted on a free plan"
exit 1
