#!/usr/bin/env python3
"""Cross-node HTTP reachability probe for AI Runtime multi-node jobs.

DE-RISKS the agentic-run topology: the LLM-judge will serve an OpenAI
HTTP endpoint on ONE node (:8000) and the training reward workers on OTHER nodes
must reach it over HTTP. We already KNOW cross-node NCCL + the Ray head's TCP
port (:6379) work, but plain HTTP to a NON-head node's :8000 is untested
and the whole judge topology hinges on it.

This mirrors the judge mechanism exactly and cheaply:
  * every node starts a tiny stand-in HTTP server on :8000 that answers the two
    calls judge_reward.py makes -- GET /health and POST /v1/chat/completions
    (returning a judge-style JSON body);
  * every node publishes its own address to a shared UC rendezvous dir -- the
    SAME cross-node channel serve_judge.sh's JUDGE_RENDEZVOUS uses;
  * every node then fetches EVERY peer (all-to-all, so we cover head->worker,
    worker->head and worker->worker regardless of where reward actors land);
  * rank 0 collects all nodes' results and prints a full reachability matrix,
    exiting non-zero unless every ordered pair is reachable.

No GPU work, stdlib only -> runs on the cheapest 2x GPU_1xA10 multi-node job.
Injected by AI Runtime (see engine/lib/ray_cluster.sh): NUM_NODES, WORLD_SIZE,
POD_RANK (also NODE_RANK), LOCAL_ADDR, MASTER_ADDR, MASTER_PORT.
"""
import http.server
import json
import os
import socket
import sys
import threading
import time
import urllib.request

RANK = int(os.environ.get("POD_RANK", os.environ.get("NODE_RANK", "0")))
NNODES = int(os.environ.get("NUM_NODES", os.environ.get("NNODES", "1")))
PORT = int(os.environ.get("PROBE_PORT", "8000"))
RDV_ROOT = os.environ.get("PROBE_RENDEZVOUS_DIR", "/Volumes/main/mshtelma/verl/data/xnode_probe")
SERVE_SECONDS = int(os.environ.get("PROBE_SERVE_SECONDS", "150"))
WAIT_SECONDS = int(os.environ.get("PROBE_WAIT_SECONDS", "120"))
EXPECT_NODES = int(os.environ.get("PROBE_EXPECT_NODES", "0"))  # if set, assert NNODES matches

# All nodes agree on this token (they share MASTER_ADDR/PORT), so a run gets its
# own rendezvous subdir and stale files from prior runs are ignored.
TOKEN = f"{os.environ.get('MASTER_ADDR', 'x')}_{os.environ.get('MASTER_PORT', '0')}".replace("/", "_")
RDV = os.path.join(RDV_ROOT, TOKEN)


def local_addr() -> str:
    a = os.environ.get("LOCAL_ADDR")
    if a:
        return a.strip()
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


ADDR = local_addr()


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("health"):
            self._send(200, b"ok", "text/plain")
        else:
            self._send(200, json.dumps({"object": "list", "data": [{"id": "probe"}]}))

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n:
                self.rfile.read(n)
        except Exception:
            pass
        content = json.dumps({"correct": True, "score": 1.0, "reason": "probe"})
        self._send(200, json.dumps({"choices": [{"message": {"content": content}}]}))

    def log_message(self, *a):  # silence per-request logging
        pass


def serve():
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


def fetch(url, method="GET", timeout=10):
    data = None
    if method == "POST":
        data = json.dumps({"model": "probe", "messages": [{"role": "user", "content": "ping"}]}).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return round((time.time() - t0) * 1000), body


def probe_peer(rank, addr):
    base = f"http://{addr}:{PORT}"
    e = {"addr": addr}
    try:
        e["health_ms"], _ = fetch(base + "/health")
        e["health"] = True
    except Exception as ex:
        e["health"] = False
        e["health_err"] = repr(ex)[:200]
    try:
        e["post_ms"], body = fetch(base + "/v1/chat/completions", method="POST")
        e["post"] = "choices" in body
    except Exception as ex:
        e["post"] = False
        e["post_err"] = repr(ex)[:200]
    e["ok"] = bool(e.get("health") and e.get("post"))
    return e


def main():
    print(f"[rank {RANK}/{NNODES}] addr={ADDR} port={PORT} token={TOKEN}", flush=True)
    print(
        f"[rank {RANK}] env: LOCAL_ADDR={os.environ.get('LOCAL_ADDR')} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')} "
        f"NUM_NODES={os.environ.get('NUM_NODES')} WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"POD_RANK={os.environ.get('POD_RANK')} NODE_RANK={os.environ.get('NODE_RANK')}",
        flush=True,
    )
    if EXPECT_NODES and NNODES < EXPECT_NODES:
        print(
            f"[rank {RANK}] FATAL: expected >= {EXPECT_NODES} nodes but NUM_NODES resolved to {NNODES}. "
            "The runtime did not inject the multi-node vars this script expects; "
            "inspect the env dump above for the right names.",
            flush=True,
        )
        sys.exit(2)

    os.makedirs(RDV, exist_ok=True)
    threading.Thread(target=serve, daemon=True).start()
    time.sleep(2)

    with open(os.path.join(RDV, f"node_{RANK}.addr"), "w") as f:
        f.write(ADDR)
    print(f"[rank {RANK}] published address -> {RDV}/node_{RANK}.addr", flush=True)

    # Wait for every node to publish its address.
    peers = {}
    deadline = time.time() + WAIT_SECONDS
    while time.time() < deadline:
        got = {}
        for r in range(NNODES):
            p = os.path.join(RDV, f"node_{r}.addr")
            if os.path.exists(p):
                try:
                    got[r] = open(p).read().strip()
                except Exception:
                    pass
        if len(got) >= NNODES:
            peers = got
            break
        time.sleep(3)
    if len(peers) < NNODES:
        print(f"[rank {RANK}] WARN: only saw {len(peers)}/{NNODES} peer addrs via rendezvous", flush=True)

    # Probe every peer (including self, as a control).
    results = {}
    for r, addr in sorted(peers.items()):
        results[r] = probe_peer(r, addr)
        print(f"[rank {RANK}] -> peer {r} ({addr}): {results[r]}", flush=True)

    all_ok = len(results) >= NNODES and all(v.get("ok") for v in results.values())
    with open(os.path.join(RDV, f"node_{RANK}.result"), "w") as f:
        json.dump({"rank": RANK, "all_ok": all_ok, "results": results}, f)

    if RANK != 0:
        # Keep serving so peers can still reach me, then exit cleanly.
        time.sleep(SERVE_SECONDS)
        print(f"[rank {RANK}] worker done (self all_ok={all_ok})", flush=True)
        sys.exit(0)

    # Rank 0 aggregates every node's result into a full matrix.
    agg = {}
    deadline = time.time() + SERVE_SECONDS
    while time.time() < deadline:
        agg = {}
        for r in range(NNODES):
            p = os.path.join(RDV, f"node_{r}.result")
            if os.path.exists(p):
                try:
                    agg[r] = json.load(open(p))
                except Exception:
                    pass
        if len(agg) >= NNODES:
            break
        time.sleep(3)

    print("\n================ CROSS-NODE HTTP REACHABILITY MATRIX ================", flush=True)
    matrix_ok = len(agg) >= NNODES
    for r in range(NNODES):
        node = agg.get(r)
        if not node:
            print(f"  rank {r}: <no result reported>", flush=True)
            matrix_ok = False
            continue
        for tgt_s, e in sorted(node.get("results", {}).items(), key=lambda kv: int(kv[0])):
            flag = "OK  " if e.get("ok") else "FAIL"
            err = (e.get("health_err", "") or "") + (e.get("post_err", "") or "")
            print(
                f"  {r} -> {int(tgt_s)}: {flag}  health={e.get('health')}({e.get('health_ms', '?')}ms) "
                f"post={e.get('post')}({e.get('post_ms', '?')}ms) {err}",
                flush=True,
            )
            if not e.get("ok"):
                matrix_ok = False
    print("====================================================================", flush=True)
    print(f"RESULT: cross-node HTTP {'REACHABLE (all pairs OK)' if matrix_ok else 'NOT fully reachable'}", flush=True)
    sys.exit(0 if matrix_ok else 1)


if __name__ == "__main__":
    main()
