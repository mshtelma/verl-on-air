"""R27: the probes the docs use as gates are gates -- non-zero unless their claim holds, with a
machine-readable verdict -- and none of them changes the environment it checks.

Reviewer reproductions, each a test below:
  * AutoBridge resolution was optional (a warning) although the docs call it the FSDP gate.
  * probe_tool_format exited 0 on UNEXPECTED / INCONCLUSIVE, and its success flag could come from
    the hand-written canonical XML sample even when the template-rendered call did not parse.
  * the HTTP aggregator passed when every rank wrote a result file with an EMPTY peer map, and
    ignored each node's own all_ok.
  * probe_vs_access returned 0 after auth / query failures and pip-installed packages.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from support import FakeTokenizer, REPO, env, load_module, pinned_verl, run

DIAG = REPO / "infra" / "diagnostics"
pv = load_module(DIAG / "probe_verdict.py")


# --- the verdict line ---------------------------------------------------------------------------
def test_a_verdict_is_one_json_line_a_file_and_an_exit_code(tmp_path: Path, capsys):
    out = tmp_path / "v" / "verdict.json"
    with env(PROBE_VERDICT_OUT=str(out)):
        assert pv.emit("p", False, reasons=["x"], n=3) == 1
        assert pv.emit("p", True) == 0
    log = capsys.readouterr().out
    first = pv.parse(log.splitlines()[0])
    assert first["ok"] is False and first["status"] == "FAIL" and first["reasons"] == ["x"] and first["n"] == 3
    assert pv.parse(log)["status"] == "PASS", "parse() returns the LAST verdict"
    assert json.loads(out.read_text())["ok"] is True
    assert pv.parse("no verdict here") is None


# --- cross-node HTTP aggregator -------------------------------------------------------------------
http = load_module(DIAG / "probe_cross_node_http.py")


def _node(rank: int, n: int, *, ok=True, all_ok=None, drop=()):
    res = {str(t): {"ok": ok, "health": ok, "post": ok} for t in range(n) if t not in drop}
    return {"rank": rank, "all_ok": ok if all_ok is None else all_ok, "results": res}


def test_a_full_reachable_matrix_passes():
    ok, lines, reasons = http.aggregate({r: _node(r, 3) for r in range(3)}, 3)
    assert ok and not reasons and len(lines) == 9


def test_empty_peer_maps_fail_the_gate():
    # the reviewer's case: every rank reported, none probed anything
    agg = {r: {"rank": r, "all_ok": True, "results": {}} for r in range(2)}
    ok, _, reasons = http.aggregate(agg, 2)
    assert not ok and sum("never probed" in r for r in reasons) == 4


@pytest.mark.parametrize("agg,why", [
    ({0: _node(0, 2), 1: _node(1, 2, drop=(0,))}, "1 -> 0 was never probed"),
    ({0: _node(0, 2), 1: _node(1, 2, all_ok=False)}, "rank 1 reported all_ok=False"),
    ({0: _node(0, 2)}, "rank 1 reported no result"),
    ({0: _node(0, 2), 1: _node(1, 2, ok=False, all_ok=True)}, "1 -> 0 unreachable"),
])
def test_any_missing_cell_failed_cell_or_node_verdict_fails(agg, why):
    ok, _, reasons = http.aggregate(agg, 2)
    assert not ok and why in reasons, reasons


# --- tool-format probe ----------------------------------------------------------------------------
tf = load_module(DIAG / "probe_tool_format.py")


class TemplateTok(FakeTokenizer):
    """A chat template that writes an assistant tool call as Qwen XML or as hermes JSON."""

    def __init__(self, style: str):
        super().__init__()
        self.style = style

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, **kw):
        out = "".join(f"<|{m['role']}|>{m.get('content', '')}\n" for m in messages if not m.get("tool_calls"))
        for m in messages:
            for c in m.get("tool_calls") or []:
                fn, args = c["function"]["name"], c["function"]["arguments"]
                args = json.loads(args) if isinstance(args, str) else args
                if self.style == "xml":
                    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
                    out += f"<tool_call>\n<function={fn}>\n{params}</function>\n</tool_call>"
                else:
                    out += f"<tool_call>\n{json.dumps({'name': fn, 'arguments': args})}\n</tool_call>"
        return out


def _probe(style: str, fmt: str = "qwen3_coder", capsys=None) -> tuple[int, dict]:
    with pinned_verl():
        tf.TOOL_FORMAT = fmt
        rc = asyncio.run(tf.run(TemplateTok(style)))
    return rc, pv.parse(capsys.readouterr().out)


def test_xml_template_with_qwen3_coder_passes(capsys):
    rc, v = _probe("xml", capsys=capsys)
    assert rc == 0 and v["status"] == "PASS" and v["detected_format"] == "QWEN_XML"
    assert v["template_calls"] == [["calculator", {"expression": "18 - 3 - 4"}]]
    assert v["informational"]["template_via_hermes"] == [], "hermes cannot read XML (the run3 bug)"


def test_the_canonical_sample_never_supplies_the_pass(capsys):
    # the template writes hermes JSON; the canonical XML still parses under qwen3_coder, which
    # used to set the success flag -- the gate must fail, and non-zero
    rc, v = _probe("json", capsys=capsys)
    assert rc == 1 and v["status"] == "FAIL" and v["detected_format"] == "HERMES_JSON"
    assert v["informational"]["canonical_xml"] == [["calculator", {"expression": "18 - 3 - 4"}]]


def test_the_matching_parser_passes_for_a_json_template(capsys):
    rc, v = _probe("json", fmt="hermes", capsys=capsys)
    assert rc == 0 and v["status"] == "PASS" and v["tool_format"] == "hermes"


@pytest.mark.parametrize("calls,status", [
    (None, "INCONCLUSIVE"), ([], "FAIL"), ([("calculator", {"expression": "18-3"})], "FAIL"),
    ([("calculator", {"expression": "18 - 3 - 4"})] * 2, "FAIL"),
    ([("calculator", {"expression": "18 - 3 - 4"})], "PASS"),
])
def test_decide_passes_only_the_exact_template_call(calls, status):
    ok, got, reasons = tf.decide("qwen3_coder", "QWEN_XML", calls)
    assert got == status and ok == (status == "PASS") and (ok or reasons)


def test_a_probe_that_cannot_run_fails(tmp_path: Path):
    r = run(["python3", str(DIAG / "probe_tool_format.py")],
            env={"MODEL_PATH": str(tmp_path / "no-such-model"), "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"})
    v = pv.parse(r.stdout)
    assert r.returncode == 1 and v["status"] == "ERROR", r.stdout[-600:]


# --- smoke test: AutoBridge is the FSDP gate --------------------------------------------------------
@pytest.mark.parametrize("require,where", [(None, "reasons"), ("0", "warnings")])
def test_autobridge_is_required_unless_explicitly_demoted(tmp_path: Path, require, where):
    e = {"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "SMOKE_MODEL_ID": str(tmp_path / "nope")}
    if require is not None:
        e["SMOKE_REQUIRE_BRIDGE"] = require
    r = run(["python3", str(DIAG / "smoke_test.py")], env=e, timeout=300)
    v = pv.parse(r.stdout)
    assert v is not None, r.stdout[-800:]
    assert "AutoBridge resolves the model" in v[where], v
    assert v["bridge_required"] is (require is None)
    assert r.returncode == 1 and v["ok"] is False   # this CPU box fails the GPU checks either way


# --- Vector Search access -------------------------------------------------------------------------
vs = load_module(REPO / "usecases" / "agentic-search" / "probe_vs_access.py")
GOOD = {"manifest": {"columns": [{"name": "id"}, {"name": "title"}, {"name": "text"}]},
        "result": {"data_array": [["1", "Inception", "Inception is a 2010 film by Christopher Nolan."]]}}


class FakeClient:
    def __init__(self, reply):
        self.calls, self._reply = [], reply
        self.config = type("C", (), {"host": "https://example.cloud.databricks.com"})()
        self.api_client = self

    def do(self, method, path, body=None):
        self.calls.append((method, path, body["query_type"]))
        return self._reply(body)


def _vs(capsys, factory, index="main.x.idx") -> tuple[int, dict]:
    with env(QA_VS_INDEX=index):
        rc = vs.main([], client_factory=factory)
    return rc, pv.parse(capsys.readouterr().out)


def test_vs_access_passes_on_both_query_types(capsys):
    c = FakeClient(lambda b: GOOD)
    rc, v = _vs(capsys, lambda: c)
    assert rc == 0 and v["ok"] and [q for *_, q in c.calls] == ["ANN", "HYBRID"]
    assert v["versions"].keys() == {"protobuf", "databricks-sdk", "databricks-vectorsearch"}


@pytest.mark.parametrize("factory,why", [
    (lambda: (_ for _ in ()).throw(ValueError("default auth: cannot configure")), "auth: ValueError"),
    (lambda: FakeClient(lambda b: (_ for _ in ()).throw(PermissionError("403 PERMISSION_DENIED"))),
     "ANN query: PermissionError"),
    (lambda: FakeClient(lambda b: {"manifest": GOOD["manifest"], "result": {"data_array": []}}),
     "ANN query returned no rows"),
    (lambda: FakeClient(lambda b: GOOD if b["query_type"] == "ANN" else {"result": {}}),
     "HYBRID query returned no rows"),
])
def test_vs_access_fails_on_auth_query_and_shape_errors(capsys, factory, why):
    rc, v = _vs(capsys, factory)
    assert rc == 1 and not v["ok"] and any(why in r for r in v["reasons"]), v["reasons"]


def test_vs_access_needs_an_index_and_never_installs_anything(capsys):
    rc, v = _vs(capsys, lambda: FakeClient(lambda b: GOOD), index="")
    assert rc == 1 and v["reasons"] == ["QA_VS_INDEX is not set"]
    src = (REPO / "usecases" / "agentic-search" / "probe_vs_access.py").read_text()
    assert "subprocess" not in src and '"install"' not in src and "pip install" not in src.split('"""', 2)[2]
