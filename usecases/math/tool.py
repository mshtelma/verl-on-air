#!/usr/bin/env python3
"""Calculator tool for the GSM8K tool-agent rollout.

Registered as a verl ``@function_tool`` named ``calculator``. The rollout loads
it via::

    actor_rollout_ref.rollout.multi_turn.function_tool_path=<this file>

after which it is offered to EVERY ``tool_agent`` sample. Function tools are
global and stateless: per-sample ``tools_kwargs`` are deliberately ignored for
them (see verl/experimental/agent_loop/tool_agent_loop.py:510-514), so no
dataset-side plumbing is required — a row only needs ``agent_name="tool_agent"``.

WHY a calculator (and not verl's shipped ``calc_gsm8k_reward``): that shipped
tool is an answer-checking ORACLE — it is handed the ground truth via
``create_kwargs`` and tells the model its current reward. That leaks the label
into the rollout and is nonsensical to pair with an LLM judge. A calculator is
an honest tool: it offloads arithmetic (where LLMs slip) without revealing the
answer, and gives us observable, un-gamed agentic signal — tool-call rate and
tool-error rate — to watch in MLflow.

Evaluation is a whitelisted AST walk (NO eval/exec, no attribute access, no
names beyond a small math allowlist), so a hostile or confused model cannot use
it to run arbitrary code inside the rollout worker.

Contract (verl/tools/function_tool.py): the function MUST carry a Google-style
docstring with an ``Args:`` block and a type hint on every parameter — the
OpenAI tool schema is inferred from them at registration time. A ``str`` return
is normalised to a text ToolResponse.
"""

from __future__ import annotations

import ast
import math
import operator

# --- verl decorator, with a no-op shim so this file is importable/testable
#     outside the verl runtime (e.g. `python usecases/math/tool.py`). ------
try:
    from verl.tools.function_tool import function_tool
except Exception:  # pragma: no cover - only taken in local unit testing

    def function_tool(name=None, *, schema=None):
        def deco(fn):
            return fn

        return deco(name) if callable(name) else deco


# Whitelisted binary and unary operators.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Whitelisted callables and constants. No builtins, no attribute access.
_CALLABLES = {
    **{n: getattr(math, n) for n in ("sqrt", "floor", "ceil", "log", "log2", "log10", "exp", "gcd", "factorial", "fabs")},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_MAX_ABS_BASE = 1e12   # guard against gigantic ** blow-ups (DoS the worker)
_MAX_ABS_EXP = 64


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"disallowed constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed operator {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if op is operator.pow and (abs(left) > _MAX_ABS_BASE or abs(right) > _MAX_ABS_EXP):
            raise ValueError("exponent out of allowed range")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed unary operator {type(node.op).__name__}")
        return op(_eval(node.operand))
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name) or node.func.id not in _CALLABLES:
            raise ValueError("only positional calls to allowlisted functions are permitted")
        return _CALLABLES[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name):
        if node.id not in _CONSTS:
            raise ValueError(f"unknown name {node.id!r} (variables are not allowed)")
        return _CONSTS[node.id]
    raise ValueError(f"disallowed syntax: {type(node).__name__}")


def evaluate(expression: str) -> str:
    """Pure, verl-free core so the logic is unit-testable standalone."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
    except Exception as e:  # noqa: BLE001 - surface any parse/eval error to the model as text
        return f"Error: could not evaluate {expression!r}: {e}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


@function_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the exact numeric result.

    Use this for ANY non-trivial arithmetic so you never make a calculation
    mistake. Supported: + - * / // % ** and parentheses; the functions sqrt,
    floor, ceil, log, log2, log10, exp, abs, round, min, max, gcd, factorial,
    fabs; and the constants pi, e, tau. Variables, assignments, comparisons and
    function definitions are NOT allowed — pass a single arithmetic expression.

    Args:
        expression: One arithmetic expression to evaluate, e.g.
            "3 * (17 + 4) / 2" or "sqrt(144) + factorial(5)".
    """
    return evaluate(expression)


# ---------------------------------------------------------------------------
# Local sanity check:  python3 usecases/math/tool.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("3 * (17 + 4) / 2", "31.5"),
        ("sqrt(144) + 5", "17"),
        ("factorial(5)", "120"),
        ("2 ** 10", "1024"),
        ("10 // 3", "3"),
        ("min(4, 9) + max(1, 2)", "6"),
        ("__import__('os').system('echo hi')", None),  # must be rejected
        ("x + 1", None),                                # variables rejected
        ("2 ** 999999", None),                          # blow-up guarded
    ]
    ok = True
    for expr, want in cases:
        got = evaluate(expr)
        rejected = got.startswith("Error:")
        if want is None:
            status = "PASS" if rejected else "FAIL"
            ok &= rejected
        else:
            status = "PASS" if got == want else "FAIL"
            ok &= got == want
        print(f"[{status}] calculator({expr!r}) -> {got}")
    raise SystemExit(0 if ok else 1)
