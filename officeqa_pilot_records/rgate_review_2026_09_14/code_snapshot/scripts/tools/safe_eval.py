#!/usr/bin/env python3
"""Decorator-free safe arithmetic evaluator (canonical copy).

`evaluate(expr) -> str` is a whitelisted AST walk (NO eval/exec, no attribute
access, no names beyond a small math allowlist), so a hostile or confused model
cannot run arbitrary code inside a rollout/eval worker.

WHY this module exists separately from calc_tool.py: verl's ``@function_tool``
registers tools in a GLOBAL registry keyed by NAME and REFUSES to register the
same name twice in one process. calc_tool.py registers ``calculator``; so does
officeqa_tools.py. If officeqa_tools imported calc_tool (to reuse its evaluate),
importing officeqa_tools would run calc_tool's decorator first and then collide
on the second ``calculator`` registration (observed: air run 1118281935002796
"Function tool 'calculator' is already registered"). Keeping the pure evaluator
here — with NO decorator — lets any tool module import the logic without dragging
in another module's tool registration. (calc_tool.py keeps its own inline copy so
the validated MATH pipeline is untouched; this is the canonical one going forward.)
"""

from __future__ import annotations

import ast
import math
import operator

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
    """Evaluate one arithmetic expression; return the exact result or an Error: str."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
    except Exception as e:  # noqa: BLE001 - surface any parse/eval error to the model as text
        return f"Error: could not evaluate {expression!r}: {e}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


if __name__ == "__main__":
    cases = [
        ("3 * (17 + 4) / 2", "31.5"), ("sqrt(144) + 5", "17"), ("factorial(5)", "120"),
        ("2 ** 10", "1024"), ("10 // 3", "3"), ("min(4, 9) + max(1, 2)", "6"),
        ("550 + 685 + 794", "2029"),
        ("__import__('os').system('echo hi')", None), ("x + 1", None), ("2 ** 999999", None),
    ]
    ok = True
    for expr, want in cases:
        got = evaluate(expr)
        rejected = got.startswith("Error:")
        status = ("PASS" if rejected else "FAIL") if want is None else ("PASS" if got == want else "FAIL")
        ok &= rejected if want is None else (got == want)
        print(f"[{status}] evaluate({expr!r}) -> {got}")
    raise SystemExit(0 if ok else 1)
