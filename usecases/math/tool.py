#!/usr/bin/env python3
"""Calculator tool for the math use case's tool-agent rollout.

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

SAFETY -- two separate properties:
  * no code execution: evaluation is a whitelisted AST walk (NO eval/exec, no
    attribute access, no names beyond a small math allowlist);
  * bounded resources: the input, the AST (nodes and depth), every integer
    (MAX_INT_BITS, intermediate or final), factorial and power arguments (checked
    BEFORE they run), round()'s digit count and the output are all capped, and
    non-finite or complex results are refused. So every allowed expression costs
    microseconds and a few KB -- bounded by construction, which a timeout on a
    thread could not guarantee (runaway big-integer arithmetic cannot be
    interrupted). verl runs the tool with asyncio.to_thread; eval.py does the same.

Every outcome, errors included, is returned as text, and that text never contains
the model's own input: the rule score (reward.py) reads the whole episode, so a
tool reply must not be able to carry a \\boxed{} or #### the model planted in it.

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

# Resource bounds (see SAFETY above).
MAX_EXPR_CHARS = 500        # the model's input
MAX_NODES = 1_000           # AST nodes (a 100-term sum is ~300)
MAX_DEPTH = 300             # AST nesting: far below the recursion limit, above any 500-char sum
MAX_INT_BITS = 3_400        # any integer, intermediate or final: ~1,000 decimal digits
MAX_ROUND_DIGITS = 100      # |ndigits| of round(): round(1, -10**9) would build 10**(10**9)
MAX_OUTPUT_CHARS = 1_100


class CalcError(ValueError):
    """A request the calculator refuses. The message is ours -- never the model's text."""


def _number(x):
    """Every intermediate result: a real, finite number of bounded size."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise CalcError("the result is not a real number")
    if isinstance(x, float) and not math.isfinite(x):
        raise CalcError("the result is not finite")
    if isinstance(x, int) and x.bit_length() > MAX_INT_BITS:
        raise CalcError(f"an integer exceeds {MAX_INT_BITS} bits (~1,000 digits)")
    return x


def _pow(base, exp):
    # Only an integer power can grow without bound; floats overflow (OverflowError) at once.
    if isinstance(base, int) and isinstance(exp, int) and exp > 0 and abs(base) > 1:
        if exp * math.log2(abs(base)) > MAX_INT_BITS:
            raise CalcError(f"the power would exceed {MAX_INT_BITS} bits (~1,000 digits)")
    return base ** exp


def _factorial(n):
    if isinstance(n, int) and not isinstance(n, bool) and n > 1 and (
            n > 10_000 or math.lgamma(n + 1) / math.log(2) > MAX_INT_BITS):
        raise CalcError(f"factorial({n}) would exceed {MAX_INT_BITS} bits (~1,000 digits)")
    return math.factorial(n)


def _round(x, *nd):
    if nd and isinstance(nd[0], int) and abs(nd[0]) > MAX_ROUND_DIGITS:
        raise CalcError(f"round() takes at most {MAX_ROUND_DIGITS} digits")
    return round(x, *nd)


_GUARDED = {"factorial": _factorial, "round": _round}


def _check_shape(tree: ast.AST) -> None:
    stack, n = [(tree, 0)], 0
    while stack:
        node, depth = stack.pop()
        n += 1
        if n > MAX_NODES:
            raise CalcError(f"the expression has more than {MAX_NODES} parts")
        if depth > MAX_DEPTH:
            raise CalcError(f"the expression is nested more than {MAX_DEPTH} levels deep")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError("only numbers are allowed as constants")
        return _number(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalcError(f"disallowed operator {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        return _number(_pow(left, right) if op is operator.pow else op(left, right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalcError(f"disallowed unary operator {type(node.op).__name__}")
        return _number(op(_eval(node.operand)))
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name) or node.func.id not in _CALLABLES:
            raise CalcError("only positional calls to the listed functions are permitted")
        fn = _GUARDED.get(node.func.id, _CALLABLES[node.func.id])
        return _number(fn(*[_eval(a) for a in node.args]))
    if isinstance(node, ast.Name):
        if node.id not in _CONSTS:
            raise CalcError("unknown name (variables are not allowed; constants: pi, e, tau)")
        return _CONSTS[node.id]
    raise CalcError(f"disallowed syntax: {type(node).__name__}")


def evaluate(expression: str) -> str:
    """Pure, verl-free core so the logic is unit-testable standalone. Never raises."""
    try:
        if not isinstance(expression, str):
            raise CalcError("the expression must be a string")
        if len(expression) > MAX_EXPR_CHARS:
            raise CalcError(f"the expression is longer than {MAX_EXPR_CHARS} characters")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            raise CalcError("not a single arithmetic expression") from None
        _check_shape(tree)
        result = _eval(tree)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        text = str(result)
        if len(text) > MAX_OUTPUT_CHARS:
            raise CalcError(f"the result is longer than {MAX_OUTPUT_CHARS} characters")
        return text
    except CalcError as e:
        return f"Error: could not evaluate the expression: {e}"
    except ZeroDivisionError:
        return "Error: could not evaluate the expression: division by zero"
    except (ArithmeticError, ValueError, TypeError) as e:  # math domain/range errors, bad arity
        return f"Error: could not evaluate the expression: {type(e).__name__}"
    except Exception as e:  # noqa: BLE001 - the tool must always answer with text
        return f"Error: could not evaluate the expression ({type(e).__name__})"


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
# Local sanity check:  python3 usecases/math/tool.py   (the full suite: usecases/math/tests)
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
        ("factorial(2000)", None),                      # used to raise past str()'s digit limit
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
