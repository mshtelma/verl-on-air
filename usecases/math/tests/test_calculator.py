"""usecases/math/tool.py: the calculator stops code execution AND resource exhaustion (R23).

Before: factorial and expression size were unbounded, `factorial(2000)` escaped the error handler
(str() of a >4,300-digit int raises ValueError), round(1, -10**9) would have built 10**(10**9),
and an error message echoed the model's own input -- so a planted \\boxed{} could reach the rule
score (reward.py reads the whole episode) through a tool reply."""
from __future__ import annotations

import time

import pytest

from support import REPO, load_module

calc = load_module(REPO / "usecases/math/tool.py")


@pytest.mark.parametrize("expr,want", [
    ("3 * (17 + 4) / 2", "31.5"),
    ("sqrt(144) + 5", "17"),
    ("factorial(5)", "120"),
    ("2 ** 10", "1024"),
    ("10 // 3", "3"),
    ("-7 % 3", "2"),
    ("min(4, 9) + max(1, 2)", "6"),
    ("round(2.675, 2)", "2.67"),
    ("gcd(84, 36)", "12"),
    ("log(8, 2)", "3"),
    ("2 ** -1", "0.5"),
    ("factorial(100) % 1000000007", str(__import__("math").factorial(100) % 1000000007)),
    ("+".join(["1"] * 100), "100"),                       # a long sum is fine
])
def test_ordinary_arithmetic_still_works(expr, want):
    assert calc.evaluate(expr) == want


@pytest.mark.parametrize("expr", [
    "factorial(2000)",                      # the review's reproduction
    "factorial(10**100)",
    "2 ** 999999",
    "9 ** 9 ** 9",
    "10 ** 3000",
    "(2 ** 3000) * (2 ** 3000)",            # each factor is fine; the product is not
    "round(1, -10**9)",
    "round(1.5, 10**6)",
    "exp(1000)",
    "1e308 * 10",                           # inf
    "1e309",
    "(-8) ** (1/3)",                        # complex
    "sqrt(-1)",
    "log(0)",
    "1 / 0",
    "10 ** 400 / 3.0",                      # int too large for a float
    "min()",
    "factorial(5.5)",
    "-" * 400 + "1",                        # nested past MAX_DEPTH
    "1" * 501,                              # past MAX_EXPR_CHARS
    "__import__('os').system('echo hi')",
    "(lambda: 1)()",
    "x + 1",
    "'a' * 10",
    "[1, 2]",
    "1 if 1 else 2",
    "True + 1",
    "",
])
def test_every_hostile_or_broken_request_is_answered_with_an_error_string(expr):
    out = calc.evaluate(expr)
    assert isinstance(out, str) and out.startswith("Error: could not evaluate the expression"), out


def test_nothing_escapes_the_error_boundary():
    for bad in (None, 42, b"1+1", ["1"]):
        assert calc.evaluate(bad).startswith("Error:")                     # type: ignore[arg-type]


@pytest.mark.parametrize("expr", [
    r"\boxed{42}", r"#### 42", r"sqrt(\boxed{42})", r"'\boxed{7}' + 1", "#### 22\n1+", r"x_\boxed{1}",
])
def test_a_tool_reply_never_carries_the_models_own_text(expr):
    out = calc.evaluate(expr)
    assert out.startswith("Error:") and "boxed" not in out and "####" not in out and "42" not in out, out


def test_the_costliest_admitted_requests_are_still_cheap():
    """Bounded by construction: the most expensive work the limits let through takes milliseconds,
    whether it is admitted or refused only once its (bounded) result is seen."""
    admitted = ["factorial(450) % 1000000007 + factorial(449) % 97",
                "(3 ** 2100) // (7 ** 1100) + gcd(2 ** 3000, 6 ** 1000)",
                "+".join(["factorial(400)"] * 30)]
    refused_late = ["(3 ** 2100) * (7 ** 1200)", "factorial(450) * factorial(449)"]
    t0 = time.perf_counter()
    for expr in admitted:
        assert not calc.evaluate(expr).startswith("Error:"), expr
    for expr in refused_late:
        assert "exceeds" in calc.evaluate(expr), expr
    assert time.perf_counter() - t0 < 0.5


def test_results_are_bounded():
    big = calc.evaluate("2 ** 3399")                 # the largest power of two admitted
    assert big.isdigit() and len(big) <= calc.MAX_OUTPUT_CHARS
    assert calc.evaluate("2 ** 3401").startswith("Error:")


def test_the_eval_runs_the_calculator_off_the_event_loop():
    src = (REPO / "usecases/math/eval.py").read_text()
    assert "asyncio.to_thread(calc_evaluate" in src and "res = calc_evaluate(" not in src
