"""Runtime-owned OfficeQA observations and terminal tool submissions."""
from __future__ import annotations

import json
import re

import path_report

SUBMIT_TOOL = "submit_report"


def new_record(episode_id: str, question: str) -> dict:
    return {"episode_id": episode_id, "question": question, "question_requirements": "",
            "observations": [], "terminal_text": "", "terminal_seen": False}


def outcome(text: str) -> str:
    if not text.strip():
        return "empty"
    stripped = text.lstrip()
    if stripped.startswith(("Error:", "Error executing", "Unknown function", "Invalid JSON")):
        return "error"
    if stripped.startswith("No ") and re.search(r"found|match|document", stripped[:80]):
        return "empty"
    return "ok"


def observation(record: dict, *, name: str, arguments: dict, text: str, request: int, max_chars: int) -> str:
    """Clip first; queue exact policy-visible bytes, without claiming delivery yet."""
    order = len(record["observations"]) + 1
    oid = f"obs_{order}"
    prefix = f"[observation {oid}]\n"
    budget = max(0, max_chars - len(prefix))
    if len(text) > budget:
        suffix = "...(truncated)"
        text = text[:max(0, budget - len(suffix))] + suffix[:budget]
    record["observations"].append({
        "observation_id": oid, "order": order, "tool": name, "args": arguments,
        "outcome": outcome(text), "generated_by_request": request,
        "delivered_to_request": None, "delivered_text": None, "_pending_text": text,
    })
    return prefix + text


def delivered(record: dict, *, request: int) -> None:
    """Called only immediately before generation on the committed tool context."""
    for row in record["observations"]:
        if "_pending_text" in row and row["generated_by_request"] < request:
            row["delivered_text"] = row.pop("_pending_text")
            row["delivered_to_request"] = request


def submit(record: dict, arguments: str | dict) -> None:
    record["terminal_seen"] = True
    try:
        report = path_report.loads_strict(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(report, dict):
            return
        if isinstance(report.get("path"), str):
            report = {**report, "path": path_report.loads_strict(report["path"])}
        record["terminal_text"] = json.dumps(report, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        record["terminal_text"] = ""


def finalize(record: dict) -> dict:
    for row in record["observations"]:
        row.pop("_pending_text", None)
    return record
