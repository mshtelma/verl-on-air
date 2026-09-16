#!/usr/bin/env python3
"""Deterministic grounding / source-identity checks for OfficeQA trajectories.

The reward's process layer (see ``officeqa_grounded_reward.py``) must catch the
OfficeQA-specific failure the user flagged: a WRONG retrieval path that lands on a
number close to / equal to the gold -- e.g. reading TOTAL national-defense (or war)
expenditures when the question asks for the ARMY component, or a neighbouring
year/period. Answer-match alone credits that lucky path, and GRPO would then
REINFORCE it. So the reward is answer-GATED but, among correct answers, graded by
how well the trajectory actually grounded in the right data.

This module is the CHEAP, DETERMINISTIC half of that grounding signal: FILE-level
source identity + trajectory hygiene, computed with ZERO model calls straight from
the trajectory. The finer SEMANTIC half -- right table / row / category, and all
components of a composite retrieved -- is the LLM judge's job (``judge_prompt.py``),
invoked only when this layer is ambiguous. Splitting it keeps the per-rollout
reward affordable: most lucky-wrong-source rollouts are caught here for free, and
the (self-hosted GLM-5.3) judge is reserved for the genuinely ambiguous band.

ONE code path for eval and training. `extract_engagement` accepts either:
  * a flat decoded trajectory STRING -- what verl passes as ``solution_str`` at
    training time (tool-call XML + tool outputs interleaved as text); or
  * a STRUCTURED list of steps ``[{reasoning, tool_calls:[{name,args}],
    tool_results:[{name,result}]}, ...]`` -- what the eval harness / MLflow trace
    yields (richer: we know which file a read/grep TARGETED, not just mentioned).

GOLD provenance (the answer key):
  * REAL OfficeQA questions -> ``source_files`` column: newline / comma-separated
    corpus file name(s). A COMPOSITE question lists SEVERAL and all are required.
  * SYNTHESISED questions -> additionally the atom value string(s) for a
    value-presence check (did the agent's tool outputs actually contain the cells
    the answer was built from).

Note the file year != the question year: UID0001 asks about calendar-1940 but the
figure lives in ``treasury_bulletin_1941_01.txt`` (bulletins report prior periods),
so we ALWAYS key off gold ``source_files``, never a year parsed from the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Canonical corpus file token, e.g. "treasury_bulletin_1941_01.txt" (the ".txt" is
# optional in call args, always present in tool outputs). Case-insensitive.
_FILE_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})(?:\.txt)?", re.IGNORECASE)
# grep tool output lines are "file:line: text"; search output lines are
# "[i] file (YYYY-MM) score=..."; read output header is "file lines a-b of N:".
_GREP_HIT_RE = re.compile(r"(treasury_bulletin_\d{4}_\d{2}\.txt)\s*:\s*\d+\s*:", re.IGNORECASE)
_SEARCH_HIT_RE = re.compile(r"\[\d+\]\s+(treasury_bulletin_\d{4}_\d{2}\.txt)", re.IGNORECASE)
# tool-call name markers across possible renderings (qwen3_coder XML / hermes JSON).
_TOOLNAMES = ("search_documents", "grep_documents", "read_document", "list_documents", "compute")

# --- trajectory-quality tunables (kept small + explicit; see trajectory_quality) --
_DUP_PENALTY_MAX = 0.30      # max deduction for repeated identical calls
_NO_PIN_PENALTY = 0.30       # searched but never grep/read -> rarely pins a cell
_NO_COMPUTE_PENALTY = 0.20   # composite question but never used compute
_TRUNCATED_PENALTY = 0.20    # ran out of turns still retrieving


def normalize_file(tok: str) -> str:
    """Canonicalise a file reference to ``treasury_bulletin_YYYY_MM.txt`` or ""."""
    m = _FILE_RE.search(str(tok or ""))
    return f"treasury_bulletin_{m.group(1)}_{m.group(2)}.txt" if m else ""


def split_gold_files(source_files: str) -> list[str]:
    """Split the CSV ``source_files`` cell (newline / comma / semicolon / space
    separated; a composite lists several) into a deduped, normalised list."""
    out: list[str] = []
    for part in re.split(r"[\s,;]+", str(source_files or "").strip()):
        f = normalize_file(part)
        if f and f not in out:
            out.append(f)
    return out


@dataclass
class Engagement:
    """What the agent actually did, extracted from its trajectory."""
    tool_counts: dict[str, int] = field(default_factory=dict)
    n_calls: int = 0
    read_files: set[str] = field(default_factory=set)        # read_document that returned content successfully
    grep_target_files: set[str] = field(default_factory=set)  # grep POINTED at (call arg) -- MAY return nothing
    grep_hit_files: set[str] = field(default_factory=set)     # grep actually MATCHED a line (real content seen)
    search_files: set[str] = field(default_factory=set)      # surfaced by search results
    mentioned_files: set[str] = field(default_factory=set)   # any file token anywhere (diagnostic only)
    search_queries: list[str] = field(default_factory=list)
    grep_patterns: list[str] = field(default_factory=list)
    call_signatures: list[str] = field(default_factory=list)  # for dup detection

    @property
    def actively_pulled(self) -> set[str]:
        """Files whose CONTENT the agent actually saw: a read, or a grep that MATCHED.
        A grep POINTED at a file that returned NO match is deliberately excluded -- a
        no-match grep naming the gold file was the confirmed fail-open (it let a correct
        number + empty retrieval score 1.0 on judge outage). Requesting a file is not
        evidence; seeing content from it is."""
        return self.read_files | self.grep_hit_files

    @property
    def grep_files(self) -> set[str]:
        """Back-compat: any grep involvement with a file (targeted or hit)."""
        return self.grep_target_files | self.grep_hit_files


def _bump(counts: dict[str, int], name: str) -> None:
    counts[name] = counts.get(name, 0) + 1


_FAILED_RE = re.compile(
    r"^(?:Error:|No matches|No documents|No passages|.*start_line .*past end|.*chunk index not available)",
    re.IGNORECASE,
)


def _successful_output(result: str) -> str:
    """Return a tool result only if it is an authenticated SUCCESS. An error string,
    empty/no-match listing, or past-end read is not evidence and must not populate
    read/grep-hit files (the review's failed-read fail-open)."""
    body = str(result or "").strip()
    return "" if _FAILED_RE.search(body) else body


def _engagement_from_text(text: str) -> Engagement:
    """Flat-string path (training-time ``solution_str``). It is untrusted text: do not
    authenticate observations merely because they contain a corpus-looking line. Only
    call parameters and explicit tool-output markers provide bounded diagnostics, and
    file tokens anywhere remain a floor only."""
    eng = Engagement()
    t = text or ""

    # tool-call counts (diagnostic only; a model can imitate XML in free text).
    for name in _TOOLNAMES:
        n = len(re.findall(rf"<function={name}\b", t)) or len(re.findall(rf'"name"\s*:\s*"{name}"', t))
        if n:
            eng.tool_counts[name] = n
    eng.n_calls = sum(eng.tool_counts.values())

    # read/grep TARGET files recovered from call blocks are requests, not observations.
    for m in re.finditer(r"<function=(read_document|grep_documents)\b(.*?)</function>", t, re.DOTALL | re.IGNORECASE):
        fn = re.search(r"<parameter=file_name>\s*(.*?)\s*</parameter>", m.group(2), re.DOTALL | re.IGNORECASE)
        f = normalize_file(fn.group(1)) if fn else ""
        if f and m.group(1).lower() == "grep_documents":
            eng.grep_target_files.add(f)

    # Only explicitly marked tool outputs can be treated as observations in a flat trace.
    # A bare assistant/compute line shaped like `treasury...txt:14:` is NOT evidence.
    for out in re.findall(r"\[tool_output:(grep_documents|search_documents)\]\n(.*?)(?=\n\[tool_output:|\Z)",
                          t, re.DOTALL | re.IGNORECASE):
        name, body = out[0].lower(), _successful_output(out[1])
        if not body:
            continue
        if name == "grep_documents":
            eng.grep_hit_files |= {normalize_file(x) for x in _GREP_HIT_RE.findall(body)}
        elif name == "search_documents":
            eng.search_files |= {normalize_file(x) for x in _SEARCH_HIT_RE.findall(body)}
    eng.mentioned_files |= {normalize_file(m.group(0)) for m in _FILE_RE.finditer(t)}
    eng.mentioned_files.discard("")
    return eng


def _engagement_from_steps(steps: list) -> Engagement:
    """Structured path (eval harness / MLflow trace). Full fidelity: each tool call
    carries its parsed args and the exact result string it produced."""
    eng = Engagement()
    for st in steps or []:
        for call in (st.get("tool_calls") or []):
            name = str(call.get("name") or "")
            args = call.get("args") or {}
            if name:
                _bump(eng.tool_counts, name)
            eng.call_signatures.append(f"{name}:{sorted(args.items()) if isinstance(args, dict) else args}")
            if name == "read_document":
                # A read is evidence only after its result below proves successful.
                pass
            elif name == "grep_documents":
                if isinstance(args, dict):
                    f = normalize_file(args.get("file_name", ""))
                    if f:
                        eng.grep_target_files.add(f)     # pointed at -- not yet evidence
                    if args.get("pattern"):
                        eng.grep_patterns.append(str(args["pattern"]))
            elif name == "search_documents" and isinstance(args, dict) and args.get("query"):
                eng.search_queries.append(str(args["query"]))
        results = st.get("tool_results") or []
        calls = st.get("tool_calls") or []
        for idx, res in enumerate(results):
            body = _successful_output(res.get("result"))
            if not body:
                continue
            rname = str(res.get("name") or (calls[idx].get("name") if idx < len(calls) else ""))
            if rname == "read_document":
                args = calls[idx].get("args") if idx < len(calls) else {}
                f = normalize_file(args.get("file_name", "")) if isinstance(args, dict) else ""
                if f:
                    eng.read_files.add(f)
            elif rname == "grep_documents":
                eng.grep_hit_files |= {normalize_file(x) for x in _GREP_HIT_RE.findall(body)}
            elif rname == "search_documents":
                eng.search_files |= {normalize_file(x) for x in _SEARCH_HIT_RE.findall(body)}
            eng.mentioned_files |= {normalize_file(m.group(0)) for m in _FILE_RE.finditer(body)}
    eng.mentioned_files.discard("")
    eng.n_calls = sum(eng.tool_counts.values())
    return eng


def extract_engagement(traj) -> Engagement:
    """Dispatch on trajectory type: flat string (training) or structured steps (eval)."""
    if isinstance(traj, str):
        return _engagement_from_text(traj)
    return _engagement_from_steps(traj)


def source_identity(eng: Engagement, gold_files: list[str]) -> dict:
    """FILE-level source identity: did the agent pull the gold file(s)?

    Two granularities:
      * strong_score -- fraction of gold files the agent ACTIVELY pulled
        (read_document / grep hit). This is what the reward's near-gate uses.
      * any_score    -- fraction merely surfaced (incl. search lists / mentions);
        a softer floor for diagnostics.
    For a COMPOSITE (len(gold) > 1) every required file must be hit for score 1.0.
    """
    gold = list(gold_files or [])
    if not gold:
        return {"gold": [], "strong_score": None, "any_score": None,
                "strong_hit": [], "missed": [], "any_hit": []}
    strong = eng.actively_pulled
    any_seen = strong | eng.grep_target_files | eng.search_files | eng.mentioned_files
    strong_hit = [f for f in gold if f in strong]
    any_hit = [f for f in gold if f in any_seen]
    return {
        "gold": gold,
        "strong_hit": strong_hit,
        "any_hit": any_hit,
        "missed": [f for f in gold if f not in any_seen],
        "strong_score": len(strong_hit) / len(gold),
        "any_score": len(any_hit) / len(gold),
    }


def value_presence(traj, gold_values: list[str]) -> dict:
    """SYNTH only: did the agent's trajectory contain the gold atom value string(s)?
    A necessary (not sufficient) grounding check -- the composed answer must be
    built from cells the agent actually saw."""
    text = traj if isinstance(traj, str) else "\n".join(
        str(r.get("result") or "") for st in (traj or []) for r in (st.get("tool_results") or [])
    )
    vals = [str(v).strip() for v in (gold_values or []) if str(v).strip()]
    if not vals:
        return {"gold_values": [], "seen": [], "score": None}
    seen = [v for v in vals if v in text]
    return {"gold_values": vals, "seen": seen, "score": len(seen) / len(vals)}


def trajectory_quality(eng: Engagement, *, is_composite: bool, truncated: bool = False) -> dict:
    """Cheap trajectory-hygiene score in [0,1]. Deliberately PENALISES pathologies
    (thrash, search-only, missing arithmetic, ran-out-of-turns) rather than
    optimising a raw tool-call count -- some questions genuinely need many calls.
    Kept small so it can never dominate the source-identity / route signal."""
    if eng.n_calls == 0:
        return {"score": 0.0, "reasons": ["no tool calls -- ungrounded"]}
    score, reasons = 1.0, []

    # repeated identical calls (only measurable on the structured path).
    if eng.call_signatures:
        dup = len(eng.call_signatures) - len(set(eng.call_signatures))
        if dup > 0:
            pen = min(_DUP_PENALTY_MAX, 0.1 * dup)
            score -= pen
            reasons.append(f"{dup} duplicate call(s) (-{pen:.2f})")

    # searched but never pinned with grep/read.
    if not (eng.read_files or eng.grep_files) and eng.tool_counts.get("search_documents", 0) > 0:
        score -= _NO_PIN_PENALTY
        reasons.append(f"search-only, never grep/read (-{_NO_PIN_PENALTY:.2f})")

    # composite question but no arithmetic tool used.
    if is_composite and eng.tool_counts.get("compute", 0) == 0:
        score -= _NO_COMPUTE_PENALTY
        reasons.append(f"composite but no compute (-{_NO_COMPUTE_PENALTY:.2f})")

    if truncated:
        score -= _TRUNCATED_PENALTY
        reasons.append(f"turn-budget truncated (-{_TRUNCATED_PENALTY:.2f})")

    return {"score": max(0.0, min(1.0, score)), "reasons": reasons}


def grounding_report(traj, gold: dict) -> dict:
    """Top-level deterministic report combining the checks above.

    gold = {
      "source_files": str,              # required (real + synth)
      "atom_values": list[str] | None,  # synth only (value-presence)
      "is_composite": bool | None,      # else inferred from #gold-files
      "truncated": bool | None,
    }
    Returns a dict the reward consumes AND the human/judge can read.
    """
    eng = extract_engagement(traj)
    gold_files = split_gold_files(gold.get("source_files", ""))
    is_comp = gold.get("is_composite")
    if is_comp is None:
        is_comp = len(gold_files) > 1
    src = source_identity(eng, gold_files)
    tq = trajectory_quality(eng, is_composite=bool(is_comp), truncated=bool(gold.get("truncated")))
    report = {
        "tool_counts": eng.tool_counts,
        "n_calls": eng.n_calls,
        "source_identity": src,
        # files the agent ACTIVELY pulled (read OR grep-HIT), NOT gold-filtered -- so the
        # reward can tell "grounded in SOME real table" (maybe a valid alternative source)
        # from "no real retrieval at all". A grep pointed at a file that returned nothing
        # is NOT counted here (that was the fail-open); it is surfaced separately.
        "files_pulled": sorted(eng.actively_pulled),
        "files_grep_no_hit": sorted(eng.grep_target_files - eng.grep_hit_files),
        "trajectory_quality": tq,
        "is_composite": bool(is_comp),
    }
    if gold.get("atom_values"):
        report["value_presence"] = value_presence(traj, gold["atom_values"])
    return report


if __name__ == "__main__":
    # Offline sanity: a grounded trajectory vs a lucky-wrong-source one.
    grounded = [
        {"reasoning": "find the 1940 defense figure", "tool_calls": [
            {"name": "search_documents", "args": {"query": "national defense expenditures 1940"}}],
         "tool_results": [{"name": "search_documents", "result": "[0] treasury_bulletin_1941_01.txt (1941-01) score=9.1\n  National defense ..."}]},
        {"reasoning": "pin the row", "tool_calls": [
            {"name": "grep_documents", "args": {"pattern": "National defense", "file_name": "treasury_bulletin_1941_01.txt"}}],
         "tool_results": [{"name": "grep_documents", "result": "treasury_bulletin_1941_01.txt:14: National defense ... 2,602"}]},
    ]
    lucky = [
        {"reasoning": "guess from a war-expenditure total", "tool_calls": [
            {"name": "grep_documents", "args": {"pattern": "War expenditures", "file_name": "treasury_bulletin_1946_06.txt"}}],
         "tool_results": [{"name": "grep_documents", "result": "treasury_bulletin_1946_06.txt:31: War expenditures ... 2,600"}]},
    ]
    gold = {"source_files": "treasury_bulletin_1941_01.txt", "atom_values": ["2,602"]}
    import json
    print("GROUNDED:", json.dumps(grounding_report(grounded, gold), indent=2, default=list))
    print("LUCKY   :", json.dumps(grounding_report(lucky, gold), indent=2, default=list))
