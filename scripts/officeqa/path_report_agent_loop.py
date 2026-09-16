"""verl custom agent loop + tools for the OfficeQA PATH-REPORT environment (GRPO training).

This reproduces, INSIDE verl's rollout, the exact environment the path-report verifier was
validated against (submit_report terminal + 3-phase funnel + observation ledger), so
train == the validated verifier == deploy. Confirmed against verl v0.9.0 (the image pin):
  * ToolAgentLoop is a pluggable ``@register`` agent loop; we subclass + register.
  * BaseTool.execute receives ``agent_data`` -> tools write the observation ledger into
    ``agent_data.extra_fields`` -> verl carries it to non_tensor_batch -> the reward's
    ``compute_score(extra_info)`` (see scripts/reward/officeqa_path_report_reward.py).
  * response_mask is native: tool-response / injected tokens get 0 -> NO gradient (only the
    model's own tokens are trained), so the funnel scaffolding is context, never learned.

Design (minimal, robust override -- we do NOT reimplement verl's tool-processing):
  * Tools are thin BaseTool wrappers over the SAME officeqa impls used at eval; each prefixes
    its output with ``[observation obs_N]`` (identical to collection) and appends the obs to the
    ledger in ``agent_data.extra_fields["_pr_obs"]``.
  * ``submit_report`` records its {answer, path} args; the LOOP detects the call and terminates.
  * ``run`` finalises EVERY episode (submit OR turn-cap OR any other exit) to a single uniform
    ``extra_fields["path_report_record"]`` and drops all ``_pr_*`` scratch -- verl's
    ``DataProto.concat`` asserts an IDENTICAL non_tensor_batch schema across the GRPO group, so a
    scratch key (e.g. ``_pr_obs``) left in only the turn-capped samples aborts the whole batch.
  * The 3-phase FUNNEL is per-turn tool-gating via ``agent_data._active_tools`` (verl-native):
    explore -> retrieval-locked (last nudge_window turns: only compute + submit) -> submit-only
    (last submit_only_window turns: only submit_report). A locked tool call returns verl's
    "Unknown function ..." redirect -- analogous to collection's locked message.

The pure helpers (funnel phase, record build) are stdlib-only and CPU-testable; the verl
classes are guarded so this module imports without verl for those tests.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the collection contract verbatim: same submit schema + submit-arg decoder + outcome
# classifier that produced the records the reward was validated on (no drift).
from path_report_collect import (  # noqa: E402
    SUBMIT_TOOL, SUBMIT_TOOL_SCHEMA, RETRIEVAL_TOOLS, DEFAULT_NUDGE_WINDOW,
    DEFAULT_SUBMIT_ONLY_WINDOW, OBS_MARKER, classify_outcome, _report_text_from_submit,
)

AGENT_LOOP_NAME = "path_report_agent"
COMPUTE_TOOL = "compute"


# ===========================================================================
# PURE, CPU-testable core
# ===========================================================================
def funnel_phase(assistant_turns: int, max_turns: int, nudge_window: int,
                 submit_only_window: int) -> str:
    """Which funnel phase the turn ABOUT to run is in. ``assistant_turns`` = turns already
    completed (verl increments it after a generation), so the turn about to run is #assistant_turns
    and ``remaining = max_turns - assistant_turns`` (this turn included) -- matches the collector's
    ``remaining = max_turns - turn``."""
    remaining = max_turns - assistant_turns
    if submit_only_window > 0 and remaining <= submit_only_window:
        return "submit_only"
    if nudge_window > 0 and remaining <= nudge_window:
        return "retrieval_lock"
    return "explore"


def active_tool_names(phase: str, all_names, *, submit_tool: str = SUBMIT_TOOL,
                      retrieval_tools=RETRIEVAL_TOOLS) -> list[str]:
    """Tools OFFERED this turn for a funnel phase. submit_report is available in EVERY phase
    (finish anytime); retrieval is dropped in the lock window; only submit in submit-only."""
    all_names = list(all_names)
    if phase == "submit_only":
        return [submit_tool] if submit_tool in all_names else []
    if phase == "retrieval_lock":
        return [n for n in all_names if n not in retrieval_tools]
    return all_names


def record_observation(obs_list: list, *, tool: str, args: dict, output: str, turn: int) -> str:
    """Append one delivered observation (plain dict; JSON-serialisable for non_tensor_batch)
    and return its runtime-issued id. Ids are obs_1.. in execution order -- the id the model
    sees in the ``[observation obs_N]`` marker, so citations resolve in the reward."""
    n = len(obs_list) + 1
    oid = f"obs_{n}"
    obs_list.append({
        "observation_id": oid, "order": n, "tool": tool,
        "args": dict(args or {}), "outcome": classify_outcome(output),
        "delivered_text": output, "generated_by_request": turn,
        "delivered_to_request": turn,   # verl appends every tool response to the context -> delivered
    })
    return oid


def build_path_report_record(*, episode_id: str, question: str, requirements: str,
                             terminal_text: str, termination: str, obs_list: list) -> dict:
    """Assemble the capture-shaped record the reward scores -- byte-identical to the offline
    collector's ``EpisodeCapture.finalize`` output (verified to round-trip through the reward).

    ``terminal_text``/``termination`` are passed EXPLICITLY so both loop exits produce the SAME
    record shape: a submit episode ('terminal_report' + the report JSON) and a turn-capped
    no-submit episode ('no_submit_at_cap' + '' -> parses malformed -> ZERO -> 0.0). The reward's
    scorer reads terminal_text/observations only (never ``termination``, which is diagnostic)."""
    return {
        "episode_id": episode_id,
        "question": question,
        "question_requirements": requirements,
        "terminal_text": terminal_text,
        "termination": termination,
        "answer_correct": None,
        "observations": obs_list,
        "raw_executions": [],
        "raw_turns": [],
        "n_observations": len(obs_list),
        "n_delivered": sum(1 for o in obs_list if o.get("delivered_to_request") is not None),
        "max_turns": None,
        "collector_version": "path_report_agent/verl",
    }


# Scratch keys the loop writes into ``agent_data.extra_fields`` DURING an episode; every one MUST
# be removed at finalize so it never reaches non_tensor_batch. verl's ``DataProto.concat`` requires
# a uniform key schema across the GRPO group, so a scratch key present in only some samples (e.g.
# ``_pr_obs`` surviving in turn-capped episodes) aborts the whole batch assembly.
_PR_SCRATCH_KEYS = ("_pr_obs", "_pr_submit", "_pr_question", "_pr_episode_id", "_pr_requirements")


def finalize_extra_fields(ef: dict) -> dict:
    """Collapse a finished episode's extra_fields to the SINGLE uniform output key.

    Idempotent and termination-agnostic (submit, turn-cap, or any other loop exit): ensures
    ``ef['path_report_record']`` exists (built from the scratch when not already present) and drops
    every ``_pr_*`` scratch key. Called from the loop's ``run`` override for EVERY sample so the
    non_tensor_batch schema is identical across the group. Returns ``ef`` (mutated in place)."""
    if "path_report_record" not in ef:
        submit_args = ef.get("_pr_submit", None)
        if submit_args is not None:
            terminal_text, termination = _report_text_from_submit(submit_args), "terminal_report"
        else:
            terminal_text, termination = "", "no_submit_at_cap"
        ef["path_report_record"] = build_path_report_record(
            episode_id=ef.get("_pr_episode_id", ""),
            question=ef.get("_pr_question", ""),
            requirements=ef.get("_pr_requirements", ""),
            terminal_text=terminal_text,
            termination=termination,
            obs_list=ef.get("_pr_obs", []) or [],
        )
    for k in _PR_SCRATCH_KEYS:
        ef.pop(k, None)
    return ef


def _first_user_question(messages) -> str:
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""


# ===========================================================================
# verl-dependent classes (guarded so the pure core imports on CPU)
# ===========================================================================
try:
    from verl.experimental.agent_loop.agent_loop import register
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
    _HAS_VERL = True
except Exception:  # noqa: BLE001 - CPU / no-verl: pure helpers above remain usable + testable
    _HAS_VERL = False


if _HAS_VERL:
    # underlying officeqa tool impls (same functions the eval @function_tools wrap)
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tools"))
    import officeqa_tools as _oqt  # noqa: E402

    def _impl(name: str, args: dict) -> str:
        a = args or {}
        if name == "search_documents":
            return _oqt._search_documents(a.get("query", ""), _oqt._coerce_int(a.get("top_k", _oqt._TOP_K), _oqt._TOP_K))
        if name == "grep_documents":
            return _oqt._grep_documents(a.get("pattern", ""), a.get("file_name", ""), a.get("year", ""),
                                        str(a.get("regex", "")).strip().lower() in ("1", "true", "yes"))
        if name == "read_document":
            return _oqt._read_document(a.get("file_name", ""), _oqt._coerce_int(a.get("start_line", 0), 0),
                                       _oqt._coerce_int(a.get("num_lines", 100), 100))
        if name == "list_documents":
            return _oqt._list_documents(a.get("year", ""))
        if name == "compute":
            return _oqt._compute(a.get("code", ""))
        return f"Error: unknown tool {name!r}."

    def _schema_obj(d: dict) -> "OpenAIFunctionToolSchema":
        return OpenAIFunctionToolSchema(**d)

    class _LedgerTool(BaseTool):
        """Wraps one officeqa impl: run it, record the delivered observation into the ledger,
        return the [observation obs_N]-prefixed text (exactly as collection delivered it)."""
        IMPL_NAME = ""

        async def execute(self, instance_id, parameters, agent_data=None, **kwargs):
            out = _impl(self.IMPL_NAME, parameters if isinstance(parameters, dict) else {})
            ef = agent_data.extra_fields
            obs = ef.setdefault("_pr_obs", [])
            oid = record_observation(obs, tool=self.IMPL_NAME,
                                     args=parameters if isinstance(parameters, dict) else {},
                                     output=out, turn=getattr(agent_data, "assistant_turns", 0))
            return ToolResponse(text=OBS_MARKER.format(oid=oid) + "\n" + out), 0.0, {}

    def _mk_tool_cls(impl_name: str):
        return type(f"Tool_{impl_name}", (_LedgerTool,), {"IMPL_NAME": impl_name})

    SearchTool = _mk_tool_cls("search_documents")
    GrepTool = _mk_tool_cls("grep_documents")
    ReadTool = _mk_tool_cls("read_document")
    ListTool = _mk_tool_cls("list_documents")
    ComputeTool = _mk_tool_cls("compute")

    class SubmitReportTool(BaseTool):
        """Terminal tool: records {answer, path} for the LOOP to finalise; the loop detects the
        call and ends the episode (verl's ToolResponse has no stop field, so termination is the
        loop's job)."""
        async def execute(self, instance_id, parameters, agent_data=None, **kwargs):
            agent_data.extra_fields["_pr_submit"] = parameters
            return ToolResponse(text="Report received. Episode complete."), 0.0, {}

    @register(AGENT_LOOP_NAME)
    class PathReportAgentLoop(ToolAgentLoop):
        """ToolAgentLoop + submit-termination + 3-phase funnel + observation ledger.

        Funnel windows come from env (set by the launcher): OQ_NUDGE_WINDOW (retrieval lock) and
        OQ_SUBMIT_ONLY_WINDOW (submit-only), matching the collector defaults. max_turns is verl's
        max_assistant_turns."""

        def _windows(self):
            nudge = int(os.environ.get("OQ_NUDGE_WINDOW", str(DEFAULT_NUDGE_WINDOW)) or "0")
            submit_only = int(os.environ.get("OQ_SUBMIT_ONLY_WINDOW", str(DEFAULT_SUBMIT_ONLY_WINDOW)) or "0")
            return nudge, submit_only

        def _apply_funnel(self, agent_data) -> None:
            max_turns = self.max_assistant_turns or 0
            if not max_turns:
                return
            nudge, submit_only = self._windows()
            phase = funnel_phase(agent_data.assistant_turns, max_turns, nudge, submit_only)
            keep = set(active_tool_names(phase, self.tools.keys()))
            agent_data._active_tools = {n: t for n, t in self.tools.items() if n in keep}
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                for n, t in self.tools.items() if n in keep
            ]

        async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
            # Gate tools for THIS turn (funnel) BEFORE generating. This sets _active_tools only --
            # it must NOT touch extra_fields, or it would trip the base copy below.
            self._apply_funnel(agent_data)
            state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
            # Stash episode identity ONCE, but only AFTER super(): on turn 0 the base runs
            # `if not agent_data.extra_fields: agent_data.extra_fields.update(output.extra_fields)`,
            # which is how the fully-async staleness stamps (min_global_steps / max_global_steps)
            # reach the sample. Pre-seeding _pr_* before super() makes extra_fields non-empty, skips
            # that copy, and leaves those stamps None -> detach_utils abs(None-None) TypeError.
            if "_pr_question" not in agent_data.extra_fields:
                agent_data.extra_fields["_pr_question"] = _first_user_question(agent_data.messages)
                tk = agent_data.tools_kwargs or {}
                agent_data.extra_fields["_pr_episode_id"] = str(
                    tk.get("uid") or tk.get("episode_id") or agent_data.request_id)
                agent_data.extra_fields["_pr_requirements"] = str(tk.get("question_requirements") or "")
            return state

        async def _handle_processing_tools_state(self, agent_data):
            state = await super()._handle_processing_tools_state(agent_data)
            # submit_report was called this turn -> end the episode. The record is built ONCE, for
            # every termination path, in run() -> finalize_extra_fields (not here), so submit and
            # turn-cap episodes emit an identical non_tensor_batch schema.
            if "_pr_submit" in agent_data.extra_fields:
                return AgentState.TERMINATED
            return state

        async def run(self, sampling_params, **kwargs):
            # ToolAgentLoop.run drives the state machine to TERMINATED, then packs
            # agent_data.extra_fields into AgentLoopOutput.extra_fields (-> non_tensor_batch).
            # Finalize on the way out so EVERY sample -- submit, turn-cap, or any other exit --
            # carries the SAME single key (path_report_record) and no _pr_* scratch; a mixed schema
            # aborts DataProto.concat for the whole GRPO group.
            output = await super().run(sampling_params, **kwargs)
            try:
                finalize_extra_fields(output.extra_fields)
            except Exception:  # noqa: BLE001 - finalization must never fail an already-done rollout
                ef = output.extra_fields
                ef.setdefault("path_report_record", build_path_report_record(
                    episode_id="", question="", requirements="",
                    terminal_text="", termination="finalize_error", obs_list=[]))
                for k in _PR_SCRATCH_KEYS:
                    ef.pop(k, None)
            return output

    # tool registry the launcher/TOOL config points at: name -> (class, schema)
    TOOL_CLASSES = {
        "search_documents": SearchTool, "grep_documents": GrepTool, "read_document": ReadTool,
        "list_documents": ListTool, "compute": ComputeTool, SUBMIT_TOOL: SubmitReportTool,
    }
