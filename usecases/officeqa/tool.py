"""Miles' six OfficeQA tool schemas and retrieval/compute implementations for verl."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import run_control  # noqa: E402

import corpus
import sandbox
from verl.tools.function_tool import function_tool

TOOLS = json.loads(Path(__file__).with_name("tools.json").read_text())
SCHEMAS = {tool["function"]["name"]: tool for tool in TOOLS}


def data_dir() -> Path:
    return Path(os.environ.get("OQ_LOCAL_DIR") or os.environ["OQ_DATA_DIR"])


def execute(name: str, arguments: dict) -> str:
    schema = SCHEMAS.get(name)
    if schema is None or name == "submit_report":
        raise ValueError(f"{name!r} is not an executable retrieval/compute tool")
    parameters = schema["function"]["parameters"]
    missing = set(parameters.get("required", [])) - set(arguments)
    unknown = set(arguments) - set(parameters["properties"])
    if missing or unknown:
        raise ValueError(f"invalid {name} arguments: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if name == "compute":
        result = sandbox.run_code(arguments["code"])
        if result.startswith(("Error: compute sandbox unavailable", "Error: compute sandbox failed to launch")):
            run_control.request_abort(result, "usecases/officeqa/tool.py")
        return result
    function = getattr(corpus, name)
    try:
        loaded = corpus.load_corpus(data_dir())
    except Exception as error:
        run_control.request_abort(f"OfficeQA corpus unavailable: {error}", "usecases/officeqa/tool.py")
        raise
    return function(loaded, **arguments)


@function_tool("search_documents", schema=SCHEMAS["search_documents"])
def search_documents(query: str, top_k: int = 6) -> str:
    return execute("search_documents", {"query": query, "top_k": top_k})


@function_tool("grep_documents", schema=SCHEMAS["grep_documents"])
def grep_documents(pattern: str, file_name: str = "", year: str = "", regex: bool = False) -> str:
    return execute("grep_documents", {"pattern": pattern, "file_name": file_name, "year": year, "regex": regex})


@function_tool("read_document", schema=SCHEMAS["read_document"])
def read_document(file_name: str, start_line: int = 0, num_lines: int = 100) -> str:
    return execute("read_document", {"file_name": file_name, "start_line": start_line, "num_lines": num_lines})


@function_tool("list_documents", schema=SCHEMAS["list_documents"])
def list_documents(year: str = "") -> str:
    return execute("list_documents", {"year": year})


@function_tool("compute", schema=SCHEMAS["compute"])
def compute(code: str) -> str:
    return execute("compute", {"code": code})


@function_tool("submit_report", schema=SCHEMAS["submit_report"])
def submit_report(answer: str, path: list[dict]) -> str:
    """The OfficeQA controller intercepts this call and terminates the episode."""
    return "Report submitted."
