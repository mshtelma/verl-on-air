"""Miles OfficeQA prompt and fixed official split, shared by training and evaluation.

Question, official_split, and SYSTEM_PROMPT are copied from the pinned Miles
source recorded in SOURCE.json. Gold answers are metadata for the reward only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a meticulous financial-data research agent answering questions about U.S. "
    "Treasury Bulletins. Ground every answer in figures you actually retrieve.\n\n"
    "Tools (each result is tagged on its first line as [observation obs_N] -- an id you must "
    "cite later):\n"
    "- search_documents(query, top_k): find the file/table for a figure.\n"
    "- grep_documents(pattern, file_name, year, regex): pin an exact row label/value.\n"
    "- read_document(file_name, start_line, num_lines): read the table cells + headers.\n"
    "- list_documents(year): list available bulletins.\n"
    "- compute(code): run Python (numpy/pandas) for ALL arithmetic.\n"
    "- submit_report(answer, path): FINISH the task. This is the ONLY way to end.\n\n"
    "How this task runs -- plan your steps around it:\n"
    "- You have up to [[MAX_TURNS]] steps, but you need not use them all: call submit_report as "
    "soon as you are confident in your answer.\n"
    "- If you have retrieved figures relevant to the question, commit your BEST supported answer "
    'and cite the observations behind it. Reserve answer="DATA NOT AVAILABLE" for when nothing you '
    "retrieved is relevant -- do not decline merely because you are unsure.\n\n"
    "FINISH by calling submit_report with your answer and the supporting path as its arguments. "
    "Do NOT print the JSON as text -- call the tool. Its arguments:\n"
    '  answer: "<ONLY the final value, in the units the question asks -- no working or other '
    'figures>"\n'
    '  path: a list of {"id","observation":"obs_N you received","claim":"row/period/units/value '
    'or the calculation","depends_on":["earlier id",...]}\n'
    "Rules for the path: cite ONLY observation ids that were delivered to you; a compute step "
    "must list in depends_on the read/search observations whose figures it used; report the path "
    "that actually supports the answer, not every dead end."
)


@dataclass(frozen=True)
class Question:
    uid: str
    question: str
    answer: str
    requirements: str = ""
    source_files: tuple[str, ...] = ()


def official_split(rows: list[dict], *, hard_train: int) -> tuple[list[Question], list[Question]]:
    """(train, held out): every easy question plus ``hard_train`` hard ones, then the other hard ones.

    Hard questions are taken in the order of a hash of their uid, so the split is fixed
    without a seed and does not follow the benchmark's own ordering.
    """
    by_difficulty: dict[str, list[Question]] = {"easy": [], "hard": []}
    for row in sorted(rows, key=lambda row: row["uid"]):
        if row["difficulty"] not in by_difficulty:
            raise ValueError(f"{row['uid']}: unknown difficulty {row['difficulty']!r}")
        by_difficulty[row["difficulty"]].append(
            Question(
                uid=row["uid"],
                question=row["question"].strip(),
                answer=row["answer"].strip(),
                source_files=tuple(name.strip() for name in (row.get("source_files") or "").split() if name.strip()),
            )
        )
    hard = sorted(by_difficulty["hard"], key=lambda question: hashlib.sha256(question.uid.encode()).hexdigest())
    if not 0 < hard_train < len(hard):
        raise ValueError("hard_train must leave hard questions in both training and the held-out set")
    return by_difficulty["easy"] + hard[:hard_train], hard[hard_train:]
