"""Answer extraction from workflow output."""

from __future__ import annotations

import re
from typing import Protocol


class AnswerExtractor(Protocol):
    """Protocol for extracting a final answer from raw workflow output."""

    def extract(self, raw_output: str) -> str | None: ...


class XMLTagExtractor:
    """Extracts from <FINAL_ANSWER>...</FINAL_ANSWER> with fallbacks.

    Strategy order:
    1. Exact XML tag match (last occurrence — most likely the refined answer)
    2. Case-insensitive variant
    3. Bold markdown: **FINAL ANSWER**: ...
    4. "Final Answer:" prefix pattern

    Returns None only if ALL strategies fail.
    No "last number" heuristic — would corrupt evaluation for text/date answers.
    """

    def __init__(self, tag: str = "FINAL_ANSWER") -> None:
        self._tag = tag

    def extract(self, raw_output: str) -> str | None:
        if not raw_output:
            return None

        # Strategy 1: Exact XML tag (take LAST match)
        pattern = rf"<{self._tag}>(.*?)</{self._tag}>"
        matches = re.findall(pattern, raw_output, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Strategy 2: Case-insensitive XML tag
        matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        # Strategy 3: Bold markdown — **FINAL ANSWER**: value or **FINAL_ANSWER**: value
        tag_human = self._tag.replace("_", " ")
        for variant in [self._tag, tag_human]:
            md_pattern = rf"\*\*{re.escape(variant)}\*\*[:\s]+(.+?)(?:\n\n|\Z)"
            match = re.search(md_pattern, raw_output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Strategy 4: "Final Answer:" prefix
        prefix_pattern = rf"(?:^|\n)\s*{re.escape(tag_human)}[:\s]+(.+?)(?:\n\n|\n(?=[A-Z#*])|\Z)"
        match = re.search(prefix_pattern, raw_output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # Strategy 5: Variable assignment extraction (namespace fallback output)
        # When the react loop exhausts its tool budget, the compute namespace is
        # dumped as "Extracted data:\nvar = value\n...". Try to extract the most
        # likely answer variable.
        text = raw_output.strip()
        if text.startswith("Extracted data:"):
            # Prefer variables whose name signals "result" or "answer"
            for pat in [
                r'(?:result|answer|final_?(?:value|answer)|total)\s*=\s*(.+?)(?:\n|$)',
            ]:
                matches = re.findall(pat, text, re.MULTILINE | re.IGNORECASE)
                if matches:
                    return matches[-1].strip().strip("'\"")

            # Fallback: last numeric variable assignment
            for line in reversed(text.split("\n")):
                m = re.match(r"\w+\s*=\s*([\d.,\[\]\-\(\) ]+)$", line.strip())
                if m:
                    return m.group(1).strip()

        return None
