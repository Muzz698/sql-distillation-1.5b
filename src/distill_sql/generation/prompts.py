"""Prompts sent to the teachers, and parsing of their JSON answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from distill_sql.data.datasets import Example

Message = dict[str, str]

SYSTEM = (
    "You are an expert data analyst. You write correct, efficient SQLite queries and "
    "natural questions that real users would ask."
)

_RULES = """Rules:
- Use only tables and columns that exist in the schema. Wrap identifiers that are SQL keywords \
or contain spaces in double quotes.
- Each query is ONE read-only SQLite statement (SELECT, or WITH ... SELECT).
- Each query must return at least one row on this database. Use filter values that appear in the \
example rows or the column values listed above.
- Questions are natural English. Do not write table or column names in snake_case, do not \
mention SQL, and make each question answerable from the database alone.
- Every question must be different from the example questions and from each other.
- Select only the columns the question asks for."""


def schema_block(ddl: str, value_hints: dict[str, list[str]]) -> str:
    if not value_hints:
        return ddl
    lines = [f"{col}: " + ", ".join(repr(v) for v in vals) for col, vals in value_hints.items()]
    return (
        ddl
        + "\n\n/* Column values (columns with few distinct values):\n"
        + "\n".join(lines)
        + "\n*/"
    )


def generation_messages(
    schema_text: str,
    fewshot: list[Example],
    n_pairs: int,
    focus: str,
    batch_no: int,
) -> list[Message]:
    examples = "\n\n".join(f"Q: {ex.question}\nSQL: {ex.sql}" for ex in fewshot)
    parts = [
        f"Database schema (SQLite):\n\n{schema_text}",
        f"Example questions for this database, with correct SQL:\n\n{examples}" if fewshot else "",
        f"Write {n_pairs} NEW question/SQL pairs for this database (batch {batch_no}).\n"
        f"Focus for this batch: {focus}.",
        _RULES,
        'Return ONLY a JSON array, no other text:\n[{"question": "...", "sql": "..."}]',
    ]
    user = "\n\n".join(p for p in parts if p)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def crosscheck_messages(schema_text: str, questions: list[str]) -> list[Message]:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    user = (
        f"Database schema (SQLite):\n\n{schema_text}\n\n"
        f"Write one SQLite query for each question:\n\n{numbered}\n\n"
        "Wrap identifiers that are SQL keywords or contain spaces in double quotes. "
        f"Return ONLY a JSON array of exactly {len(questions)} SQL strings, in order:\n"
        '["SELECT ...", "SELECT ..."]'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class ParsedPair:
    question: str
    sql: str


class ParseError(ValueError):
    pass


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_array(text: str) -> Any:
    text = _THINK.sub("", text).strip()
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        start, end = cand.find("["), cand.rfind("]")
        if start == -1 or end <= start:
            continue
        try:
            return json.loads(cand[start : end + 1])
        except json.JSONDecodeError:
            continue
    raise ParseError("no JSON array found in the response")


def parse_pairs(text: str) -> list[ParsedPair]:
    data = _extract_json_array(text)
    if not isinstance(data, list):
        raise ParseError("response JSON is not a list")
    pairs = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q, s = item.get("question"), item.get("sql")
        if isinstance(q, str) and isinstance(s, str) and q.strip() and s.strip():
            pairs.append(ParsedPair(" ".join(q.split()), s.strip().rstrip(";").strip()))
    if not pairs:
        raise ParseError("JSON array contains no valid {question, sql} objects")
    return pairs


def parse_sql_list(text: str, expected: int) -> list[str | None]:
    data = _extract_json_array(text)
    if not isinstance(data, list):
        raise ParseError("response JSON is not a list")
    out: list[str | None] = [
        s.strip().rstrip(";").strip() if isinstance(s, str) and s.strip() else None for s in data
    ]
    if len(out) != expected:
        raise ParseError(f"expected {expected} SQL strings, got {len(out)}")
    return out
