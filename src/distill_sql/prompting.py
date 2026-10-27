"""The text-to-SQL task prompt, shared by evaluation, training and serving.

Every model (base, fine-tuned, teacher) sees exactly this format at evaluation time, and the
fine-tuned model is trained on it, so scores are comparable and train/serve skew is avoided.
"""

from __future__ import annotations

import re

Message = dict[str, str]

SYSTEM = (
    "You are a SQLite expert. Given a database schema and a question, write one SQLite query "
    "that answers the question. Return only the SQL query, with no explanation."
)


def sql_task_messages(schema_ddl: str, question: str, evidence: str | None = None) -> list[Message]:
    parts = [f"Database schema:\n\n{schema_ddl}", f"Question: {question}"]
    if evidence:
        parts.append(f"Hint: {evidence}")  # BIRD's "external knowledge" field
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


_THINK = re.compile(r"<(think|thought)>.*?(</\1>|$)", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
_PREFIX = re.compile(r"^\s*(?:sql(?:ite)?\s*(?:query)?\s*:)\s*", re.IGNORECASE)


def _first_statement(sql: str) -> str:
    """Text up to the first `;` that is not inside quotes."""
    quote: str | None = None
    for i, ch in enumerate(sql):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            return sql[:i]
    return sql


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply: drop reasoning blocks, prefer a ```sql fence,
    strip a leading "SQL:" label, keep only the first statement."""
    text = _THINK.sub("", text).strip()
    if (m := _FENCE.search(text)) is not None:
        text = m.group(1)
    text = _PREFIX.sub("", text.strip())
    return " ".join(_first_statement(text).split())
