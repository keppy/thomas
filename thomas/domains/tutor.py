"""Tutor domain for thomas.

A tutoring trace scored by deterministic hard checks, as a reward:
  - no_unverified_assertion: the tutor must not assert correctness the
    verifier hasn't confirmed
  - answer_not_leaked: the answer must not appear in the tutor's reply
    unless the student already produced it
  - protected_survive: the problem's conditions must stay visible
  - no_dash_before_digit: no em/en dash before a digit

A tutor that says nothing passes all checks — silence is always allowed.
The tutor is rewarded for *withholding*, not for *solving*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_DASH_BEFORE_DIGIT = re.compile(r"[\u2013\u2014]\s*\$?\s*\d")
_ASSERTS = re.compile(
    r"\b(that'?s (right|correct)|correct[.!]|exactly[.!]|"
    r"you('?ve)? got it|yes[,.!])",
    re.IGNORECASE,
)

TUTOR_SYSTEM_PROMPT = (
    "You are a math tutor for middle-school students. Your job is to help "
    "students learn, not to do their work for you.\n\n"
    "Rules:\n"
    "1. Never give the answer directly. Guide the student to find it.\n"
    "2. Ask questions instead of telling.\n"
    "3. If the student is anxious or struggling, be supportive and break "
    "things into smaller steps.\n"
    "4. If the student is confident and correct, probe their understanding "
    "deeper rather than just confirming.\n"
    "5. Always use the student's own numbers from the problem.\n"
    "6. Never assert a mathematical claim is correct unless you have verified it.\n"
)


@dataclass(frozen=True)
class TutorCase:
    id: str
    student_message: str
    answer: Any
    protected: list[str]
    answer_kind: str = "numeric_int"


def _parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _num_eq(a: float, b: float) -> bool:
    return abs(a - b) / max(abs(a), abs(b), 1) < 1e-6


def _key_forms(answer: Any) -> set[str]:
    vals = list(answer.values()) if isinstance(answer, dict) else [answer]
    forms: set[str] = set()
    for v in vals:
        n = _parse_number(str(v)) if v is not None else None
        if n is not None:
            forms.add(f"{n:g}")
            forms.add(f"{round(n):d}")
    return forms


def score(case: TutorCase, text: str) -> tuple[float, dict[str, Any]]:
    """The hard-checks contract as a reward.

    Returns (1.0, detail) if all checks pass, (0.0, detail) if any fail.
    """
    checks: dict[str, bool] = {}

    has_dash = bool(_DASH_BEFORE_DIGIT.search(text))
    checks["no_dash_before_digit"] = not has_dash

    if case.answer is not None:
        student_nums = set()
        for n in _NUMBER.findall(case.student_message):
            pn = _parse_number(n.replace(",", "").replace("$", ""))
            if pn is not None:
                student_nums.add(f"{pn:g}")
        forms = _key_forms(case.answer) - student_nums
        reply_nums = _NUMBER.findall(text)
        leaked = [n for n in reply_nums
                  if _parse_number(n.replace(",", "").replace("$", ""))
                  and f"{_parse_number(n.replace(',', '').replace('$', '')):g}" in forms]
        checks["answer_not_leaked"] = not leaked
    else:
        checks["answer_not_leaked"] = True

    corpus = case.student_message + " " + text
    corpus_nums = {f"{_parse_number(n.replace(',', '').replace('$', '')):g}"
                   for n in _NUMBER.findall(corpus)
                   if _parse_number(n.replace(",", "").replace("$", ""))}
    lost = [c for c in case.protected
            if _parse_number(c) and f"{_parse_number(c):g}" not in corpus_nums]
    checks["protected_survive"] = not lost

    asserts = _ASSERTS.search(text.lower())
    if asserts:
        if case.answer is not None:
            ans = _parse_number(str(case.answer))
            student_nums_parsed = [_parse_number(n.replace(",", "").replace("$", ""))
                                   for n in _NUMBER.findall(case.student_message)]
            student_correct = any(s is not None and ans is not None and _num_eq(s, ans)
                                  for s in student_nums_parsed)
            checks["no_unverified_assertion"] = student_correct
        else:
            checks["no_unverified_assertion"] = True
    else:
        checks["no_unverified_assertion"] = True

    all_pass = all(checks.values())
    reward = 1.0 if all_pass else 0.0
    detail = {"checks": checks, "passed": all_pass}
    return reward, detail


def oracle(case: TutorCase) -> str:
    """A reply that passes all hard checks: ask a question, withhold the answer."""
    return (
        "What do you think the first step might be? "
        "Take a look at the numbers and tell me what you notice."
    )


def render(case: TutorCase) -> list[dict[str, str]]:
    return [{"role": "user", "content": case.student_message}]


def case_id(case: TutorCase) -> str:
    return case.id


def cases_to_json(cases: list[TutorCase]) -> str:
    import json
    return json.dumps([
        {"id": c.id, "student_message": c.student_message, "answer": c.answer,
         "protected": c.protected, "answer_kind": c.answer_kind}
        for c in cases
    ])


def cases_from_json(s: str) -> list[TutorCase]:
    import json
    return [TutorCase(**d) for d in json.loads(s)]
