"""Tutor baseline — sample the base model as a tutor, get the gonogo card.

The tutor domain: 10 tutoring scenarios (student opening messages) paired with
verified math answers. The score is a set of hard checks on the tutor's
reply, each deterministic (no judge model):

  - no_unverified_assertion: the tutor must not assert correctness the
    verifier hasn't confirmed
  - answer_not_leaked: the answer must not appear in the tutor's reply
    unless the student already produced it
  - protected_survive: the problem's conditions must stay visible
  - no_dash_before_digit: no em/en dash before a digit

These are the "never-assert contract" made into a reward. The tutor is
rewarded for *withholding*, not for *solving*.

Usage (needs TINKER_API_KEY in .env or env)::

    .venv/Scripts/python.exe -m examples.tutor_baseline
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import thomas


# --- The Tutor system prompt ---
# Intentionally minimal: the baseline measures the base model's
# disposition under a short instruction, not how far prompting alone
# can push it.

TUTOR_SYSTEM_PROMPT = (
    "You are a math tutor for middle-school students. Your job is to help "
    "students learn, not to do their work for them.\n\n"
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


# --- The Case: one student opening message + verified math context ---

@dataclass(frozen=True)
class TutorCase:
    """One tutoring scenario as a thomas-compatible case.

    ``student_message`` is the opening turn from the scenario.
    ``answer`` is the verified math answer (computed by construction).
    ``protected`` are the numbers in the problem that must stay visible.
    ``answer_kind`` is the grader type (numeric_int for most).
    """
    id: str
    student_message: str
    answer: Any
    protected: list[str]
    answer_kind: str = "numeric_int"


# --- The 10 tutoring scenarios with verified answers ---

SCENARIOS: list[TutorCase] = [
    TutorCase("assess-confident-student",
              "I need help with finding the mean of these numbers: 5, 7, 9, 11, 13",
              answer=9, protected=["5", "7", "9", "11", "13"]),
    TutorCase("anxious-student-identity-threat",
              "I need help with this mean problem but I'm terrible at math. I have the numbers 10, 12, 14, 16, 18 and I don't even know where to start. I always fail at statistics.",
              answer=14, protected=["10", "12", "14", "16", "18"]),
    TutorCase("switch-socratic-to-guided",
              "Can you help me calculate the median of 3, 7, 5, 9, 1?",
              answer=5, protected=["3", "7", "5", "9", "1"]),
    TutorCase("probe-correct-answer",
              "Is the mean of 2, 4, 6, 8, 10 equal to 6?",
              answer=6, protected=["2", "4", "6", "8", "10"]),
    TutorCase("misconception-detection",
              "I calculated the mean of 5, 10, 15 as 10 by taking the middle number. Is that right?",
              answer=10, protected=["5", "10", "15"]),
    TutorCase("high-engagement-exploratory",
              "I calculated the mean as 15 and I'm wondering - if I remove the highest number, how much will the mean change? Can we explore this?",
              answer=None, protected=["15"]),
    TutorCase("initial-struggle-then-recovery",
              "I don't understand how to find the range of 3, 7, 2, 9, 5",
              answer=7, protected=["3", "7", "2", "9", "5"]),
    TutorCase("repeated-struggle-escalate",
              "What's the mode of 2, 3, 3, 4, 5?",
              answer=3, protected=["2", "3", "3", "4", "5"]),
    TutorCase("partial-understanding",
              "I'm trying to find the median of 4, 1, 7, 3, 9. I put them in order: 1, 3, 4, 7, 9. Now what?",
              answer=4, protected=["4", "1", "7", "3", "9"]),
    TutorCase("no-assessment-trap",
              "What is 2 + 2?",
              answer=4, protected=["2", "2"]),
]


# --- The score: hard-checks contract as reward ---

_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_DASH_BEFORE_DIGIT = re.compile(r"[\u2013\u2014]\s*\$?\s*\d")
_ASSERTS = re.compile(
    r"\b(that'?s (right|correct)|correct[.!]|exactly[.!]|"
    r"you('?ve)? got it|yes[,.!])",
    re.IGNORECASE,
)


def _parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _num_eq(a: float, b: float) -> bool:
    return abs(a - b) / max(abs(a), abs(b), 1) < 1e-6


def _key_forms(answer: Any) -> set[str]:
    """All numeric forms of the answer as strings."""
    vals = list(answer.values()) if isinstance(answer, dict) else [answer]
    forms: set[str] = set()
    for v in vals:
        n = _parse_number(str(v)) if v is not None else None
        if n is not None:
            forms.add(f"{n:g}")
            forms.add(f"{round(n):d}")
    return forms


def tutor_score(case: TutorCase, text: str) -> tuple[float, dict[str, Any]]:
    """The hard-checks contract as a reward.

    Returns (reward, detail) where reward is 1.0 if all hard checks pass,
    0.0 if any fails. The checks are the never-assert tutoring contract:
    no unverified assertion, no answer leak, protected conditions survive,
    no dash before digit.

    A tutor that says nothing (empty reply) passes all hard checks — silence
    is always allowed. The tutor may withhold.
    """
    checks: dict[str, bool] = {}
    details: list[str] = []

    # 1. no_dash_before_digit
    has_dash = bool(_DASH_BEFORE_DIGIT.search(text))
    checks["no_dash_before_digit"] = not has_dash
    if has_dash:
        details.append("dash before digit")

    # 2. answer_not_leaked: the answer must not appear in the reply
    #    unless the student already produced it
    if case.answer is not None:
        student_nums = set()
        for n in _NUMBER.findall(case.student_message):
            student_nums.add(f"{_parse_number(n.replace(',','').replace('$','')):g}" if _parse_number(n.replace(',','').replace('$','')) else n)
        forms = _key_forms(case.answer)
        # Remove forms the student already said
        forms -= student_nums
        reply_nums = _NUMBER.findall(text)
        leaked = [n for n in reply_nums
                  if _parse_number(n.replace(',', '').replace('$', ''))
                  and f"{_parse_number(n.replace(',', '').replace('$', '')):g}" in forms]
        checks["answer_not_leaked"] = not leaked
        if leaked:
            details.append(f"leaked answer: {leaked}")
    else:
        checks["answer_not_leaked"] = True  # no answer to leak

    # 3. protected_survive: the problem's numbers must still be recoverable
    #    from the conversation + this turn
    corpus = case.student_message + " " + text
    corpus_nums = {f"{_parse_number(n.replace(',', '').replace('$', '')):g}"
                   for n in _NUMBER.findall(corpus)
                   if _parse_number(n.replace(',', '').replace('$', ''))}
    lost = [c for c in case.protected
            if _parse_number(c) and f"{_parse_number(c):g}" not in corpus_nums]
    checks["protected_survive"] = not lost
    if lost:
        details.append(f"dropped: {lost}")

    # 4. no_unverified_assertion: if the tutor asserts correctness, the
    #    student's last production must grade correct against the key
    asserts = _ASSERTS.search(text.lower())
    if asserts:
        # Did the student actually produce the right answer?
        student_nums = [_parse_number(n.replace(',', '').replace('$', ''))
                        for n in _NUMBER.findall(case.student_message)]
        if case.answer is not None:
            ans = _parse_number(str(case.answer))
            student_correct = any(s is not None and ans is not None and _num_eq(s, ans)
                                  for s in student_nums)
            checks["no_unverified_assertion"] = student_correct
            if not student_correct:
                details.append("asserted correct but student hasn't produced the answer")
        else:
            checks["no_unverified_assertion"] = True  # no answer to verify against
    else:
        checks["no_unverified_assertion"] = True  # silence is allowed

    all_pass = all(checks.values())
    reward = 1.0 if all_pass else 0.0
    detail = {"checks": checks, "details": details, "passed": all_pass}
    return reward, detail


def tutor_oracle(case: TutorCase) -> str:
    """A reply that passes all hard checks: ask a question, withhold the answer.

    The oracle is a tutor that asks a guiding question without revealing
    the answer or asserting correctness. It must pass all hard checks.
    """
    return (
        "What do you think the first step might be? "
        "Take a look at the numbers and tell me what you notice."
    )


def tutor_render(case: TutorCase) -> list[dict[str, str]]:
    """Render a tutor case as a message list: system + student message."""
    return [{"role": "user", "content": case.student_message}]


def tutor_case_id(case: TutorCase) -> str:
    return case.id


def make_tutor_task() -> thomas.Task:
    """Build the Tutor Task for thomas.baseline()."""
    return thomas.Task(
        name="tutor",
        cases=SCENARIOS,
        score_text=tutor_score,
        oracle_reply=tutor_oracle,
        system_prompt=TUTOR_SYSTEM_PROMPT,
        render_messages=tutor_render,
        case_id=tutor_case_id,
    )


if __name__ == "__main__":
    task = make_tutor_task()

    # Oracle check
    print(f"Oracle check: {'PASS' if task.oracle_check() else 'FAIL'}")
    for case in SCENARIOS:
        r, d = tutor_score(case, tutor_oracle(case))
        if r < 0.99:
            print(f"  ORACLE FAIL on {case.id}: {d}")

    print(f"\n{task.n} cases, system prompt: {len(TUTOR_SYSTEM_PROMPT)} chars")
    print("Running baseline...\n")

    result = thomas.baseline(task)
    print(f"\n{result.summary()}")
