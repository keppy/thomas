"""Dream-stream experiment — Phase 0 arms as thomas Tasks.

One case set, four arms (see README.md):

    A  direct answer, no stream
    B  convergent CoT, then answer
    C  dream stream (divergent), then answer
    D  filler stream (compute-only control), then answer

thomas itself is untouched: an arm is just a Task whose
``render_messages`` carries the arm's stream instruction and whose
``score_text`` strips everything before the ANSWER marker before
calling the domain's real scorer. One reward function, all arms.

Offline self-test (no Tinker, no credits):

    python examples/dream_stream/dream_task.py

asserts every arm's oracle scores 1.0 (the env is winnable by
construction) and that the stripper never changes a bare answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from thomas.task import Task

ANSWER_MARKER = "ANSWER:"
DREAM_OPEN, DREAM_CLOSE = "<dream>", "</dream>"

# --- Toy domain: small arithmetic, exact-match verifier ----------------------
# A real domain (foundermath, eerie_rl) plugs in its own cases and
# score_text; only the arm wrappers below are dream-specific.


@dataclass(frozen=True)
class ArithCase:
    id: str
    a: int
    b: int
    op: str  # "+" or "*"

    @property
    def answer(self) -> int:
        return self.a + self.b if self.op == "+" else self.a * self.b


CASES = [
    ArithCase("add-1", 17, 25, "+"),
    ArithCase("add-2", 148, 267, "+"),
    ArithCase("mul-1", 13, 12, "*"),
    ArithCase("mul-2", 47, 8, "*"),
]


def score_arith(case: ArithCase, text: str) -> tuple[float, dict]:
    """The domain's one reward function: exact integer match."""
    extracted = text.strip().rstrip(".")
    try:
        ok = int(extracted) == case.answer
    except ValueError:
        ok = False
    return (1.0 if ok else 0.0), {"extracted": extracted, "expected": case.answer}


# --- Dream-specific wrappers -------------------------------------------------


def strip_stream(text: str) -> str:
    """Keep only what follows the last ANSWER marker.

    The verifier never sees the stream. A bare answer (no marker) passes
    through unchanged, so a model that ignores the stream instruction is
    still scored fairly.
    """
    if ANSWER_MARKER in text:
        return text.rsplit(ANSWER_MARKER, 1)[1].strip()
    return text.strip()


def make_score(case_scorer=score_arith):
    def score_text(case: ArithCase, text: str) -> tuple[float, dict]:
        reward, detail = case_scorer(case, strip_stream(text))
        detail["arm_stripped_chars"] = len(text) - len(strip_stream(text))
        return reward, detail

    return score_text


def render_question(case: ArithCase) -> str:
    sym = "plus" if case.op == "+" else "times"
    return f"What is {case.a} {sym} {case.b}?"


ARM_INSTRUCTIONS = {
    "direct": f"Reply with just '{ANSWER_MARKER} <number>'. Nothing else.",
    "cot": (
        "Think step by step, briefly and carefully. Then give the final "
        f"line '{ANSWER_MARKER} <number>'."
    ),
    "dream": (
        f"Before answering, dream: between {DREAM_OPEN} and {DREAM_CLOSE}, "
        "free-associate — images, memories, half-related facts, whatever "
        "surfaces. It does not need to be relevant or correct. Then wake "
        f"up and give the final line '{ANSWER_MARKER} <number>'."
    ),
    "filler": (
        "Before answering, output a line of dots (like '... ... ...'). "
        f"Then give the final line '{ANSWER_MARKER} <number>'."
    ),
}


def make_task(arm: str) -> Task:
    if arm not in ARM_INSTRUCTIONS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_INSTRUCTIONS)}")
    instruction = ARM_INSTRUCTIONS[arm]

    def render_messages(case: ArithCase) -> list[dict[str, str]]:
        return [{"role": "user", "content": render_question(case)}]

    def oracle_reply(case: ArithCase) -> str:
        ans = f"{ANSWER_MARKER} {case.answer}"
        if arm == "cot":
            return f"{case.a} {case.op} {case.b} = {case.answer}\n{ans}"
        if arm == "dream":
            return f"{DREAM_OPEN}a river of numbers{ans.split()[-1]}{DREAM_CLOSE}\n{ans}"
        if arm == "filler":
            return f"... ... ...\n{ans}"
        return ans

    return Task(
        name=f"dream-stream-{arm}",
        cases=list(CASES),
        score_text=make_score(),
        oracle_reply=oracle_reply,
        system_prompt=f"You are a careful calculator. {instruction}",
        render_messages=render_messages,
    )


def self_test() -> None:
    for arm in ARM_INSTRUCTIONS:
        task = make_task(arm)
        assert task.oracle_check(), f"arm {arm}: oracle must score 1.0"
        # Stripper is transparent to bare answers.
        r, _ = task.score_text(task.cases[0], str(task.cases[0].answer))
        assert r == 1.0, f"arm {arm}: bare answer must score 1.0"
        # Wrong answer inside a well-formed stream still fails.
        bad = f"{DREAM_OPEN}nice dream{DREAM_CLOSE}\n{ANSWER_MARKER} -999"
        r, _ = task.score_text(task.cases[0], bad)
        assert r == 0.0, f"arm {arm}: wrong answer must score 0.0"
        print(f"arm {arm:6s} OK  ({task.n} cases)")


if __name__ == "__main__":
    self_test()
