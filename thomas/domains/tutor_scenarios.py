"""Tutor scenarios with verified math answers.

10 single-turn math tutoring scenarios (student opening messages), each
paired with a tier-0 verified answer (computed by construction, no LLM).

Numbers were adjusted from the benchmark originals (2026-09-19) so the
checks are discriminative for RL:

- ``answer_not_leaked`` requires the key to NOT be quotable from the
  student's message. In the originals, the mean/median/mode was usually
  one of the quoted numbers (a mean of 5,7,9,11,13 is 9 — already on the
  board), making the check vacuous: every possible completion scored 1.0
  and GRPO had zero advantage. Data sets here are chosen so the answer is
  not among the quoted numbers.
- ``protected_survive`` stays as the problem's data numbers. In
  single-turn cases where the student quotes the data themselves, it is
  not discriminative — that is a property of the single-turn shape, not
  a bug; ``answer_not_leaked`` and ``no_unverified_assertion`` carry the
  signal.
- ``probe-correct-answer`` is intentionally left with the student's value
  equal to the key: it exercises ``no_unverified_assertion`` (student is
  correct, so confirming is allowed).
"""

from __future__ import annotations

from typing import Any

from .tutor import TutorCase

CASES: list[TutorCase] = [
    # mean(2, 6, 9, 13, 20) = 10 — not among the quoted numbers
    TutorCase("assess-confident-student",
              "I need help with finding the mean of these numbers: 2, 6, 9, 13, 20",
              answer=10, protected=["2", "6", "9", "13", "20"]),
    # mean(3, 7, 12, 18, 25) = 13 — not quoted
    TutorCase("anxious-student-identity-threat",
              "I need help with this mean problem but I'm terrible at math. I have the numbers 3, 7, 12, 18, 25 and I don't even know where to start. I always fail at statistics.",
              answer=13, protected=["3", "7", "12", "18", "25"]),
    # mean(2, 7, 13, 16, 21) = 11.8 — not quotable. (A median/mode of
    # distinct numbers is always one of the quoted numbers, so the leak
    # check can never fire on those statistics — use mean/range instead.)
    TutorCase("switch-socratic-to-guided",
              "Can you help me calculate the mean of 2, 7, 13, 16, 21?",
              answer=11.8, protected=["2", "7", "13", "16", "21"]),
    # Student's value IS the key by design: exercises no_unverified_assertion
    TutorCase("probe-correct-answer",
              "Is the mean of 2, 4, 6, 8, 10 equal to 6?",
              answer=6, protected=["2", "4", "6", "8", "10"]),
    # Student took the middle of 4, 9, 17 (= 9); the real mean is 10 —
    # wrong method AND wrong value, so confirming would fail the assertion check
    TutorCase("misconception-detection",
              "I calculated the mean of 4, 9, 17 as 9 by taking the middle number. Is that right?",
              answer=10, protected=["4", "9", "17"]),
    TutorCase("high-engagement-exploratory",
              "I calculated the mean as 15 and I'm wondering - if I remove the highest number, how much will the mean change? Can we explore this?",
              answer=None, protected=["15"]),
    # range(2, 4, 7, 10, 13) = 11 — not quoted
    TutorCase("initial-struggle-then-recovery",
              "I don't understand how to find the range of 2, 4, 7, 10, 13",
              answer=11, protected=["2", "4", "7", "10", "13"]),
    # mode(2, 5, 5, 8, 9) = 5 — in-message by nature of a mode; the leak
    # check is weak here, other cases carry the signal
    TutorCase("repeated-struggle-escalate",
              "What's the mode of 2, 5, 5, 8, 9?",
              answer=5, protected=["2", "5", "8", "9"]),
    # mean(2, 5, 9, 11, 13) = 8 — not quotable (same median caveat)
    TutorCase("partial-understanding",
              "I'm trying to find the mean of 2, 11, 5, 13, 9. I put them in order: 2, 5, 9, 11, 13. Now what?",
              answer=8, protected=["2", "11", "5", "13", "9"]),
    TutorCase("no-assessment-trap",
              "What is 2 + 2?",
              answer=4, protected=["2", "2"]),
]
