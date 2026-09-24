"""Skill classification domain for thomas.

This lets the System 1 skill classifier run through thomas's harness:
  - thomas.baseline: sample a model on the cases, score, produce a gonogo card
  - thomas.compare: before/after comparison with a paired McNemar test

The domain owns the Case shape (SkillCase) and the reward function
(exact match on skill_id). thomas owns the plumbing.

Usage:
    from thomas.domains.skill_classify import SkillClassifyTask
    from thomas.baseline import baseline_from_outputs

    task = SkillClassifyTask(cases)
    # Run model, collect outputs, then score:
    result = baseline_from_outputs(task, outputs_dict)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thomas.task import Task


@dataclass(frozen=True)
class _SkillCaseProxy:
    """Lightweight case shape for thomas — a skill-tagged prompt.

    Any object or dict with ``id``, ``prompt`` and ``skill_id`` works; this
    proxy is what the JSONL loader builds.
    """

    id: str
    prompt: str
    skill_id: str
    tier: str = "entry"
    group: str | None = None


def score_text(case: Any, text: str) -> tuple[float, dict[str, Any]]:
    """The one reward function: exact match on skill_id.

    ``text`` is the model's predicted skill label. Returns (1.0, detail) on
    match, (0.0, detail) on mismatch. This is the same contract as
    ``thomas.domains.tutor.score`` and ``thomas.domains.eerie.score``.
    """
    expected = case.skill_id if hasattr(case, "skill_id") else case.get("skill_id")
    ok = text.strip() == expected
    return (
        1.0 if ok else 0.0,
        {"predicted": text, "expected": expected, "correct": ok},
    )


def oracle_reply(case: Any) -> str:
    """A reply that scores 1.0 by construction: the ground-truth skill_id."""
    return case.skill_id if hasattr(case, "skill_id") else case["skill_id"]


def render_messages(case: Any) -> list[dict[str, str]]:
    """Render a case as a message list for the model.

    For a classifier, the 'prompt' is the math problem text and the model
    is asked to output the skill label.
    """
    prompt = case.prompt if hasattr(case, "prompt") else case["prompt"]
    return [
        {
            "role": "user",
            "content": (
                f"Classify the following math problem into one skill. "
                f"Reply with only the skill label.\n\n{prompt}"
            ),
        }
    ]


SYSTEM_PROMPT = (
    "You are a skill classifier for middle-school math problems. "
    "Given a math problem, output the skill label it exercises. "
    "Output only the label, nothing else."
)


def build_task(cases: list[Any], name: str = "skill_classify") -> Task:
    """Build a thomas Task from a list of SkillCase-like objects.

    Each case must have: id, prompt, skill_id. Optional: tier, group.
    """
    # Normalize cases to the proxy shape
    normalized = []
    for c in cases:
        if hasattr(c, "prompt"):
            normalized.append(c)  # already a dataclass
        elif isinstance(c, dict):
            normalized.append(_SkillCaseProxy(**c))
        else:
            raise TypeError(f"Expected dict or dataclass, got {type(c)}")

    return Task(
        name=name,
        cases=normalized,
        score_text=score_text,
        oracle_reply=oracle_reply,
        system_prompt=SYSTEM_PROMPT,
        render_messages=render_messages,
        case_id=lambda c: c.id,
    )
