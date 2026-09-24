"""The Task: a domain's plug into the harness.

A Task is the bundle a domain supplies to thomas:

  - ``cases``: a list of Case objects (any shape — thomas does not impose a
    Case type; the domain owns it)
  - ``score_text``: ``(case, text) → (reward, detail)`` — the one reward
    function that drives both the baseline card and the RL loop
  - ``oracle_reply``: ``(case) → str`` — a reply that scores 1.0, proving the
    env is winnable by construction
  - ``render_messages``: ``(case) → list[dict[str, str]]`` — how to turn a
    Case into a message list for the model (system + user turns)
  - ``system_prompt``: the base system line the model sees

The domain owns the Case shape, the reward, and the prompt. thomas owns
everything that wraps them: sampling, the gonogo card, the Tinker Env, the
training loop, the before/after comparison.

This mirrors ``eerie_rl.case.score_text`` and ``eerie_rl.env.EerieEnv``:
``score_text`` is the single scoring path, used by both the env's ``step``
and any offline scorer, so the baseline and the training loop see identical
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# The reward function signature. Pure: no I/O, no model calls.
#   case: the domain's Case object (any shape)
#   text:  what the model produced (str)
# Returns:
#   reward: float — the scalar the RL loop maximizes
#   detail: dict or str — human-readable, for the card and logs
ScoreFn = Callable[[Any, str], tuple[float, Any]]

# A reply that scores 1.0 by construction, proving the env is winnable.
OracleFn = Callable[[Any], str]

# How to turn a Case into a message list for the model.
RenderFn = Callable[[Any], list[dict[str, str]]]


def default_render_messages(case: Any) -> list[dict[str, str]]:
    """Render a Case's input as a message list.

    Tries common shapes: a ``prompt`` attribute (eerie_rl), a dict with
    ``messages`` (gonogo-style), a dict with ``input`` that has
    ``messages``, or falls back to stringifying the input.
    """
    # eerie_rl Case: has .prompt
    if hasattr(case, "prompt"):
        return [{"role": "user", "content": case.prompt}]

    # Dict case: look for "input" key, then "messages"
    if isinstance(case, dict):
        inp = case.get("input", case)
        if isinstance(inp, dict) and "messages" in inp:
            return [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in inp["messages"]
            ]
        if isinstance(case, dict) and "messages" in case:
            return [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in case["messages"]
            ]
        return [{"role": "user", "content": str(inp)}]

    # Object case: try .input
    inp = getattr(case, "input", case)
    if isinstance(inp, dict) and "messages" in inp:
        return [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in inp["messages"]
        ]

    # bare input
    return [{"role": "user", "content": str(inp)}]


def default_case_id(case: Any) -> str:
    """Get a case's id, trying common shapes."""
    if hasattr(case, "id"):
        return str(case.id)
    if isinstance(case, dict) and "id" in case:
        return str(case["id"])
    return repr(case)[:50]


@dataclass
class Task:
    """A domain's plug into the thomas harness.

    Attributes:
        name: domain name, used in gonogo task labels and logging tags
        cases: the Case set — any shape the domain uses
        score_text: ``(case, text) → (reward, detail)`` — the one reward function
        oracle_reply: ``(case) → str`` — a reply that scores 1.0
        system_prompt: the base system line the model sees
        render_messages: how to turn a Case into a message list
        case_id: how to get a case's id (for logging and gonogo)
    """

    name: str
    cases: list[Any]
    score_text: ScoreFn
    oracle_reply: OracleFn
    system_prompt: str = ""
    render_messages: RenderFn = field(default=default_render_messages)
    case_id: Callable[[Any], str] = field(default=default_case_id)

    def __post_init__(self):
        if not self.cases:
            raise ValueError(f"Task '{self.name}' has no cases")
        if not self.name:
            raise ValueError("Task.name is required")

    @property
    def n(self) -> int:
        return len(self.cases)

    def oracle_check(self) -> bool:
        """Verify every case's oracle reply scores 1.0.

        The env is winnable by construction: ``score_text(case,
        oracle_reply(case))`` must be 1.0 for every case, or the env is
        asking for something its own grader cannot accept.
        """
        for case in self.cases:
            reward, _ = self.score_text(case, self.oracle_reply(case))
            if reward < 0.99:
                return False
        return True
