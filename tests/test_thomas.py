"""Tests for thomas — the harness itself, not any domain.

Tests run without Tinker (no API key needed): the Task abstraction, the
score_text contract, the oracle check, the default render_messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from thomas import Task, baseline_from_outputs
from thomas.task import default_render_messages, default_case_id


# --- A tiny fake domain for testing the harness -------------------------------


@dataclass(frozen=True)
class FakeCase:
    """Minimal case: a prompt, a target answer, an id."""
    id: str
    prompt: str
    target: str


def fake_score(case: FakeCase, text: str) -> tuple[float, str]:
    """Exact-match reward: 1.0 if the text equals the target, else 0.0."""
    if text.strip() == case.target.strip():
        return 1.0, "exact match"
    return 0.0, f"got {text!r}, expected {case.target!r}"


def fake_oracle(case: FakeCase) -> str:
    return case.target


def make_fake_task() -> Task:
    cases = [
        FakeCase(id="c1", prompt="What is 2+2?", target="4"),
        FakeCase(id="c2", prompt="What is 3+3?", target="6"),
        FakeCase(id="c3", prompt="What is 4+4?", target="8"),
    ]
    return Task(
        name="fake-math",
        cases=cases,
        score_text=fake_score,
        oracle_reply=fake_oracle,
        system_prompt="You are a calculator. Answer with just the number.",
    )


# --- Tests -------------------------------------------------------------------


class TestTask:
    def test_task_requires_cases(self):
        with pytest.raises(ValueError, match="no cases"):
            Task(name="empty", cases=[], score_text=fake_score, oracle_reply=fake_oracle)

    def test_task_requires_name(self):
        with pytest.raises(ValueError, match="name"):
            Task(name="", cases=[FakeCase("a", "b", "c")], score_text=fake_score, oracle_reply=fake_oracle)

    def test_n(self):
        task = make_fake_task()
        assert task.n == 3

    def test_oracle_check_passes(self):
        task = make_fake_task()
        assert task.oracle_check() is True

    def test_oracle_check_fails_on_bad_oracle(self):
        cases = [FakeCase(id="c1", prompt="p", target="4")]
        task = Task(
            name="bad",
            cases=cases,
            score_text=fake_score,
            oracle_reply=lambda c: "wrong answer",
        )
        assert task.oracle_check() is False


class TestRenderMessages:
    def test_frozen_dataclass_with_prompt(self):
        case = FakeCase(id="c1", prompt="hello", target="world")
        msgs = default_render_messages(case)
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_dict_with_messages(self):
        case = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
        msgs = default_render_messages(case)
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_bare_string(self):
        case = {"input": "just a string"}
        msgs = default_render_messages(case)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"


class TestCaseId:
    def test_frozen_dataclass(self):
        case = FakeCase(id="abc", prompt="p", target="t")
        assert default_case_id(case) == "abc"

    def test_dict_with_id(self):
        assert default_case_id({"id": "xyz"}) == "xyz"


class TestBaselineFromOutputs:
    """Test the no-Tinker scoring path."""

    def test_perfect_outputs(self):
        task = make_fake_task()
        outputs = {"c1": "4", "c2": "6", "c3": "8"}
        result = baseline_from_outputs(task, outputs)
        assert result.pass_rate == 1.0
        assert len(result.per_case) == 3
        assert all(r["reward"] == 1.0 for r in result.per_case)

    def test_partial_outputs(self):
        task = make_fake_task()
        outputs = {"c1": "4", "c2": "wrong", "c3": "8"}
        result = baseline_from_outputs(task, outputs)
        assert result.pass_rate == pytest.approx(2/3)

    def test_missing_output(self):
        task = make_fake_task()
        outputs = {"c1": "4"}  # missing c2, c3
        result = baseline_from_outputs(task, outputs)
        assert result.pass_rate == pytest.approx(1/3)
