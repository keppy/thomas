"""Tests for the skill_classify thomas domain."""

from __future__ import annotations

from thomas.domains.skill_classify import (
    _SkillCaseProxy,
    build_task,
    score_text,
    oracle_reply,
    render_messages,
    SYSTEM_PROMPT,
)
from thomas.task import Task


def _sample_cases():
    return [
        _SkillCaseProxy(id="FM-1", prompt="You sell mugs for $17 each...", skill_id="break_even", tier="entry"),
        _SkillCaseProxy(id="FM-2", prompt="Draw a box plot...", skill_id="compare_box_plots", tier="core"),
    ]


class TestScoreText:
    def test_correct(self):
        case = _SkillCaseProxy(id="x", prompt="p", skill_id="break_even")
        reward, detail = score_text(case, "break_even")
        assert reward == 1.0
        assert detail["correct"] is True

    def test_wrong(self):
        case = _SkillCaseProxy(id="x", prompt="p", skill_id="break_even")
        reward, detail = score_text(case, "compare_box_plots")
        assert reward == 0.0
        assert detail["correct"] is False

    def test_stripped(self):
        case = _SkillCaseProxy(id="x", prompt="p", skill_id="break_even")
        reward, _ = score_text(case, "  break_even  ")
        assert reward == 1.0


class TestOracle:
    def test_oracle_scores_perfect(self):
        case = _SkillCaseProxy(id="x", prompt="p", skill_id="break_even")
        reply = oracle_reply(case)
        assert reply == "break_even"
        reward, _ = score_text(case, reply)
        assert reward == 1.0


class TestRender:
    def test_render(self):
        case = _SkillCaseProxy(id="x", prompt="What is 2+2?", skill_id="break_even")
        msgs = render_messages(case)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "What is 2+2?" in msgs[0]["content"]


class TestBuildTask:
    def test_build_from_dataclasses(self):
        cases = _sample_cases()
        task = build_task(cases)
        assert task.name == "skill_classify"
        assert len(task.cases) == 2
        assert task.case_id(task.cases[0]) == "FM-1"

    def test_build_from_dicts(self):
        cases = [
            {"id": "FM-1", "prompt": "p1", "skill_id": "break_even"},
            {"id": "FM-2", "prompt": "p2", "skill_id": "compare_box_plots"},
        ]
        task = build_task(cases)
        assert len(task.cases) == 2

    def test_oracle_winnable(self):
        """Every case should be winnable with the oracle reply."""
        cases = _sample_cases()
        task = build_task(cases)
        for case in task.cases:
            reply = task.oracle_reply(case)
            reward, _ = task.score_text(case, reply)
            assert reward == 1.0
