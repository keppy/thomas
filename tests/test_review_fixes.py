"""Free regression checks for review: split leakage, reward identity and Modal hydration."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from thomas import Task
from thomas.encoder_train import EncoderTrainConfig
from thomas.post_train import _validate_eval_split, _validate_builder_roundtrip, _new_log_path, post_train
from thomas.trl_grpo import _build_dataset, _completion_text, _make_reward_fn, run_grpo_modal
import thomas.trl_grpo as grpo


def score(case, text):
    return (float(text == case["answer"]), "exact")


def task(cases):
    return Task(name="test", cases=cases, score_text=score,
                oracle_reply=lambda c: c["answer"],
                render_messages=lambda c: [{"role": "user", "content": c["prompt"]}])


def test_eval_split_required_before_paid_imports():
    train = task([{"id": "train", "prompt": "x", "answer": "A"}])
    with pytest.raises(ValueError, match="eval_task is required"):
        post_train(train, score_text_fn="tests.test_review_fixes.score")


def test_eval_split_rejects_same_case_or_prompt():
    train = task([{"id": "train", "prompt": "x", "answer": "A"}])
    with pytest.raises(ValueError, match="ids overlap"):
        _validate_eval_split(train, task([{"id": "train", "prompt": "y", "answer": "B"}]))
    with pytest.raises(ValueError, match="prompts overlap"):
        _validate_eval_split(train, task([{"id": "eval", "prompt": "x", "answer": "B"}]))
    _validate_eval_split(train, task([{"id": "eval", "prompt": "y", "answer": "B"}]))


def test_tinker_dotted_reward_must_be_task_reward():
    train = task([{"id": "train", "prompt": "x", "answer": "A"}])
    holdout = task([{"id": "eval", "prompt": "y", "answer": "B"}])
    with pytest.raises(ValueError, match="same function"):
        post_train(train, eval_task=holdout,
                   score_text_fn="thomas.domains.tutor.score")


def test_tinker_roundtrip_detects_independent_case_shape():
    train = task([{"id": "train", "prompt": "x", "answer": "A"}])
    with pytest.raises(ValueError, match="round-trip cannot score"):
        _validate_builder_roundtrip(train, "thomas.post_train._default_from_json", "", "", "")


def test_tutor_roundtrip_matches_documented_dotted_callables():
    from thomas.domains import tutor, tutor_scenarios
    train = Task(name="tutor", cases=tutor_scenarios.CASES[:2],
                 score_text=tutor.score, oracle_reply=tutor.oracle,
                 render_messages=tutor.render, case_id=tutor.case_id)
    _validate_builder_roundtrip(train,
                                "thomas.domains.tutor.cases_from_json",
                                "thomas.domains.tutor.oracle",
                                "thomas.domains.tutor.render",
                                "thomas.domains.tutor.case_id")


def test_tinker_roundtrip_checks_case_fields_beyond_oracle_reward(monkeypatch):
    import json
    from types import ModuleType
    module = ModuleType("roundtrip_probe")
    module.render = lambda case: [{"role": "user", "content": case["prompt"]}]
    def changed(s):
        cases = json.loads(s)
        cases[0]["wrong_reward"] = 0.75
        return cases
    module.changed = changed
    monkeypatch.setitem(sys.modules, "roundtrip_probe", module)
    def scorer(case, text):
        return (1.0 if text == case["answer"] else case["wrong_reward"], "graded")
    t = Task(name="probe", cases=[{"id": "x", "prompt": "p", "answer": "A", "wrong_reward": 0.0}],
             score_text=scorer, oracle_reply=lambda c: c["answer"],
             render_messages=module.render)
    with pytest.raises(ValueError, match="serialized case fields changed"):
        _validate_builder_roundtrip(t, "roundtrip_probe.changed", "", "roundtrip_probe.render", "")


def test_post_train_log_path_is_unique_and_not_path_traversal():
    first, second = _new_log_path("../same name"), _new_log_path("../same name")
    assert first != second
    assert first.startswith("./logs/thomas-") and ".." not in first


def test_modal_gpu_argument_cannot_lie_about_fixed_allocation():
    with pytest.raises(ValueError, match="A100-40GB"):
        run_grpo_modal(task([{"id": "x", "prompt": "p", "answer": "A"}]),
                       "unused://endpoint", gpu="L4")


def test_reward_fn_identity_not_prompt_and_common_text():
    t = task([{"id": "one", "prompt": "same", "answer": "A"},
              {"id": "two", "prompt": "same", "answer": "B"}])
    rewards = _make_reward_fn(t)(
        prompts=[[{"role": "user", "content": "same"}]] * 2,
        completions=[[{"role": "assistant", "content": "<think>A</think>B"}],
                     "A"], case_id=["two", "one"])
    assert rewards == [1.0, 1.0]
    assert _completion_text([{"role": "user", "content": "ignore"},
                             {"role": "assistant", "content": "<think>hidden</think> visible "}]) == "visible"
    with pytest.raises(ValueError, match="case_id"):
        _make_reward_fn(t)([], ["A"])
    with pytest.raises(ValueError, match="unique"):
        _make_reward_fn(task([{"id": "same", "prompt": "x", "answer": "A"},
                              {"id": "same", "prompt": "y", "answer": "B"}]))


def test_dataset_carries_identity_for_duplicate_prompts(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        Dataset=SimpleNamespace(from_list=lambda rows: rows)))
    records = _build_dataset(task([{"id": "one", "prompt": "same", "answer": "A"},
                                   {"id": "two", "prompt": "same", "answer": "B"}]))
    assert [r["case_id"] for r in records] == ["one", "two"]
    assert records[0]["prompt"] == records[1]["prompt"]


def test_modal_api_hydrates_before_remote(monkeypatch):
    trace = []
    @contextmanager
    def app_run():
        trace.append("open")
        yield
        trace.append("close")
    class FakeRemote:
        def remote(self, **kwargs):
            assert trace == ["open"]
            assert kwargs["score_fn_path"] == "thomas.domains.tutor.score"
            trace.append("remote")
            return {"adapter_path": "fake", "metrics": {}}
    monkeypatch.setattr(grpo, "_grpo_app", SimpleNamespace(run=app_run))
    monkeypatch.setattr(grpo, "_grpo_train", FakeRemote())
    t = task([{"id": "one", "prompt": "p", "answer": "A"}])
    t._score_fn_path = "thomas.domains.tutor.score"
    result = run_grpo_modal(t, "unused://endpoint")
    assert result.adapter_path == "fake" and trace == ["open", "remote", "close"]


@pytest.mark.parametrize("changes", [
    {"epochs": 0}, {"batch_size": -1}, {"lr": float("nan")},
    {"calib_size": 0}, {"num_labels": 1}, {"max_length": 0},
])
def test_encoder_config_rejects_invalid_hyperparameters(changes):
    with pytest.raises(ValueError):
        EncoderTrainConfig(**{"model_name": "tiny", "num_labels": 2, **changes})
