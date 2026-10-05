"""Tests for the evalroute lane-encoder recipe driver (examples/evalroute_lane_encoder.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "evalroute_lane_encoder.py"

import importlib.util

_spec = importlib.util.spec_from_file_location("evalroute_lane_encoder", EXAMPLE)
evalroute_lane_encoder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evalroute_lane_encoder)


LABELS = ["lane_a", "lane_b", "lane_c"]


def _make_cases(n: int, seed: int) -> list[dict]:
    """Short distinct strings so the tiny random model's rows are unambiguous."""
    cases = []
    for i in range(n):
        label = LABELS[i % len(LABELS)]
        cases.append({
            "id": f"seed:{i}" if i % 2 else f"taskset:{i}",
            "text": f"task {label} marker {i}",
            "label": label,
        })
    return cases


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """Random-init BertConfig classifier + tokenizer from a 10-token vocab."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    root = tmp_path_factory.mktemp("tiny-model")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
        f"tok{i}" for i in range(5)
    ]
    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)
    config = BertConfig(
        vocab_size=10, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=1, intermediate_size=32,
        num_labels=len(LABELS),
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(str(root))
    tokenizer.save_pretrained(str(root))
    return str(root)


def _run_main(tmp_path, tiny_model, argv_extra: list[str]):
    """Run the script's main() with argv patched; returns (exit_code, out_dir)."""
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    out_dir = tmp_path / "out"
    with open(train_path, "w", encoding="utf-8") as f:
        for c in _make_cases(12, 1):
            f.write(json.dumps(c) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for c in _make_cases(6, 2):
            f.write(json.dumps(c) + "\n")

    argv = [
        "evalroute_lane_encoder.py",
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--out", str(out_dir),
        "--model", tiny_model,
        "--backend", "local",
        "--epochs", "1",
        "--calib-size", "3",
        *argv_extra,
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        code = evalroute_lane_encoder.main()
    finally:
        sys.argv = old_argv
    return code, out_dir


def test_local_train_writes_contract_artifact(tmp_path, tiny_model):
    code, out_dir = _run_main(tmp_path, tiny_model, [])
    assert code == 0

    for name in ("config.json", "label2id.json", "temperature.json", "metrics.json"):
        assert (out_dir / name).exists(), f"missing {name}"

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    for key in (
        "eval_accuracy", "eval_n", "eval_per_lane", "defer_below",
        "coverage_at_defer", "train_sources", "contract_version", "recipe",
        "model", "epochs", "lr", "seed", "device", "elapsed_s",
    ):
        assert key in metrics, f"metrics.json missing {key}"
    assert metrics["eval_n"] == 6
    assert metrics["recipe"] == "evalroute_lane_encoder"
    assert 0.0 <= metrics["defer_below"] <= 1.0
    assert 0.0 <= metrics["coverage_at_defer"] <= 1.0
    assert metrics["eval_per_lane"].keys() == {"lane_a", "lane_b", "lane_c"}
    assert metrics["train_sources"]["seed"] == 6
    assert metrics["train_sources"]["taskset"] == 6

    preds = [
        json.loads(line)
        for line in (out_dir / "eval_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(preds) == 6
    for row in preds:
        assert set(row) == {"id", "label", "predicted", "confidence"}
        assert row["predicted"] in LABELS


def test_dry_run_writes_nothing(tmp_path, tiny_model):
    code, out_dir = _run_main(tmp_path, tiny_model, ["--dry-run"])
    assert code == 0
    assert not out_dir.exists()


def test_calib_size_refusal(tmp_path, tiny_model):
    code, out_dir = _run_main(tmp_path, tiny_model, ["--calib-size", "6"])
    assert code != 0
    assert not out_dir.exists()


def test_modal_requires_yes(tmp_path, tiny_model, monkeypatch):
    sentinel = type("ModalSentinel", (), {
        "__getattr__": lambda self, name: (_ for _ in ()).throw(
            AssertionError("modal imported without --yes")
        ),
    })()
    monkeypatch.setitem(sys.modules, "modal", sentinel)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for c in _make_cases(12, 1):
            f.write(json.dumps(c) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for c in _make_cases(6, 2):
            f.write(json.dumps(c) + "\n")

    argv = [
        "evalroute_lane_encoder.py",
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--out", str(tmp_path / "out"),
        "--model", tiny_model,
        "--backend", "modal",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        code = evalroute_lane_encoder.main()
    finally:
        sys.argv = old_argv
    assert code != 0
    assert sys.modules.get("modal") is sentinel
