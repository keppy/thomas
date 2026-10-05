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


# --- train 2026-10-H brief 01: selection, multi-seed, curve ------------------


def test_selection_caps_aug_and_weights_real():
    from lane_encoder_common import LABELS, evalroute_lane_encoder

    cases = []
    for i in range(10):
        for lane in LABELS:
            cases.append({"id": f"aug:{i}", "text": f"aug {lane} {i}", "label": lane})
    # real: 2 in lane_a, 0 in lane_b, 4 in lane_c (contributor prefix counts as real)
    for i in range(2):
        cases.append({"id": f"taskset:{i}", "text": f"real a {i}", "label": "lane_a"})
    for i in range(4):
        cases.append({"id": f"keppy:{i}", "text": f"real c {i}", "label": "lane_c"})
    for lane in LABELS:
        cases.append({"id": "seed:0", "text": f"seed {lane}", "label": lane})

    rows, report = evalroute_lane_encoder.select_training_rows(cases)
    pl = report["per_lane"]
    assert pl["lane_a"]["aug_kept"] == 5 and pl["lane_a"]["aug_dropped"] == 5
    assert pl["lane_a"]["real"] == 2
    assert pl["lane_b"]["aug_kept"] == 10 and pl["lane_b"]["aug_dropped"] == 0  # zero real
    assert pl["lane_b"]["real"] == 0
    assert pl["lane_c"]["aug_kept"] == 5 and pl["lane_c"]["real"] == 4  # max(5, 4)
    assert all(pl[l]["seed"] == 1 for l in LABELS)
    assert report["real_weight"] == 3 and report["aug_cap"] is None

    counts: dict = {}
    for text, label in rows:
        counts[(text, label)] = counts.get((text, label), 0) + 1
    assert counts[("real a 0", "lane_a")] == 3  # real rows repeated 3x
    assert counts[("seed lane_b", "lane_b")] == 1  # seed rows once
    assert counts[("aug lane_c 0", "lane_c")] == 1
    assert ("aug lane_a 9", "lane_a") not in counts  # capped away

    rows2, report2 = evalroute_lane_encoder.select_training_rows(
        cases, aug_cap=2, real_weight=1)
    pl2 = report2["per_lane"]
    assert pl2["lane_a"]["aug_kept"] == 2
    assert pl2["lane_b"]["aug_kept"] == 10  # zero-real scaffolding ignores the cap
    assert pl2["lane_c"]["aug_kept"] == 2
    counts2: dict = {}
    for text, _ in rows2:
        counts2[text] = counts2.get(text, 0) + 1
    assert counts2["real a 0"] == 1


def test_single_seed_unchanged_shape(tmp_path, tiny_model):
    from lane_encoder_common import _run_main

    code, out_dir = _run_main(tmp_path, tiny_model, [])
    assert code == 0
    for name in ("config.json", "label2id.json", "temperature.json", "metrics.json"):
        assert (out_dir / name).exists()
    assert not (out_dir / "seed-7").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["seeds"] == [7]
    assert metrics["chosen_seed"] == 7
    assert metrics["eval_accuracy_sd"] == 0.0
    assert "selection" in metrics
    assert (out_dir / "seeds.json").exists()


def test_multi_seed_picks_best_calibrated(tmp_path, tiny_model):
    from lane_encoder_common import _run_main

    code, out_dir = _run_main(tmp_path, tiny_model, ["--seeds", "1,2"])
    assert code == 0
    seeds_doc = json.loads((out_dir / "seeds.json").read_text(encoding="utf-8"))
    assert len(seeds_doc["seeds"]) == 2
    assert "mean" in seeds_doc and "sd" in seeds_doc
    assert (out_dir / "seed-1" / "metrics.json").exists()
    assert (out_dir / "seed-2").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    per_seed = {m["seed"]: m for m in seeds_doc["seeds"]}
    expected = min(per_seed, key=lambda s: (
        abs(per_seed[s]["temperature"] - 1.0), -per_seed[s]["eval_accuracy"], s))
    assert metrics["chosen_seed"] == expected
    for name in ("label2id.json", "temperature.json"):
        chosen_dir = out_dir / f"seed-{expected}"
        assert (out_dir / name).read_bytes() == (chosen_dir / name).read_bytes()
    assert abs(metrics["eval_accuracy_mean"] - seeds_doc["mean"]["eval_accuracy"]) < 1e-9


def test_curve_writes_three_points_and_cleans_up(tmp_path, tiny_model):
    from lane_encoder_common import _run_main

    code, out_dir = _run_main(tmp_path, tiny_model, ["--curve"])
    assert code == 0
    curve = json.loads((out_dir / "learning_curve.json").read_text(encoding="utf-8"))
    assert [p["real_fraction"] for p in curve] == [0.0, 0.5, 1.0]
    assert [p["n_train"] for p in curve] == sorted(p["n_train"] for p in curve)
    assert len({p["n_train"] for p in curve}) == 3
    assert all(p["eval_per_lane"].keys() == {"lane_a", "lane_b", "lane_c"} for p in curve)
    assert not list(out_dir.glob("curve-*"))


def test_dry_run_prints_selection(tmp_path, tiny_model, capsys):
    from lane_encoder_common import _run_main, LABELS

    code, out_dir = _run_main(tmp_path, tiny_model, ["--dry-run"])
    assert code == 0
    assert not out_dir.exists()
    out = capsys.readouterr().out
    assert "selection" in out
    for lane in LABELS:
        assert f"lane {lane}:" in out


def test_multi_seed_refused_on_modal(tmp_path, tiny_model, monkeypatch, capsys):
    from lane_encoder_common import _run_main, evalroute_lane_encoder

    def _boom(*a, **k):
        raise AssertionError("modal entry point called")

    monkeypatch.setattr(
        evalroute_lane_encoder, "run_encoder_train_modal", _boom, raising=False)
    import thomas.encoder_train as et
    monkeypatch.setattr(et, "run_encoder_train_modal", _boom)
    code, out_dir = _run_main(
        tmp_path, tiny_model,
        ["--backend", "modal", "--seeds", "1,2", "--dry-run", "--yes"])
    assert code == 1
    assert "multi-seed is a local-backend feature" in capsys.readouterr().err
    assert not out_dir.exists()
