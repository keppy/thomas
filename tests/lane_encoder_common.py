"""Shared bootstrap for lane-encoder tests: loads the example script as a module."""

from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "evalroute_lane_encoder.py"

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


def _run_main(tmp_path, tiny_model, argv_extra: list[str]):
    """Run the script's main() with argv patched; returns (exit_code, out_dir)."""
    import json
    import sys

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
