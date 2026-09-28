"""Public prediction receipt is complete and replayable without a model/GPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_banking77_receipt_replays_without_text():
    root = Path(__file__).resolve().parents[1] / "examples"
    spec = importlib.util.spec_from_file_location("banking77_receipt", root / "banking77_receipt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.read_predictions(module.RECEIPT)
    assert len(rows) == 250
    assert all(set(r) == {"id", "expected", "output", "confidence"} for r in rows)
    assert sum(r["expected"] == r["output"] for r in rows) == 218
    module.decision(rows)
    assert module._sha256(module.RECEIPT) == module._manifest()["predictions_jsonl_sha256"]


def test_provenance_rejects_modified_prediction_rows(tmp_path):
    root = Path(__file__).resolve().parents[1] / "examples"
    spec = importlib.util.spec_from_file_location("banking77_receipt", root / "banking77_receipt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(module.RECEIPT.read_bytes().replace(b"msg-0000", b"msg-9999"))
    with pytest.raises(ValueError, match="digest"):
        module.read_predictions(changed)


def test_local_model_and_cases_match_checked_provenance_when_present():
    root = Path(__file__).resolve().parents[1] / "examples"
    spec = importlib.util.spec_from_file_location("banking77_receipt", root / "banking77_receipt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_dir = root / "artifacts" / "banking77-enc"
    cases = root / "banking77_cases.jsonl"
    if model_dir.is_dir() and cases.is_file():
        module.verify_local_sources(model_dir, cases)
