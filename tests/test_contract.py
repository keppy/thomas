"""The contract surface: thomas.contract reads what encoder_train writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thomas.contract import ARTIFACT_SIDECARS, CONTRACT_VERSION, read_artifact

REPO = Path(__file__).resolve().parents[1]


def make_artifact(d: Path, *, version=CONTRACT_VERSION, temperature=0.8, label2id=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "label2id.json").write_text(json.dumps(label2id or {"a": 0, "b": 1}), encoding="utf-8")
    (d / "temperature.json").write_text(json.dumps({"temperature": temperature}), encoding="utf-8")
    m = {"calib_accuracy": 0.9}
    if version is not None:
        m["contract_version"] = version
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    return d


def test_reads_current_version(tmp_path):
    out = read_artifact(make_artifact(tmp_path / "a"))
    assert out["contract_version"] == CONTRACT_VERSION and out["temperature"] == 0.8


def test_missing_version_reads_as_1(tmp_path):
    assert read_artifact(make_artifact(tmp_path / "a", version=None))["contract_version"] == 1


def test_rejects_newer_version(tmp_path):
    with pytest.raises(ValueError, match="contract_version"):
        read_artifact(make_artifact(tmp_path / "a", version=CONTRACT_VERSION + 1))


@pytest.mark.parametrize("missing", ARTIFACT_SIDECARS)
def test_rejects_missing_sidecar(tmp_path, missing):
    d = make_artifact(tmp_path / "a")
    (d / missing).unlink()
    with pytest.raises(ValueError, match=missing):
        read_artifact(d)


def test_rejects_bad_temperature_and_gappy_ids(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        read_artifact(make_artifact(tmp_path / "t", temperature=0))
    with pytest.raises(ValueError, match="label2id"):
        read_artifact(make_artifact(tmp_path / "g", label2id={"a": 0, "b": 2}))


def test_shipped_banking77_artifact_meets_contract():
    art = REPO / "examples" / "artifacts" / "banking77-enc"
    if not art.is_dir():
        pytest.skip("artifact not present")
    out = read_artifact(art)
    assert out["contract_version"] == 1 and len(out["label2id"]) == 77


def test_contract_doc_states_the_version():
    doc = (REPO / "docs" / "CONTRACT.md").read_text(encoding="utf-8")
    assert f"Contract version: **{CONTRACT_VERSION}**" in doc
