"""Tests for examples/banking77_predict.py — mocked model/tokenizer, no downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from banking77_predict import (  # noqa: E402
    format_prediction,
    load_sidecars,
    predict_texts,
)

ID2LABEL = {0: "card_payment_not_recognised", 1: "top_up_failed",
            2: "card_payment_fee_charged", 3: "terminating"}


class TestLoadSidecars:
    def _write(self, tmp_path, temp):
        (tmp_path / "label2id.json").write_text(json.dumps({l: i for i, l in ID2LABEL.items()}))
        (tmp_path / "temperature.json").write_text(json.dumps({"temperature": temp}))
        return tmp_path

    def test_roundtrip(self, tmp_path):
        label2id, t = load_sidecars(self._write(tmp_path, 1.37))
        assert t == 1.37
        assert label2id == {l: i for i, l in ID2LABEL.items()}

    def test_rejects_nonpositive_temperature(self, tmp_path):
        with pytest.raises(ValueError, match="non-positive"):
            load_sidecars(self._write(tmp_path, 0.0))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sidecars(tmp_path)


class TestFormatPrediction:
    def test_block_format(self):
        probs = [0.913, 0.031, 0.018, 0.005]
        block = format_prediction("Why was I charged twice this month?", probs, ID2LABEL)
        lines = block.splitlines()
        assert lines[0] == "  Why was I charged twice this month?"
        # top label first, no indent, with percentage
        assert lines[1].strip().startswith("card_payment_not_recognised")
        assert lines[1].rstrip().endswith("91.3%")
        # runner-ups indented, in descending probability order
        assert lines[2].startswith("    ")
        assert "top_up_failed" in lines[2] and "3.1%" in lines[2]
        assert "card_payment_fee_charged" in lines[3] and "1.8%" in lines[3]
        assert len(lines) == 4  # top_k=3

    def test_top_k_limits_output(self):
        probs = [0.5, 0.2, 0.2, 0.1]
        block = format_prediction("msg", probs, ID2LABEL, top_k=2)
        assert len(block.splitlines()) == 3  # header + 2 labels

    def test_labels_sorted_by_probability(self):
        probs = [0.1, 0.2, 0.6, 0.1]
        block = format_prediction("msg", probs, ID2LABEL)
        assert "termining" not in block
        assert block.splitlines()[1].strip().startswith("card_payment_fee_charged")


class TestPredictTexts:
    def test_temperature_applied_before_softmax(self):
        """probs == softmax(logits / T) over the FULL distribution."""
        import torch

        temperature = 2.5
        logits = torch.tensor([[3.0, 1.0, 0.5, -1.0], [0.1, 0.2, 0.3, 0.4]])

        class FakeModel:
            def eval(self):
                return self

            def __call__(self, **enc):
                import types
                return types.SimpleNamespace(logits=logits)

        class FakeTokenizer:
            def __call__(self, texts, **kw):
                return {"input_ids": torch.zeros((len(texts), 4), dtype=torch.long),
                        "attention_mask": torch.ones((len(texts), 4), dtype=torch.long)}

        probs = predict_texts(FakeModel(), FakeTokenizer(),
                              ["a", "b"], temperature)
        expected = torch.softmax(logits / temperature, dim=-1)
        assert len(probs) == 2
        torch.testing.assert_close(torch.tensor(probs), expected)
        # full distribution: rows sum to 1
        assert all(abs(sum(row) - 1.0) < 1e-6 for row in probs)
