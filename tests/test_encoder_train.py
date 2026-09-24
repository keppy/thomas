"""Tests for thomas.encoder_train — CPU-only, no network.

Builds a tiny random BERT locally (no Hub download), trains 1 epoch on a
synthetic 2-class dataset, and checks the encoder-finetune sanity
discipline: finite losses at every step, finite temperature > 0, correct
label2id, and a saved dir that reloads via from_pretrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thomas.encoder_train import (
    EncoderTrainConfig,
    confidence,
    expected_calibration_error,
    fit_temperature,
    scaled_softmax,
    train_classifier,
)

# --- A tiny offline model ------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A 1-layer, 32-hidden BERT saved to disk — no Hub access needed."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        BertConfig,
        BertModel,
        PreTrainedTokenizerFast,
    )

    words = ["card", "lost", "stolen", "freeze", "transfer", "money", "send", "abroad",
             "help", "please", "my", "the", "i", "want", "to", "bank", "account"]
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]"
    )

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    torch.manual_seed(0)
    model = BertModel(config)

    d = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(str(d))
    tokenizer.save_pretrained(str(d))
    return str(d)


@pytest.fixture()
def tiny_rows():
    """~40 short strings, 2 classes with keyword signal."""
    rows = []
    for i in range(20):
        rows.append((f"i lost my card please freeze it {i}", "lost_card"))
        rows.append((f"i want to transfer money abroad {i}", "transfer"))
    return rows


# --- Tests ----------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        c = EncoderTrainConfig(model_name="m", num_labels=2)
        assert c.epochs == 3
        assert c.batch_size == 32
        assert c.lr == 2e-5
        assert c.max_length == 128
        assert c.calib_size == 500
        assert c.seed == 0

    def test_json_roundtrip(self):
        """The Modal wrapper serializes configs as vars() -> JSON."""
        c = EncoderTrainConfig(model_name="m", num_labels=77, seed=7)
        c2 = EncoderTrainConfig(**json.loads(json.dumps(vars(c))))
        assert c2 == c

    def test_num_labels_mismatch_raises(self, tiny_model_dir, tiny_rows):
        config = EncoderTrainConfig(
            model_name=tiny_model_dir, num_labels=3, epochs=1,
            batch_size=4, calib_size=8,
        )
        with pytest.raises(ValueError, match="num_labels=3"):
            train_classifier(tiny_rows, config)

    def test_calib_too_large_raises(self, tiny_model_dir, tiny_rows):
        config = EncoderTrainConfig(
            model_name=tiny_model_dir, num_labels=2, epochs=1,
            calib_size=500,
        )
        with pytest.raises(ValueError, match="calib_size"):
            train_classifier(tiny_rows, config)


class TestTrainClassifier:
    def test_train_classifier(self, tiny_model_dir, tiny_rows, tmp_path):
        out = str(tmp_path / "run")
        config = EncoderTrainConfig(
            model_name=tiny_model_dir,
            num_labels=2,
            epochs=1,
            batch_size=4,
            lr=2e-4,
            calib_size=8,
            seed=3,
            output_dir=out,
        )
        result = train_classifier(tiny_rows, config)

        # Every training step loss is finite (the NaN sanity discipline).
        assert result.metrics["step_losses"], "no steps were recorded"
        assert all(loss == loss and abs(loss) != float("inf")
                   for loss in result.metrics["step_losses"])

        # Temperature is finite and strictly positive.
        assert result.temperature > 0
        assert result.temperature == result.temperature  # not NaN

        # label2id maps both labels to distinct ids.
        assert set(result.label2id) == {"lost_card", "transfer"}
        assert sorted(result.label2id.values()) == [0, 1]

        # Metrics sanity.
        assert 0.0 <= result.metrics["calib_accuracy"] <= 1.0
        assert 0.0 <= result.metrics["ece_before"] <= 1.0
        assert 0.0 <= result.metrics["ece_after"] <= 1.0
        assert result.metrics["train_size"] == 32
        assert result.metrics["calib_size"] == 8

        # The saved dir reloads via from_pretrained and carries the
        # contract's JSON sidecars.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        reloaded = AutoModelForSequenceClassification.from_pretrained(out)
        assert reloaded.config.num_labels == 2
        AutoTokenizer.from_pretrained(out)  # must not raise
        label2id = json.loads((Path(out) / "label2id.json").read_text())
        assert label2id == result.label2id
        temp = json.loads((Path(out) / "temperature.json").read_text())
        assert temp["temperature"] == result.temperature
        metrics = json.loads((Path(out) / "metrics.json").read_text())
        from thomas.contract import CONTRACT_VERSION, read_artifact
        assert metrics["contract_version"] == CONTRACT_VERSION
        assert read_artifact(out)["temperature"] > 0
        assert metrics["calib_accuracy"] == result.metrics["calib_accuracy"]

    def test_calib_split_is_seeded(self, tiny_model_dir, tiny_rows, tmp_path):
        """Same seed -> same held-out calib split (contract reproducibility)."""
        outs = []
        for i in range(2):
            out = str(tmp_path / f"run{i}")
            config = EncoderTrainConfig(
                model_name=tiny_model_dir, num_labels=2, epochs=1,
                batch_size=4, lr=2e-4, calib_size=8, seed=3, output_dir=out,
            )
            train_classifier(tiny_rows, config)
            outs.append(out)
        # Identical seed + rows -> identical final metrics (deterministic).
        m0 = json.loads((Path(outs[0]) / "metrics.json").read_text())
        m1 = json.loads((Path(outs[1]) / "metrics.json").read_text())
        assert m0["step_losses"] == m1["step_losses"]

    def test_sanity_pass_catches_nan(self, tiny_model_dir, tiny_rows, tmp_path, monkeypatch):
        """A NaN first loss aborts before any real training (poisoned model)."""
        import torch

        config = EncoderTrainConfig(
            model_name=tiny_model_dir, num_labels=2, epochs=1,
            batch_size=4, calib_size=8, output_dir=str(tmp_path / "nan"),
        )

        class Poisoned(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def __call__(self, *a, **kw):
                out = self.inner(*a, **kw)
                out.loss = torch.tensor(float("nan"))
                return out

        # Patch at the source: train_classifier's from-import resolves
        # through the concrete class (Auto* dispatches to it).
        from transformers import BertForSequenceClassification

        orig = BertForSequenceClassification.from_pretrained.__func__

        def fake(cls, *a, **k):
            return Poisoned(orig(cls, *a, **k))

        monkeypatch.setattr(
            BertForSequenceClassification, "from_pretrained", classmethod(fake)
        )
        with pytest.raises(RuntimeError, match="not finite"):
            train_classifier(tiny_rows, config)


class TestCalibration:
    def test_fit_temperature_confident_correct(self):
        """Confident + always-correct logits -> T <= 1 (NLL rewards sharpening)."""
        import torch

        torch.manual_seed(0)
        logits = torch.randn(200, 2) * 6  # confident
        labels = logits.argmax(dim=-1)     # always correct by construction
        t = fit_temperature(logits, labels)
        assert 0 < t <= 1.0

    def test_fit_temperature_overconfident(self):
        """Overconfident wrong-ish logits -> T > 1 flattens them."""
        import torch

        torch.manual_seed(0)
        logits = torch.randn(200, 2) * 20  # very overconfident
        labels = (logits.argmax(dim=-1) + 1) % 2  # always wrong
        t = fit_temperature(logits, labels)
        assert t > 1.0

    def test_scaled_softmax_full_distribution(self):
        """Confidence = max(softmax(logits / T)) over the FULL distribution,
        temperature applied BEFORE the softmax (contract requirement)."""
        logits = [[10.0, 0.0, 0.0]]
        # T=2 halves the logit gap before softmax.
        expected = scaled_softmax([[5.0, 0.0, 0.0]], 1.0)[0]
        got = scaled_softmax(logits, 2.0)[0]
        assert abs(float(got[0]) - float(expected[0])) < 1e-6
        assert abs(float(got.sum()) - 1.0) < 1e-6  # full-distribution softmax

    def test_confidence(self):
        logits = [[3.0, 1.0]]
        c = confidence(logits, 1.0)
        assert 0.5 < c < 1.0
        # Higher T -> lower confidence (flatter).
        assert confidence(logits, 5.0) < c

    def test_ece(self):
        # Perfectly calibrated: confidence 0.75, accuracy 0.75.
        conf = [0.75] * 100
        correct = [True] * 75 + [False] * 25
        assert expected_calibration_error(conf, correct) == pytest.approx(0.0)
        # Miscalibrated: confidence 1.0, accuracy 0.5.
        conf = [1.0] * 100
        correct = [True] * 50 + [False] * 50
        assert expected_calibration_error(conf, correct) == pytest.approx(0.5)
