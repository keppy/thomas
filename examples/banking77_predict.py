"""Banking77 demo inference CLI — interactive predictions from a trained encoder.

Consumes an exported artifact dir per the cross-repo contract: an HF
``save_pretrained`` output plus ``label2id.json`` and
``temperature.json``. Confidence is the one contract definition —
``max(softmax(logits / T))`` over the FULL label distribution, with T
applied to the logits BEFORE the softmax — via the same
``thomas.encoder_train.scaled_softmax`` code path used in training eval.

Usage::

    # one or more messages (batched in a single tokenizer call):
    python examples/banking77_predict.py --model-dir artifacts/run1 \\
        "Why was I charged twice this month?"

    # no messages -> interactive REPL, one prediction per line:
    python examples/banking77_predict.py --model-dir artifacts/run1

CPU-only (model.eval() + torch.inference_mode, no Modal).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOP_K_DEFAULT = 3


def load_sidecars(model_dir: str | Path) -> tuple[dict[str, int], float]:
    """Read label2id.json + temperature.json from an artifact dir.

    Stdlib-only so tests (and misconfigured environments) can check the
    contract files without touching torch.
    """
    d = Path(model_dir)
    label2id = json.loads((d / "label2id.json").read_text(encoding="utf-8"))
    temperature = json.loads((d / "temperature.json").read_text(encoding="utf-8"))["temperature"]
    temperature = float(temperature)
    if not (temperature > 0):
        raise ValueError(f"temperature.json has non-positive T={temperature!r} in {d}")
    return label2id, temperature


def format_prediction(text: str, probs, id2label: dict[int, str], top_k: int = _TOP_K_DEFAULT) -> str:
    """One clean demo block: message, top label + confidence, runner-ups.

    ``probs`` is the full-distribution softmax over all labels (already
    temperature-scaled by the caller).
    """
    ranked = sorted(range(len(probs)), key=lambda i: -probs[i])[:top_k]
    shown = [(id2label[i], probs[i] * 100.0) for i in ranked]
    width = max([24] + [len(label) + 2 for label, _ in shown])
    lines = [f"  {text}"]
    for j, (label, pct) in enumerate(shown):
        prefix = "" if j == 0 else "  "
        lines.append(f"  {prefix}{label:<{width}}{pct:5.1f}%")
    return "\n".join(lines)


def predict_texts(model, tokenizer, texts: list[str], temperature: float) -> list[list[float]]:
    """Full-distribution probabilities for a batch of messages.

    Confidence contract: softmax(logits / T) over ALL labels, T applied
    before the softmax — the shared ``scaled_softmax`` path.
    """
    import torch

    from thomas.encoder_train import scaled_softmax

    enc = tokenizer(texts, padding=True, truncation=True, max_length=128,
                    return_tensors="pt")
    model.eval()
    with torch.inference_mode():
        logits = model(**enc).logits
    probs = scaled_softmax(logits, temperature)
    return probs.tolist()


def _load_model(model_dir: str | Path):
    """Lazily import torch/transformers with a helpful error message."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:  # pragma: no cover - environment guard
        raise SystemExit(
            "transformers/torch not installed — run "
            "`pip install thomas-train[encoder]` (" + str(e) + ")"
        )
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return model, tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="exported artifact dir (save_pretrained + label2id.json + temperature.json)")
    parser.add_argument("--top-k", type=int, default=_TOP_K_DEFAULT)
    parser.add_argument("messages", nargs="*",
                        help="messages to classify; if none, an interactive REPL starts")
    args = parser.parse_args(argv)

    if not Path(args.model_dir).is_dir():
        raise SystemExit(f"--model-dir {args.model_dir} does not exist")

    label2id, temperature = load_sidecars(args.model_dir)
    id2label = {i: l for l, i in label2id.items()}
    model, tokenizer = _load_model(args.model_dir)

    def show(texts: list[str]) -> None:
        for text, probs in zip(texts, predict_texts(model, tokenizer, texts, temperature)):
            print(format_prediction(text, probs, id2label, top_k=args.top_k))
            print()

    if args.messages:
        show(args.messages)  # one tokenizer call for the whole batch
        return 0

    # Interactive REPL: one message at a time, friendly prompt.
    print(f"banking77 predictor ({args.model_dir}, T={temperature:.3f}) — "
          "type a message, or 'quit' / Ctrl-D to exit")
    while True:
        try:
            line = input("msg> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            return 0
        show([line])


if __name__ == "__main__":
    sys.exit(main())
