"""Supervised encoder fine-tune path — SFT on Modal instead of Tinker RL.

Takes (text, label) pairs, fine-tunes a HuggingFace encoder with a
classification head, temperature-scales on a held-out calibration split,
and saves model + label map + temperature to a Modal volume or local dir.

This is the System-1 counterpart to trl_grpo.py: GRPO post-trains a
decoder via RL; this supervised-trains an encoder. Both save artifacts a
gonogo card can then rate.

Usage::

    # Local (CPU-testable core):
    from thomas.encoder_train import EncoderTrainConfig, train_classifier
    result = train_classifier(rows, config)

    # Modal (remote GPU, no local GPU needed):
    modal run -m thomas.encoder_train \\
        --model johnnyboycurtis/ModernBERT-small-v2 --data cases.jsonl

Confidence is defined once, here, and used everywhere (training eval,
saved metrics, and downstream gonogo agents): ``max(softmax(logits / T))``
— temperature applied to the logits BEFORE the softmax, softmax over the
FULL ``num_labels`` distribution every time. No top-1-vs-runner-up margin,
no sigmoid, no subset renormalization.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

import modal

from .contract import CONTRACT_VERSION

_enc_app = modal.App("thomas-encoder")

# Smaller than _grpo_image: no vLLM/TRL — just torch + transformers
# (>=4.48 for ModernBERT support). Build once, reuse across runs.
_enc_image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers>=4.48",
    "datasets",
    "numpy",
    "scipy",
)

_enc_vol = modal.Volume.from_name("thomas-encoder", create_if_missing=True)


# --- Config / result ---------------------------------------------------------


@dataclass
class EncoderTrainConfig:
    """Strict, few-knob config for the encoder fine-tune path.

    Attributes:
        model_name: HF Hub id or local path of the base encoder
        num_labels: number of classes (must equal the distinct labels in rows)
        epochs: training epochs over the (calib-held-out) train split
        batch_size: tokens-per-batch is capped by max_length; rows per step
        lr: AdamW learning rate
        max_length: tokenizer truncation length
        calib_size: rows held out from train for temperature scaling
            (never trained on); sampled with random.Random(seed)
        seed: drives the calib holdout and epoch shuffling
        output_dir: where save_pretrained + the JSON sidecars land
    """

    model_name: str
    num_labels: int
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    max_length: int = 128
    calib_size: int = 500
    seed: int = 0
    output_dir: str = "thomas-encoder-run"


@dataclass
class EncoderTrainResult:
    """The output of an encoder fine-tune run."""

    model_dir: str
    temperature: float
    label2id: dict[str, int]
    metrics: dict[str, Any] = field(default_factory=dict)


# --- Calibration (Guo et al. 2017) -------------------------------------------


def scaled_softmax(logits, temperature: float):
    """softmax(logits / temperature) over the FULL label distribution.

    The one confidence definition (cross-repo contract): temperature is
    applied to the logits BEFORE the softmax, and the softmax runs over
    every class — never a filtered subset.
    """
    import torch

    return torch.softmax(torch.as_tensor(logits, dtype=torch.float32) / temperature, dim=-1)


def confidence(logits, temperature: float) -> float:
    """max softmax probability after temperature scaling."""
    probs = scaled_softmax(logits, temperature)
    return float(probs.max(dim=-1).values)


def fit_temperature(logits, labels) -> float:
    """Fit one scalar T on held-out (calib) logits by minimizing NLL (L-BFGS).

    The calib split is never trained on. A NaN T means the model is
    poisoned (NaN loss upstream), not a calibration bug — this raises
    rather than masking it.
    """
    import torch

    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    labels_t = torch.as_tensor(labels)
    # Optimize log(T) so T > 0 by construction.
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        nll = torch.nn.functional.cross_entropy(logits_t / log_t.exp(), labels_t)
        nll.backward()
        return nll

    opt.step(closure)
    t = float(log_t.exp().item())
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"fit_temperature produced non-finite T={t!r} — "
                         "model is poisoned (see encoder-finetune skill: "
                         "never calibrate a model whose loss went NaN)")
    return t


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """ECE over equal-width confidence bins."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = [i for i in range(n) if (confidences[i] >= lo if b == 0 else confidences[i] > lo)
                  and confidences[i] <= hi]
        if in_bin:
            acc = sum(correct[i] for i in in_bin) / len(in_bin)
            conf = sum(confidences[i] for i in in_bin) / len(in_bin)
            ece += (len(in_bin) / n) * abs(acc - conf)
    return ece


# --- CPU-testable training core ----------------------------------------------


def _param_groups(model):
    """AdamW parameter groups with the encoder stability recipe baked in.

    betas=(0.9, 0.98) + eps=1e-6 + no weight decay on bias/LayerNorm.weight
    (DeBERTa-v3 lesson from the encoder-finetune skill: beta2=0.999 makes
    AdamW NaN on sensitive encoders at any lr; this recipe survives the
    first step where defaults diverge). Cheap insurance on ModernBERT too.
    """
    import torch

    no_decay = ("bias", "LayerNorm.weight")
    named = list(model.named_parameters())
    return [
        {
            "params": [p for n, p in named if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in named if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def train_classifier(rows: list[tuple[str, str]], config: EncoderTrainConfig) -> EncoderTrainResult:
    """Fine-tune an encoder classifier on (text, label) pairs.

    Holds out ``config.calib_size`` rows (seeded) for temperature scaling,
    trains with AdamW + the stability recipe, fits T on the calib split,
    and saves model + tokenizer + label2id/temperature/metrics JSON to
    ``config.output_dir``. CPU-testable; no Modal involved.
    """
    import os

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not rows:
        raise ValueError("no rows to train on")

    labels = sorted({label for _, label in rows})
    if len(labels) != config.num_labels:
        raise ValueError(
            f"num_labels={config.num_labels} but rows contain {len(labels)} "
            f"distinct labels: {labels[:5]}..."
        )
    label2id = {l: i for i, l in enumerate(labels)}
    if config.calib_size >= len(rows):
        raise ValueError(
            f"calib_size={config.calib_size} must be smaller than the "
            f"{len(rows)} rows given"
        )

    # Held-out calibration split (never trained on). Seeded so the
    # banking77 driver reproduces the contract's Random(7) split by
    # passing seed=7 with the full train set.
    rng = random.Random(config.seed)
    calib_idx = set(rng.sample(range(len(rows)), config.calib_size))
    train_rows = [r for i, r in enumerate(rows) if i not in calib_idx]
    calib_rows = [rows[i] for i in sorted(calib_idx)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)  # reproducible head init + dropout
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    ).to(device)
    # A freshly initialized classification head (MISSING keys on load) is
    # expected when the base checkpoint has no head — that is the point.

    def encode(batch_rows):
        texts = [t for t, _ in batch_rows]
        ys = torch.tensor([label2id[l] for _, l in batch_rows], device=device)
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=config.max_length, return_tensors="pt").to(device)
        return enc, ys

    # --- Sanity pass: forward -> backward -> step -> forward ---------------
    # A few seconds that save an hour: if the first step NaNs, the model
    # is poisoned and everything downstream (predictions, T, ECE) would
    # be garbage. Refuse to train rather than save a NaN model.
    model.train()
    groups = _param_groups(model)
    # betas=(0.9, 0.98), eps=1e-6, no WD on bias/LayerNorm.weight — the
    # DeBERTa-v3 stability recipe (see _param_groups).
    opt = torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.98), eps=1e-6)

    def _assert_finite(name: str, value: float):
        if not math.isfinite(value):
            raise RuntimeError(
                f"{name} is not finite ({value}) — aborting before the run; "
                "see the encoder-finetune NaN diagnosis recipe"
            )

    probe_enc, probe_y = encode(train_rows[: min(config.batch_size, len(train_rows))])
    loss0 = model(**probe_enc, labels=probe_y).loss
    _assert_finite("sanity first forward loss", float(loss0.detach()))
    loss0.backward()
    opt.step()
    opt.zero_grad()
    loss1 = model(**probe_enc, labels=probe_y).loss
    _assert_finite("sanity post-step loss", float(loss1.detach()))

    # --- Training loop ------------------------------------------------------
    step_losses: list[float] = []
    for epoch in range(config.epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            batch = [train_rows[i] for i in order[start:start + config.batch_size]]
            enc, ys = encode(batch)
            loss = model(**enc, labels=ys).loss
            _assert_finite(f"epoch {epoch} loss", float(loss.detach()))
            loss.backward()
            opt.step()
            opt.zero_grad()
            step_losses.append(float(loss.detach()))
        epoch_losses = step_losses[-(len(order) // config.batch_size + 1):]
        print(f"  epoch {epoch + 1}/{config.epochs}: "
              f"mean loss {sum(epoch_losses) / len(epoch_losses):.4f}")

    # --- Calib eval + temperature scaling ------------------------------------
    model.eval()
    calib_logits, calib_labels = [], []
    with torch.no_grad():
        for start in range(0, len(calib_rows), config.batch_size):
            batch = calib_rows[start:start + config.batch_size]
            enc, ys = encode(batch)
            calib_logits.append(model(**enc).logits.cpu())
            calib_labels.append(ys.cpu())
    logits = torch.cat(calib_logits)
    ys = torch.cat(calib_labels)

    temperature = fit_temperature(logits, ys)

    # Confidence = max(softmax(logits / T)) over the FULL distribution —
    # same scaled_softmax path everywhere (contract).
    preds = logits.argmax(dim=-1)
    correct = (preds == ys).tolist()
    conf_before = [float(scaled_softmax(l, 1.0).max()) for l in logits]
    conf_after = [float(scaled_softmax(l, temperature).max()) for l in logits]
    ece_before = expected_calibration_error(conf_before, correct)
    ece_after = expected_calibration_error(conf_after, correct)

    metrics = {
        "final_train_loss": step_losses[-1] if step_losses else float("nan"),
        "mean_train_loss": sum(step_losses) / len(step_losses),
        "step_losses": step_losses,
        "train_size": len(train_rows),
        "calib_size": len(calib_rows),
        "calib_accuracy": sum(correct) / len(correct),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "temperature": temperature,
        "num_labels": config.num_labels,
        "contract_version": CONTRACT_VERSION,
    }

    # --- Save ----------------------------------------------------------------
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    with open(os.path.join(config.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open(os.path.join(config.output_dir, "temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"temperature": temperature}, f, indent=2)
    with open(os.path.join(config.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return EncoderTrainResult(
        model_dir=config.output_dir,
        temperature=temperature,
        label2id=label2id,
        metrics=metrics,
    )


# --- Modal mode ---------------------------------------------------------------
#
# Mirrors eval_modal.py / trl_grpo.py: module-level app + function, JSON-
# serialized inputs, results as JSON strings. The remote function calls
# the same train_classifier core, so local CPU tests cover the training
# logic the GPU run executes.


@_enc_app.function(
    image=_enc_image,
    gpu="L4",
    timeout=3600,
    volumes={"/root/thomas-encoder": _enc_vol},
)
def _enc_train_remote(rows_json: str, config_json: str, run_name: str) -> str:
    """Run encoder fine-tuning on Modal. Called via .remote()."""
    rows = [tuple(r) for r in json.loads(rows_json)]
    config = EncoderTrainConfig(**json.loads(config_json))
    config.output_dir = f"/root/thomas-encoder/{run_name}"
    result = train_classifier(rows, config)
    return json.dumps({
        "model_dir": result.model_dir,
        "temperature": result.temperature,
        "label2id": result.label2id,
        "metrics": result.metrics,
    })


@_enc_app.function(
    image=_enc_image,
    timeout=300,
    volumes={"/root/thomas-encoder": _enc_vol},
)
def _enc_pull_artifact(run_name: str) -> str:
    """Read a run's artifact dir off the volume as {filename: base64}."""
    import base64
    from pathlib import Path

    root = Path("/root/thomas-encoder") / run_name
    if not root.exists():
        raise FileNotFoundError(f"no artifact at {root}")
    out = {}
    for p in sorted(root.iterdir()):
        if p.is_file():
            out[p.name] = base64.b64encode(p.read_bytes()).decode()
    return json.dumps(out)


def run_encoder_train_modal(
    rows: list[tuple[str, str]], config: EncoderTrainConfig, run_name: str
) -> EncoderTrainResult:
    """Dispatch encoder training to a Modal L4 GPU; returns the volume path.

    **Ask before spending credits** (AGENTS.md rule 5) — this launches a
    GPU job. Print the config and get operator approval first.
    """
    result_json: str
    with _enc_app.run():  # .remote() needs a running app context
        result_json = _enc_train_remote.remote(
            rows_json=json.dumps([list(r) for r in rows]),
            config_json=json.dumps(vars(config)),
            run_name=run_name,
        )
    return EncoderTrainResult(**json.loads(result_json))


def pull_encoder_artifact(run_name: str, local_dir: str):
    """Copy a run's artifacts from the Modal volume to a local dir."""
    import base64
    from pathlib import Path

    files: str
    with _enc_app.run():
        files = _enc_pull_artifact.remote(run_name)
    payload = json.loads(files)
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    for name, b64 in payload.items():
        (local / name).write_bytes(base64.b64decode(b64))
    return local


# --- CLI ----------------------------------------------------------------------
# modal run -m thomas.encoder_train --model johnnyboycurtis/ModernBERT-small-v2 \
#     --data cases.jsonl


def _read_rows_jsonl(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append((d["text"], d["label"]))
    return rows


@_enc_app.local_entrypoint()
def cli(
    model: str = "johnnyboycurtis/ModernBERT-small-v2",
    data: str = "",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    calib_size: int = 500,
    seed: int = 7,
    run_name: str = "",
):
    if not data:
        raise SystemExit("--data cases.jsonl is required (JSONL with text/label)")
    import time

    rows = _read_rows_jsonl(data)
    labels = sorted({l for _, l in rows})
    config = EncoderTrainConfig(
        model_name=model,
        num_labels=len(labels),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        calib_size=calib_size,
        seed=seed,
    )
    name = run_name or f"enc-{int(time.time())}"
    print(f"=== encoder train: {name} ===")
    print(f"  model: {model} ({config.num_labels} labels)")
    print(f"  rows: {len(rows)} (calib {calib_size}, seed {seed})")
    print(f"  epochs {epochs}, batch {batch_size}, lr {lr}, GPU L4")
    result = run_encoder_train_modal(rows, config, name)
    print(f"\n  model_dir (volume): {result.model_dir}")
    print(f"  temperature: {result.temperature:.4f}")
    print(f"  calib accuracy: {result.metrics['calib_accuracy']:.4f}")
    print(f"  ECE: {result.metrics['ece_before']:.4f} -> {result.metrics['ece_after']:.4f}")
