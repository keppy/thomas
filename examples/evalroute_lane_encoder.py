"""evalroute lane-encoder recipe — train the lane classifier's first layer.

evalroute (the sibling library) exports its task→lane training rows in
thomas's encoder-case shape (docs/CONTRACT.md §1: ``{"id", "text", "label"}``)
and consumes the §4 artifact. thomas owns the recipe, evalroute owns the
data and the routing decision. No lane logic lives here.

Usage::

    # Local (default): train in-process, evaluate, write the artifact dir.
    python examples/evalroute_lane_encoder.py --train train.jsonl --eval eval.jsonl --out runs/lane-1

    # Print config + refusal checks; train nothing:
    python examples/evalroute_lane_encoder.py --train t.jsonl --eval e.jsonl --out runs/x --dry-run

    # Modal GPU run (costs credits; asks for confirmation unless --yes):
    python examples/evalroute_lane_encoder.py --train t.jsonl --eval e.jsonl --out runs/x --backend modal --yes

Never prints row text — counts and ids only (evalroute's rows are a
user's task text). The eval pass always runs in-process on this machine,
even when training went to Modal.

Borrowed GPU box
----------------

The recipe is a plain script on any machine with torch; a borrowed GPU
box needs no Modal account:

    pip install "thomas-train[encoder]"
    python examples/evalroute_lane_encoder.py --train train.jsonl --eval eval.jsonl --out runs/lane-1
    # copy the runs/lane-1 directory back (model + sidecars + eval metrics)

No account, no credentials — the box just needs the two JSONL files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

DEFAULT_MODEL = "johnnyboycurtis/ModernBERT-small-v2"
RECIPE = "evalroute_lane_encoder"


def _read_cases(path: str) -> list[dict]:
    """Read §1 encoder-case rows, keeping id (train_classifier drops it)."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cases.append({"id": d["id"], "text": d["text"], "label": d["label"]})
    return cases


def _sources(cases: list[dict]) -> dict[str, int]:
    """Counts by id prefix before the first ':'; 'ledger' when no prefix."""
    counts: Counter[str] = Counter()
    for c in cases:
        cid = c["id"]
        counts["ledger" if ":" not in cid else cid.split(":", 1)[0]] += 1
    return dict(sorted(counts.items()))


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _defer_threshold(confs: list[float], correct: list[bool], target: float = 0.9):
    """Smallest threshold with accuracy among non-abstained rows >= target.

    Returns (defer_below, coverage_at_defer): the fraction of rows kept.
    (1.0, 0.0) when accuracy never reaches the target.
    """
    pairs = sorted(zip(confs, correct))
    n = len(pairs)
    kept = 0
    kept_correct = 0
    best = None
    # Walk from the lowest-confidence row upward: at each cut we keep the
    # rows above it (the tail of the sorted list).
    for i in range(n):
        kept += 1
        kept_correct += 1 if pairs[n - 1 - i][1] else 0
        kept_conf = pairs[n - 1 - i][0]
        if kept_correct / kept >= target:
            best = (kept_conf, kept / n)
    if best is None:
        return 1.0, 0.0
    return best


def evaluate(out_dir: str, cases: list[dict], config_kwargs: dict, elapsed_s: float, device: str):
    """Load the artifact from out_dir and evaluate on this machine."""
    import torch
    from thomas.encoder_train import scaled_softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    with open(Path(out_dir) / "label2id.json", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(Path(out_dir) / "temperature.json", encoding="utf-8") as f:
        temperature = json.load(f)["temperature"]
    id2label = {i: l for l, i in label2id.items()}

    confs: list[float] = []
    preds: list[str] = []
    model.eval()
    with torch.no_grad():
        for c in cases:
            enc = tokenizer(c["text"], truncation=True, return_tensors="pt")
            logits = model(**enc).logits[0]
            confs.append(float(scaled_softmax(logits, temperature).max()))
            preds.append(id2label[int(logits.argmax())])

    labels = sorted(label2id)
    per_lane: dict[str, dict] = {}
    for lane in labels:
        rows = [(c["label"], p) for c, p in zip(cases, preds) if c["label"] == lane]
        per_lane[lane] = {
            "n": len(rows),
            "acc": sum(l == p for l, p in rows) / len(rows) if rows else 0.0,
        }
    defer_below, coverage = _defer_threshold(
        confs, [c["label"] == p for c, p in zip(cases, preds)]
    )

    predictions_path = Path(out_dir) / "eval_predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for c, p, conf in zip(cases, preds, confs):
            f.write(json.dumps({
                "id": c["id"], "label": c["label"],
                "predicted": p, "confidence": conf,
            }) + "\n")

    metrics_path = Path(out_dir) / "metrics.json"
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    metrics.update({
        "eval_accuracy": sum(c["label"] == p for c, p in zip(cases, preds)) / len(cases),
        "eval_n": len(cases),
        "eval_per_lane": per_lane,
        "defer_below": defer_below,
        "coverage_at_defer": coverage,
        "train_sources": _sources(cases),  # replaced with train ids below
        "contract_version": metrics.get("contract_version", 1),
        "recipe": RECIPE,
        "model": config_kwargs["model_name"],
        "epochs": config_kwargs["epochs"],
        "lr": config_kwargs["lr"],
        "seed": config_kwargs["seed"],
        "device": device,
        "elapsed_s": round(elapsed_s, 2),
    })
    metrics["train_sources"] = config_kwargs.pop("train_sources")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  eval: {metrics['eval_n']} rows, "
          f"accuracy {metrics['eval_accuracy']:.4f}")
    for lane, d in sorted(per_lane.items()):
        print(f"    lane {lane}: n={d['n']} acc={d['acc']:.4f}")
    print(f"  defer_below: {defer_below:.4f} "
          f"(coverage {coverage:.4f}; accuracy >= 0.9 among non-abstained)")
    print(f"  wrote {predictions_path} and updated {metrics_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True, dest="eval_path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calib-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", choices=("local", "modal"), default="local")
    parser.add_argument("--run-name", default="evalroute-lane")
    parser.add_argument("--dry-run", action="store_true",
                        help="print config + refusal checks; train nothing")
    parser.add_argument("--yes", action="store_true",
                        help="skip the approval prompt (operator already approved)")
    args = parser.parse_args()

    train_cases = _read_cases(args.train)
    eval_cases = _read_cases(args.eval_path)
    labels = sorted({c["label"] for c in train_cases})
    device = str(_pick_device())

    # Refusal checks (printed even on --dry-run).
    errors: list[str] = []
    if not train_cases:
        errors.append(f"no train rows in {args.train}")
    if not eval_cases:
        errors.append(f"no eval rows in {args.eval_path}")
    if args.calib_size * 2 >= len(train_cases) and train_cases:
        errors.append(
            f"--calib-size {args.calib_size} must be < 50% of the "
            f"{len(train_cases)} train rows"
        )
    if errors:
        for e in errors:
            print(f"refusing: {e}", file=sys.stderr)
        if not args.dry_run:
            return 1

    train_counts = dict(sorted(Counter(c["label"] for c in train_cases).items()))
    config_kwargs = dict(
        model_name=args.model, num_labels=len(labels),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        calib_size=args.calib_size, seed=args.seed, output_dir=args.out,
        train_sources=_sources(train_cases),
    )

    print("=== evalroute lane-encoder ===")
    print(f"  backend: {args.backend}, device: {device}")
    print(f"  train rows: {len(train_cases)}, eval rows: {len(eval_cases)}")
    print(f"  labels ({len(labels)}): {labels}")
    print(f"  train per-label counts: {train_counts}")
    print(f"  sources: {config_kwargs['train_sources']}")
    print(f"  model: {args.model}, epochs: {args.epochs}, lr: {args.lr}, "
          f"batch_size: {args.batch_size}, calib_size: {args.calib_size}, seed: {args.seed}")
    if args.dry_run:
        if errors:
            print("[dry-run] refusal checks above WOULD fail; would exit 1.")
        else:
            print("[dry-run] checks pass; nothing trained, nothing written.")
        return 0

    if args.backend == "local":
        from thomas.encoder_train import EncoderTrainConfig, train_classifier

        train_sources = config_kwargs.pop("train_sources")
        config = EncoderTrainConfig(**config_kwargs)
        t0 = time.time()
        result = train_classifier(
            [(c["text"], c["label"]) for c in train_cases], config
        )
        elapsed = time.time() - t0
        print(f"  trained on {device} in {elapsed:.1f}s "
              f"(temperature {result.temperature:.4f}, "
              f"calib acc {result.metrics['calib_accuracy']:.4f})")
        evaluate(args.out, eval_cases, {**config_kwargs, "train_sources": train_sources},
                 elapsed, device)
    else:
        if not args.yes:
            answer = input("\nProceed with the Modal GPU run? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("aborted — no Modal call made.")
                return 1
        from thomas.encoder_train import (
            EncoderTrainConfig,
            pull_encoder_artifact,
            run_encoder_train_modal,
        )

        train_sources = config_kwargs.pop("train_sources")
        config = EncoderTrainConfig(**config_kwargs)
        print("=== Modal launch (L4 GPU — costs credits) ===")
        for k, v in config_kwargs.items():
            print(f"  {k}: {v}")
        t0 = time.time()
        run_encoder_train_modal(
            [(c["text"], c["label"]) for c in train_cases], config, args.run_name
        )
        pull_encoder_artifact(args.run_name, args.out)
        elapsed = time.time() - t0
        print(f"  artifacts pulled to {args.out}; eval runs locally on {device}")
        evaluate(args.out, eval_cases, {**config_kwargs, "train_sources": train_sources},
                 elapsed, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
