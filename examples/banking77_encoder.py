"""Banking77 encoder fine-tune driver — the domain example for thomas.encoder_train.

Builds the gonogo canary splits from the PolyAI Banking77 CSVs, dispatches
training via ``run_encoder_train_modal``, and writes the cases JSONL +
artifact dir the gonogo-side example (``banking77_encoder_eval.py``)
consumes. The split derivation is the cross-repo contract
(banking77-contract.md): the eval pilot must match
``gonogo/examples/banking77_routing.py`` case-for-case.

Split logic is COPIED from gonogo/examples/banking77_routing.py (the source
of truth) — no cross-repo imports. The calibration split (500 rows,
``random.Random(7)`` from train) is taken inside ``train_classifier`` by
passing ``seed=7`` with the full train set.

Usage::

    # Build splits + print config, no Modal call (safe, free):
    python examples/banking77_encoder.py --dry-run

    # Real run on a Modal L4 (asks for confirmation first — GPU credits):
    python examples/banking77_encoder.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

# --- Split constants (contract: banking77-contract.md) ------------------------
# Source of truth: gonogo/examples/banking77_routing.py. Must match exactly.

TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)
PILOT_SEED = 11    # gonogo canary seed — do not change
PILOT_SIZE = 250   # eval pilot size — do not change
CALIB_SEED = 7     # calib split seed (applied inside train_classifier)
CALIB_SIZE = 500   # rows held out from train for temperature scaling

DATA_DIR = Path(__file__).parent / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CASES_OUT = Path(__file__).parent / "banking77_cases.jsonl"
DEFAULT_MODEL = "johnnyboycurtis/ModernBERT-small-v2"


def fetch(url: str, dest: Path) -> Path:
    """Download url to dest unless already cached."""
    if dest.exists():
        print(f"  cached: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_rows(path: Path) -> list[tuple[str, str]]:
    """CSV with columns text,category -> (text, label) rows."""
    with open(path, encoding="utf-8", newline="") as f:
        return [(r["text"], r["category"]) for r in csv.DictReader(f)]


def build_splits():
    """Fetch the PolyAI CSVs (cached) and build train/calib/eval-pilot splits.

    Eval pilot: random.Random(11).sample(test, 250) with msg-%04d ids —
    copied from gonogo/examples/banking77_routing.py so both repos derive
    the identical canary. Calib: 500 rows from train via random.Random(7),
    taken inside train_classifier (seed=CALIB_SEED), never trained on.
    """
    train_csv = fetch(TRAIN_URL, DATA_DIR / "banking77_train.csv")
    test_csv = fetch(TEST_URL, DATA_DIR / "banking77_test.csv")
    train_rows = load_rows(train_csv)
    test_rows = load_rows(test_csv)

    # Copied from banking77_routing.py (source of truth):
    pilot = random.Random(PILOT_SEED).sample(test_rows, PILOT_SIZE)
    cases = [
        {"id": f"msg-{i:04d}", "text": text, "label": label}
        for i, (text, label) in enumerate(pilot)
    ]
    return train_rows, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--run-name", default="banking77-enc")
    parser.add_argument("--cases-out", default=str(CASES_OUT))
    parser.add_argument("--dry-run", action="store_true",
                        help="build splits + print config; NO Modal call")
    parser.add_argument("--yes", action="store_true",
                        help="skip the approval prompt (operator already approved)")
    args = parser.parse_args()

    print("=== banking77 encoder splits ===")
    train_rows, cases = build_splits()
    labels = sorted({label for _, label in train_rows})

    config = dict(
        model_name=args.model,
        num_labels=len(labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        calib_size=CALIB_SIZE,
        seed=CALIB_SEED,
    )

    # Reward-discrimination pre-flight analog: print the full launch
    # config before any GPU spend (thomas AGENTS.md rule 5).
    print(f"\n=== launch config (run_name={args.run_name}) ===")
    print(f"  model: {args.model} ({len(labels)} labels)")
    print(f"  train rows: {len(train_rows)} (calib held out: {CALIB_SIZE}, seed {CALIB_SEED})")
    print(f"  eval pilot: {len(cases)} cases (seed {PILOT_SEED})")
    print(f"  epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}")
    print(f"  Modal GPU: L4, timeout 3600s (expect <15 min, well under $1)")

    if args.dry_run:
        print("\n[dry-run] splits built, config above; no Modal call made.")
        return 0

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

    enc_config = EncoderTrainConfig(
        model_name=args.model,
        num_labels=len(labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        calib_size=CALIB_SIZE,
        seed=CALIB_SEED,
    )
    result = run_encoder_train_modal(train_rows, enc_config, args.run_name)

    # Cases JSONL for the gonogo-side example (contract format).
    cases_out = Path(args.cases_out)
    with open(cases_out, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"\n  wrote {len(cases)} cases -> {cases_out}")

    # Artifact dir copied locally from the Modal volume (contract format).
    local = pull_encoder_artifact(args.run_name, ARTIFACTS_DIR / args.run_name)
    print(f"  artifacts -> {local}")
    print(f"  temperature: {result.temperature:.4f}")
    print(f"  calib accuracy: {result.metrics['calib_accuracy']:.4f}")
    print(f"  ECE: {result.metrics['ece_before']:.4f} -> {result.metrics['ece_after']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
