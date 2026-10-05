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

Selection rules
---------------

Training rows are selected before training: per lane, all real and seed rows
are kept, and ``aug:`` rows are capped at ``max(5, that lane's real count)``
(except a lane with zero real rows keeps all its aug rows — paraphrases are
the only thing that makes such a lane exist). Real rows are repeated
``--real-weight`` times so they outweigh the paraphrase dialect, and
``--seeds``/``--curve`` report mean ± sd so a single-seed swing on a small
eval set can't be mistaken for signal.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


def _prefix(cid: str) -> str:
    return "ledger" if ":" not in cid else cid.split(":", 1)[0]


def _row_kind(cid: str) -> str:
    p = _prefix(cid)
    if p == "aug":
        return "aug"
    if p == "seed":
        return "seed"
    return "real"


def select_training_rows(cases: list[dict], aug_cap: int | None = None,
                         real_weight: int = 3) -> tuple[list[tuple[str, str]], dict]:
    """Cap aug rows per lane against that lane's real rows, up-weight real rows.

    Real = any id whose prefix is not ``aug``/``seed``. Per lane: keep all
    real and seed rows; keep the first N aug rows in file order, where
    N = aug_cap if given else max(5, real_count); a lane with zero real rows
    keeps all its aug rows (scaffolding). Real rows are repeated real_weight
    times; aug and seed rows once. Returns (rows, selection_report).
    """
    per_lane: dict[str, dict] = {}
    for lane in sorted({c["label"] for c in cases}):
        per_lane[lane] = {"real": 0, "aug_kept": 0, "aug_dropped": 0, "seed": 0}

    real_by_lane: dict[str, list[dict]] = {}
    aug_by_lane: dict[str, list[dict]] = {}
    for c in cases:
        kind = _row_kind(c["id"])
        if kind == "real":
            real_by_lane.setdefault(c["label"], []).append(c)
        elif kind == "aug":
            aug_by_lane.setdefault(c["label"], []).append(c)

    rows: list[tuple[str, str]] = []
    for lane, report in per_lane.items():
        reals = real_by_lane.get(lane, [])
        augs = aug_by_lane.get(lane, [])
        cap = aug_cap if aug_cap is not None else max(5, len(reals))
        if len(reals) == 0:
            cap = len(augs)  # zero real rows: scaffolding, keep all aug
        kept, dropped = augs[:cap], augs[cap:]
        report.update(real=len(reals), aug_kept=len(kept), aug_dropped=len(dropped))
        rows.extend((c["text"], c["label"]) for c in kept)
        rows.extend((c["text"], c["label"]) for c in reals for _ in range(real_weight))
        seed_count = sum(1 for c in cases if c["label"] == lane and _row_kind(c["id"]) == "seed")
        report["seed"] = seed_count
        rows.extend((c["text"], c["label"]) for c in cases
                    if c["label"] == lane and _row_kind(c["id"]) == "seed")
    report = {"per_lane": per_lane, "real_weight": real_weight, "aug_cap": aug_cap}
    return rows, report


def _print_selection(report: dict) -> None:
    print("  selection (real_weight {rw}, aug_cap {ac}):".format(
        rw=report["real_weight"], ac=report["aug_cap"]))
    for lane, d in report["per_lane"].items():
        print(f"    lane {lane}: real {d['real']} x{report['real_weight']}, "
              f"aug kept {d['aug_kept']} (dropped {d['aug_dropped']}), seed {d['seed']}")


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


CURVE_FRACTIONS = [0.0, 0.5, 1.0]


def _train_one(rows: list[tuple[str, str]], config_kwargs: dict, train_sources,
               eval_cases, out_dir: str, device: str) -> dict:
    """Train once into out_dir, evaluate, return the seed-metric subset."""
    import shutil

    from thomas.encoder_train import EncoderTrainConfig, train_classifier

    config = EncoderTrainConfig(**{**config_kwargs, "output_dir": out_dir})
    t0 = time.time()
    result = train_classifier(rows, config)
    elapsed = time.time() - t0
    evaluate(out_dir, eval_cases, {**config_kwargs, "train_sources": train_sources},
             elapsed, device)
    metrics_path = Path(out_dir) / "metrics.json"
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    subset = {k: metrics[k] for k in (
        "eval_accuracy", "eval_per_lane", "defer_below", "coverage_at_defer",
        "calib_accuracy", "temperature")}
    return subset


def _curve_real_subset(train_cases: list[dict], fraction: float,
                       seed: int) -> list[dict]:
    """Seeded shuffle over real ids, keep fraction of them; aug/seed rows kept."""
    import random

    real = [c for c in train_cases if _row_kind(c["id"]) == "real"]
    other = [c for c in train_cases if _row_kind(c["id"]) != "real"]
    ids = sorted(c["id"] for c in real)
    rng = random.Random(f"{seed}-curve")
    rng.shuffle(ids)
    n = int(round(fraction * len(ids)))
    keep = set(ids[:n])
    return [c for c in real if c["id"] in keep] + other


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
    parser.add_argument("--seeds", default=None,
                        help="comma list of seeds (default: [--seed])")
    parser.add_argument("--aug-cap", type=int, default=None,
                        help="per-lane aug row cap (default: max(5, lane real count))")
    parser.add_argument("--real-weight", type=int, default=3,
                        help="times each real row is repeated in training (>= 1)")
    parser.add_argument("--curve", action="store_true",
                        help="train {0,50,100}%% of real rows first, write learning_curve.json")
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
    if args.real_weight < 1:
        print(f"refusing: --real-weight must be >= 1, got {args.real_weight}",
              file=sys.stderr)
        return 1
    seeds = ([args.seed] if args.seeds is None
             else [int(s) for s in args.seeds.split(",") if s.strip()])
    if args.backend == "modal" and len(seeds) > 1:
        print("refusing: multi-seed is a local-backend feature; run Modal seeds one at a time",
              file=sys.stderr)
        return 1
    selection_rows, selection = select_training_rows(
        train_cases, args.aug_cap, args.real_weight)
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
    _print_selection(selection)
    if args.curve:
        print(f"  curve: will pre-train on {len(CURVE_FRACTIONS)} real-row fractions "
              f"{CURVE_FRACTIONS} (first seed only)")
    if len(seeds) > 1:
        print(f"  seeds: {seeds}")
    if args.dry_run:
        if errors:
            print("[dry-run] refusal checks above WOULD fail; would exit 1.")
        else:
            print("[dry-run] checks pass; nothing trained, nothing written.")
        return 0

    if args.backend == "local":
        from thomas.encoder_train import EncoderTrainConfig, train_classifier

        train_sources = config_kwargs.pop("train_sources")

        # Learning curve first (--curve): extra models on reduced real rows.
        if args.curve:
            curve_points = []
            for frac in CURVE_FRACTIONS:
                subset_cases = _curve_real_subset(train_cases, frac, seeds[0])
                cur_rows, _ = select_training_rows(
                    subset_cases, args.aug_cap, args.real_weight)
                curve_dir = str(Path(args.out) / f"curve-{int(frac * 100)}")
                sub = _train_one(cur_rows, {**config_kwargs}, train_sources,
                                 eval_cases, curve_dir, device)
                curve_points.append({
                    "real_fraction": frac, "n_train": len(cur_rows),
                    "eval_accuracy": sub["eval_accuracy"],
                    "eval_per_lane": sub["eval_per_lane"],
                })
                print(f"  curve {int(frac * 100)}%: n_train={len(cur_rows)} "
                      f"eval_accuracy={sub['eval_accuracy']:.4f}")
                shutil.rmtree(curve_dir, ignore_errors=True)
            Path(args.out).mkdir(parents=True, exist_ok=True)
            with open(Path(args.out) / "learning_curve.json", "w", encoding="utf-8") as f:
                json.dump(curve_points, f, indent=2)

        seed_metrics: list[dict] = []
        for s in seeds:
            rows, sel_report = select_training_rows(
                train_cases, args.aug_cap, args.real_weight)
            seed_dir = str(Path(args.out) / f"seed-{s}") if len(seeds) > 1 else args.out
            if len(seeds) == 1:
                _print_selection(sel_report)
            sub = _train_one(rows, {**config_kwargs, "seed": s}, train_sources,
                             eval_cases, seed_dir, device)
            sub["seed"] = s
            seed_metrics.append(sub)
            print(f"  seed {s}: eval_accuracy={sub['eval_accuracy']:.4f} "
                  f"temperature={sub['temperature']:.4f}")

        def _pop_sd(vals: list[float]) -> float:
            if len(vals) < 2:
                return 0.0
            mu = sum(vals) / len(vals)
            return (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5

        seeds_path = Path(args.out) / "seeds.json"
        if len(seeds) > 1:
            seeds_doc: dict = {"seeds": seed_metrics}
            seeds_doc["mean"] = {"eval_accuracy": sum(
                m["eval_accuracy"] for m in seed_metrics) / len(seed_metrics)}
            seeds_doc["sd"] = {"eval_accuracy": _pop_sd(
                [m["eval_accuracy"] for m in seed_metrics])}
            lanes = sorted(seed_metrics[0]["eval_per_lane"])
            seeds_doc["mean"]["per_lane_acc"] = {
                l: sum(m["eval_per_lane"][l]["acc"] for m in seed_metrics) / len(seed_metrics)
                for l in lanes}
            seeds_doc["sd"]["per_lane_acc"] = {
                l: _pop_sd([m["eval_per_lane"][l]["acc"] for m in seed_metrics])
                for l in lanes}
            with open(seeds_path, "w", encoding="utf-8") as f:
                json.dump(seeds_doc, f, indent=2)
            best = min(seed_metrics, key=lambda m: (
                abs(m["temperature"] - 1.0), -m["eval_accuracy"], m["seed"]))
            chosen = best["seed"]
            # Copy the chosen seed's artifact up to <out>/ (seed dirs remain).
            for item in Path(args.out, f"seed-{chosen}").iterdir():
                if item.is_file():
                    shutil.copy2(item, Path(args.out) / item.name)
            mean_acc = seeds_doc["mean"]["eval_accuracy"]
            sd_acc = seeds_doc["sd"]["eval_accuracy"]
            print(f"seeds: {len(seed_metrics)}  eval acc {mean_acc:.4f} ± {sd_acc:.4f}  "
                  f"chosen seed {chosen} (|T−1| = {abs(best['temperature'] - 1.0):.4f})")
        else:
            chosen = seeds[0]
            with open(seeds_path, "w", encoding="utf-8") as f:
                json.dump({"seeds": seed_metrics}, f, indent=2)
            mean_acc = seed_metrics[0]["eval_accuracy"]
            sd_acc = 0.0
        metrics_path = Path(args.out) / "metrics.json"
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["seeds"] = seeds
        metrics["chosen_seed"] = chosen
        metrics["eval_accuracy_mean"] = mean_acc
        metrics["eval_accuracy_sd"] = sd_acc
        metrics["selection"] = selection
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
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
