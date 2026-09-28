"""Replay the public Banking77 prediction receipt, or verify it with a local artifact.

The receipt has no customer text and is not a downloadable model. To verify
model inference, supply the privately retained model dir and the 250 test
cases derived by banking77_encoder.build_splits(); no Modal calls are made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

RECEIPT = Path(__file__).parent / "receipts" / "banking77_predictions.jsonl"
PROVENANCE = RECEIPT.with_name("banking77_provenance.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest() -> dict:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def verify_local_sources(model_dir: Path, cases_path: Path) -> None:
    provenance = _manifest()
    for filename, digest in provenance["model_files_sha256"].items():
        if _sha256(model_dir / filename) != digest:
            raise ValueError(f"local model file {filename} does not match the checked receipt provenance")
    if _sha256(cases_path) != provenance["source"]["cases_jsonl_sha256"]:
        raise ValueError("local held-out cases do not match the checked receipt provenance")


def read_predictions(path: Path) -> list[dict]:
    if _sha256(path) != _manifest()["predictions_jsonl_sha256"]:
        raise ValueError("prediction receipt differs from its checked provenance digest")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [r["id"] for r in rows]
    if len(rows) != 250 or len(set(ids)) != 250:
        raise ValueError("receipt must have 250 unique case ids")
    for r in rows:
        p = r["confidence"]
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid confidence")
    return rows


def predict(model_dir: Path, cases_path: Path) -> list[dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from thomas.encoder_train import scaled_softmax

    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    labels = json.loads((model_dir / "label2id.json").read_text(encoding="utf-8"))
    id2label = {int(idx): label for label, idx in labels.items()}
    temperature = json.loads((model_dir / "temperature.json").read_text(encoding="utf-8"))["temperature"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    out = []
    for i in range(0, len(cases), 64):
        chunk = cases[i:i + 64]
        tokens = tokenizer([r["text"] for r in chunk], padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**tokens).logits
        confidence, idx = scaled_softmax(logits, temperature).max(dim=-1)
        for row, p, k in zip(chunk, confidence.tolist(), idx.tolist()):
            out.append({"id": row["id"], "expected": row["label"],
                        "output": id2label[int(k)], "confidence": round(float(p), 6)})
    return out


def decision(rows: list[dict]) -> None:
    import gonogo as g
    d = g.decide([(r["confidence"], r["expected"] == r["output"]) for r in rows],
                 target=0.95, level=0.95, unit="cases")
    print(f"{sum(r['expected'] == r['output'] for r in rows)}/{len(rows)} "
          f"({d.pass_rate}); {d.verdict.value}")
    if d.operating_point:
        op = d.operating_point
        print(f"threshold={op.threshold}, coverage={op.coverage}, "
              f"deferred={op.n_deferred}, precision={op.precision}")
    print(f"calibration_error={d.calibration_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="optional local, unpublished model artifact")
    parser.add_argument("--cases-path", type=Path, help="250 held-out Banking77 cases JSONL")
    parser.add_argument("--write", action="store_true", help="replace the receipt from real CPU inference")
    args = parser.parse_args()
    if bool(args.model_dir) != bool(args.cases_path):
        parser.error("--model-dir and --cases-path must be supplied together")
    if args.write and not args.model_dir:
        parser.error("--write requires a model and cases")
    if args.model_dir:
        verify_local_sources(args.model_dir, args.cases_path)
        actual = predict(args.model_dir, args.cases_path)
        if args.write:
            content = "\n".join(json.dumps(r, ensure_ascii=False) for r in actual) + "\n"
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest != _manifest()["predictions_jsonl_sha256"]:
                raise SystemExit("refusing to replace receipt with bytes outside its checked provenance")
            RECEIPT.write_text(content, encoding="utf-8")
        elif actual != read_predictions(RECEIPT):
            raise SystemExit("local model predictions differ from checked-in receipt")
        else:
            print("Local CPU predictions match receipt case-for-case")
        decision(actual)
    else:
        decision(read_predictions(RECEIPT))


if __name__ == "__main__":
    main()
