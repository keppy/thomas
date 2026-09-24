# The thomas ↔ gonogo contract

Contract version: **1** (thomas 0.2.0, gonogo-eval `>=0.2,<0.3`)

thomas trains models; gonogo decides whether they can ship. They share no code
at runtime beyond thomas calling gonogo's public API. What they do share is
this page: the case shape, the reward signature, the confidence definition,
and the artifact a training run leaves on disk. Anything that reads a thomas
artifact (gonogo's example scripts, the Hermes plugin, a serving process)
reads it against this page.

## Versioning

- `thomas.contract.CONTRACT_VERSION` is the number above, as an int.
- Encoder runs write it into `metrics.json` as `contract_version`. A reader
  treats a missing field as `1`, and refuses a version newer than it knows.
- **Adding** an optional field or file keeps the version.
- **Removing, renaming, or changing the meaning** of anything on this page
  bumps it, and gets a CHANGELOG entry saying what readers must change.
- The gonogo range is pinned in `pyproject.toml` and restated here. Widening it
  to a new gonogo minor goes through the test suite first.

`thomas.contract.read_artifact(dir)` checks an artifact against this page with
the stdlib only, so readers do not need torch or modal installed.

## 1. Cases

thomas does not own a case type. A domain's cases can be any shape, as long as
the domain supplies what `thomas.Task` asks for:

| Field | Signature | Meaning |
|---|---|---|
| `cases` | `list[Any]` | the case set, non-empty |
| `case_id` | `(case) -> str` | a stable id; gonogo pairs runs by it |
| `render_messages` | `(case) -> list[{"role", "content"}]` | what the model sees |
| `score_text` | `(case, text) -> (reward: float, detail)` | the one reward function |
| `oracle_reply` | `(case) -> str` | a reply that scores 1.0; proves the env is winnable |

The default `case_id` reads `.id` or `["id"]`; the default `render_messages`
reads `.prompt`, `["messages"]`, or `["input"]["messages"]`.

For encoder training the case is a JSONL row: `{"id"?, "text", "label"}`,
with `label` a non-empty string.

## 2. Reward → gonogo

`score_text` is the only scoring path. The baseline card, the RL loop and the
before/after comparison all call it, so they see identical numbers.

When thomas hands results to gonogo, a case **passes** when
`reward >= 0.99`, and the reward travels as gonogo's `score`. The `detail` is
stringified into gonogo's `detail`. Pairing across runs uses `case_id`.

## 3. Confidence

One definition, everywhere (training eval, saved metrics, downstream
gonogo agents):

```
confidence = max(softmax(logits / T))
```

- `T` is applied to the logits **before** the softmax.
- The softmax runs over the **full** label set, never a filtered subset.
- No top-1-minus-runner-up margin, no sigmoid, no renormalization.

The implementation is `thomas.encoder_train.scaled_softmax`. Readers that can
import thomas should call it rather than reimplement it. gonogo reads the
result from `Prediction.confidence`; that is what feeds its calibration error
and its operating point.

## 4. Encoder artifact

A directory holding HuggingFace `save_pretrained` output (model + tokenizer,
including `config.json`), plus:

| File | Shape | Notes |
|---|---|---|
| `label2id.json` | `{label: int}` | ids are `0..n-1`, no gaps |
| `temperature.json` | `{"temperature": float}` | `> 0`, fit by NLL on the calib split |
| `metrics.json` | object | below |

`metrics.json` fields: `contract_version`, `temperature`, `num_labels`,
`train_size`, `calib_size`, `calib_accuracy`, `ece_before`, `ece_after`,
`final_train_loss`, `mean_train_loss`, `step_losses`.

`calib_accuracy` and the ECE numbers are measured on the calibration split,
which is the split `T` was fit on. **They are not an eval result.** An eval
runs gonogo on cases the run never saw.

## 5. Splits

- The calibration split is `calib_size` rows drawn from the training rows with
  `random.Random(seed)`, removed before training. Never trained on.
- The eval set is the caller's job and must not overlap the training rows.
  thomas does not see it.
- The Banking77 canary (`examples/banking77_encoder.py`) copies its split
  logic from `gonogo/examples/banking77_routing.py`: test pilot
  `random.Random(11).sample(test, 250)`, ids `msg-%04d`, calib seed 7, 500
  rows. The gonogo copy is the source of truth; both sides must match
  case-for-case so reports pair.

## 6. gonogo surface thomas depends on

`gonogo.Case`, `gonogo.Prediction`, `gonogo.evaluate`, `gonogo.compare`, and
`Report.to_dict()` (per-case `id`, `passed`, `score`, `confidence`). These are
gonogo's public API as of 0.2; a gonogo change to any of them is a contract
bump on this side.
