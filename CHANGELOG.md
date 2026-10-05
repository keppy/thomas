# Changelog

## [0.2.3] - 2026-10-05

evalroute lane-encoder recipe learns from train F's learning curve.

- `examples/evalroute_lane_encoder.py`: `select_training_rows` — per-lane cap on `aug:` rows
  (`--aug-cap`, default max(5, the lane's real-row count); lanes with zero real rows keep all
  scaffolding), real rows repeated `--real-weight` times (default 3); `--seeds 7,8,9` trains
  per-seed dirs, writes `seeds.json` (mean ± sd), copies the best-calibrated seed (lowest |T−1|)
  to the artifact root; `--curve` trains on 0/50/100% of real rows and writes
  `learning_curve.json`; `--dry-run` prints the selection. `metrics.json` gains `selection`,
  `seeds`, `chosen_seed`, `eval_accuracy_mean/sd`. On the v1 data, three seeds: 71.9% ± 6.6
  (v1 single seed 63.2%) with no new rows.
- `encoder_train`: when rows repeat a text (weighting by repetition), the calibration split is
  drawn over distinct texts and every copy stays out of training; refuses `calib_size` ≥
  distinct texts. An index draw let copies of held-out rows into training and T collapsed to
  0.003. All-distinct inputs (the contract's banking77 split) are unchanged.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
thomas is pre-1.0: minor versions can break the Python API. The artifact and
case contract is versioned separately in `docs/CONTRACT.md`; any change to it
is called out here with the contract version.

## [0.2.2] - 2026-10-04

### Added

- `examples/evalroute_lane_encoder.py`: recipe for evalroute's lane encoder over
  `encoder_train.train_classifier`. `--backend local` (default; CPU is enough at a few
  hundred rows — v1 trained in 30 s) or `modal`; evaluates on a held-out real-only
  JSONL after training and adds `eval_accuracy`, `eval_n`, `eval_per_lane`,
  `defer_below` (max-coverage confidence cut at ≥ 0.9 held-out accuracy),
  `coverage_at_defer`, `train_sources` to `metrics.json`; writes
  `eval_predictions.jsonl`. Prints counts and ids, never row text. Contract
  version unchanged (additive keys).

## [0.2.1] - 2026-09-28

Contract version **1** remains unchanged: no artifact or case field changed.
Requires `gonogo-eval>=0.3,<0.4` (the `compare()` interval and grouped
pairing changed there; the 0.2 line is not compatible).

- `post_train` now requires an untouched `eval_task` for before/after results
  (a source-breaking change for callers without an eval split), rejects
  obvious split overlap, verifies its dotted `score_text_fn` is the exact
  Task scorer, and checks all serialized Tinker case fields, ids, prompts,
  and several reward probes after deserialization. These checks do not prove
  arbitrary scorer equivalence. Every attempt gets a unique log path;
  missing current-run checkpoints fail instead of inheriting an old sampler.
- TRL GRPO rewards use stable case IDs instead of prompt matching, with one
  visible-text extraction path for local and Modal completions. Programmatic
  `run_grpo_modal()` hydrates its app before `.remote()` and rejects GPU
  overrides because its Modal function is fixed to A100-40GB.
- Encoder configs reject invalid hyperparameters before training. A text-free
  Banking77 per-case prediction receipt, source/artifact digest manifest, and
  CPU replay are tracked; the trained model remains local-only and unpublished.

## [0.2.0] - 2026-09-23

Contract version **1**.

### Added

- `docs/CONTRACT.md`: the case shape, reward signature, confidence definition,
  encoder artifact layout and split rules that thomas and gonogo share.
- `thomas.contract`: `CONTRACT_VERSION`, `read_artifact()` (stdlib-only artifact
  check). Encoder runs now write `contract_version` into `metrics.json`;
  artifacts without it read as version 1.
- `thomas.encoder_train`: supervised encoder fine-tune on Modal with temperature
  scaling (the path behind the Banking77 canary, 87.2% [82.5%, 90.8%]).
- `thomas.trl_grpo`: TRL GRPO LoRA RL on Modal, vLLM colocate.

### Changed

- The tutoring example domain is now `thomas.domains.tutor` /
  `thomas.domains.tutor_scenarios`, and `--domain tutor` on the Modal CLIs.
- gonogo requirement narrowed to `>=0.2,<0.3`, the range the contract is
  tested against.
- `trl_grpo --vllm-endpoint` has no default (colocate mode does not use it);
  `serve_vllm` builds its URL from `MODAL_WORKSPACE`.

## [0.1.0] - 2026-09-18

First version: `Task`, `baseline`, `baseline_from_outputs`, `compare`, and the
Tinker `post_train` loop.
