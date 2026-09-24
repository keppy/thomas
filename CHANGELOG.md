# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
thomas is pre-1.0: minor versions can break the Python API. The artifact and
case contract is versioned separately in `docs/CONTRACT.md`; any change to it
is called out here with the contract version.

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
