# thomas — AGENTS.md

## What this repo is

**thomas** is an open-source training harness. Take a Case set and a
`score_text` function, get a baseline card (via gonogo), train (encoder SFT on
Modal, or LoRA RL via Tinker or TRL GRPO on Modal), and compare before/after.
The domain supplies the Case shape and the reward; thomas supplies the plumbing.

    Case set ──► thomas.baseline        (sample base model, score, gonogo card)
             ──► thomas.encoder_train   ((text, label) → calibrated classifier, Modal L4)
             ──► thomas.post_train      (Case → TinkerEnv → LoRA RL → checkpoint)
             ──► thomas.trl_grpo        (Case → GRPO + vLLM colocate → LoRA, Modal)

A pretrain path (Case → shards → nanogpt loop) is a stub.

**The contract with gonogo is `docs/CONTRACT.md`** (case shape, reward
signature, confidence definition, artifact layout), versioned by
`thomas.contract.CONTRACT_VERSION`. Read it before touching any of those.

## What thomas is not

- Not gonogo (that's the scoring/decision layer; thomas imports it)
- Not a model (that's Tinker or nanogpt; thomas drives them)
- Not a case format (that's the domain)
- Not an eval framework (that's gonogo + the domain's checks)

## Rules

1. **No domain-specific code.** thomas knows nothing about math, literary
   registers, or tutoring traces. The domain plugs in its Case type and `score_text`.
2. **One reward function.** `score_text(case, text) → (reward, detail)` is
   the single scoring path — used by both the baseline and the RL loop, so
   they see identical numbers.
3. **No hardcoded paths.** This is an open-source package. No `~/git/...`,
   no `.env` paths baked in. Load keys from env or `./.env`.
4. **gonogo is the only required dependency.** Tinker is optional
   (`pip install thomas-train[post_train]`).
5. **Ask before spending credits.** `post_train`, `encoder_train` and every
   Modal entry point cost money. Report the config and stop for approval before running.
6. **Modal GRPO colocate pitfalls** (each cost a run — see README
   "Modal GRPO" for details): `vllm_gpu_memory_utilization` must cover
   trainer + vLLM weights + KV (it subtracts in-process memory); use
   `vllm_max_model_length` (TRL 1.13 kwarg, not `max_model_len`); load the
   model with `torch_dtype="auto"`; reward fns get chat-message lists, not
   strings; VLM-class models (`*ForConditionalGeneration`, e.g. Qwen3.5)
   need the shipped trl#5269 prefix remap — prefer text-only checkpoints.
7. **Modal `.remote()` needs a running app**: call it inside `with <app>.run():`
   or it raises `ExecutionError: Function has not been hydrated`. Also, a
   read-function container started immediately after a training container may
   see a stale volume snapshot (artifact "not found" though `modal volume ls`
   shows it) — retry the pull before re-diagnosing.
8. **Contract changes are versioned.** Adding an optional field to the
   artifact or case shape keeps `CONTRACT_VERSION`; removing, renaming or
   re-meaning anything bumps it, updates `docs/CONTRACT.md`, and gets a
   CHANGELOG entry. `tests/test_contract.py` pins the doc to the constant.
9. **Keep it generic.** No employer, product or internal-repo names, no
   workspace-specific Modal URLs. Endpoints and workspaces come from args/env.

## Architecture

```
Case ─► ThomasEnv ─► initial_observation() ─► sample ─► step() ─► reward
                     ↑                                          │
                     └── task.score_text() ──────────────────┘
```

One Case → one Env, `step()` decodes tokens → text → `score_text` → reward.
Single step, `episode_done=True`.

## Tests

```bash
pytest tests/ -v
```

Tests run without Tinker, Modal or a GPU: the Task abstraction, the
`score_text` contract, oracle checks, the artifact contract, and a CPU encoder
fine-tune on a tiny random BERT. On Windows add `--basetemp` if pytest's
temp dir throws PermissionError.
