"""Pretrain → Post-train handoff: Modal GPU → vLLM serve → LoRA RL.

The full pipeline on Modal, no Tinker required:

1. Modal pretrain (nanogpt loop, full-parameter training)
   → HuggingFace checkpoint on the runs volume
2. vLLM serve (Modal) → OpenAI-compatible endpoint
   → the pretrained model is now servable
3. Post-train (TRL GRPO LoRA RL, Modal)
   → samples from the vLLM endpoint, trains a LoRA adapter
   → the register (pretrain) + the disposition (post-train) in one model
4. vLLM serve the trained LoRA → re-baseline → compare

This is the single-run path: pretrain and post-train on the same GPU,
the checkpoint never leaves Modal, and the baseline samples from the
served endpoint.

The Tinker path is still available (``backend="tinker"``) for cases
where the base model is Tinker-supported and token-priced RL is
preferable to GPU rental. But the Modal path is the one that works
end-to-end with a custom pretrained checkpoint.

Architecture:

    Modal pretrain → HF checkpoint → vLLM serve → GRPO LoRA RL → vLLM serve
         │                                              │              │
         └─ baseline (before)                           │              │
                                      └─ baseline (after pretrain)     │
                                                       └─ baseline (after post-train)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .task import Task
from .baseline import BaselineResult, baseline, baseline_from_outputs


@dataclass
class HandoffResult:
    """The output of a pretrain → post-train handoff."""

    task_name: str
    model: str
    pretrain_before: BaselineResult
    pretrain_after: BaselineResult | None = None
    post_train_before: BaselineResult | None = None
    post_train_after: BaselineResult | None = None
    config: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    vllm_endpoint: str | None = None
    backend: str = "modal"


def handoff(
    pretrain_task: Task,
    post_train_task: Task,
    *,
    model: str | None = None,
    backend: str = "modal",
    modal_app_path: str | None = None,
    train_steps: int = 3250,
    run_post_train: bool = False,
    lora_rank: int = 32,
    group_size: int = 4,
    max_steps: int = 5,
    learning_rate: float = 1e-5,
    target: float = 0.95,
    **kwargs: Any,
) -> HandoffResult:
    """Run the full pretrain → serve → post-train pipeline.

    Backend:
      - ``"modal"`` (default): Modal pretrain → vLLM serve → TRL GRPO LoRA RL.
        The full pipeline runs on Modal. No Tinker dependency. This is
        the path that works with a custom pretrained checkpoint.
      - ``"tinker"``: Modal pretrain → Tinker LoRA RL. Requires a
        Tinker-supported base model; the pretrained weights can't be
        loaded into Tinker unless they're in Tinker's curated list.

    **Ask before spending credits.** Both Modal GPU time and (if Tinker)
    Tinker training cost money.
    """
    if modal_app_path is None:
        modal_app_path = str(Path.home() / "git/modded-nanogpt/modal_app.py")

    # 1. Pretrain baseline
    print("=== Stage 1: Pretrain Baseline ===")
    print(f"  Task: {pretrain_task.name}")
    pretrain_before = baseline(pretrain_task, model=model, max_tokens=256, target=target)

    # 2. Modal pretrain
    print(f"\n=== Stage 2: Modal Pretrain ({train_steps} steps) ===")
    print(f"  Modal app:   {modal_app_path}")
    print(f"  Model:        {pretrain_before.model}")
    print(f"  Cases (eval): {pretrain_task.n}")

    run_id = _invoke_modal(modal_app_path, train_steps)
    print(f"\n  Training complete. Run ID: {run_id}")

    # 3. Serve the pretrained model (vLLM on Modal)
    print(f"\n=== Stage 3: Serve (vLLM on Modal) ===")
    vllm_endpoint = _serve_vllm(modal_app_path, run_id)
    print(f"  vLLM endpoint: {vllm_endpoint}")

    # 4. Post-train (optional)
    post_train_before = None
    post_train_after = None
    if run_post_train:
        # Baseline after pretrain, before RL
        print(f"\n=== Stage 4: Post-Train Baseline (after pretrain) ===")
        post_train_before = _baseline_from_endpoint(
            post_train_task, vllm_endpoint, target=target,
            model_name=kwargs.get("vllm_model_name", "pretrained"),
        )

        # RL
        print(f"\n=== Stage 5: Post-Train (backend={backend}) ===")
        if backend == "modal":
            from .trl_grpo import run_grpo_modal
            run_grpo_modal(
                post_train_task, vllm_endpoint,
                model_path=kwargs.get("model_path", pretrain_before.model),
                lora_rank=lora_rank, group_size=group_size,
                max_steps=max_steps, learning_rate=learning_rate,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 1.0),
            )
        elif backend == "tinker":
            from .post_train import post_train as _pt
            _pt(
                post_train_task,
                model=pretrain_before.model,
                lora_rank=lora_rank, group_size=group_size,
                max_steps=max_steps, learning_rate=learning_rate,
                target=target,
            )

    return HandoffResult(
        task_name=pretrain_task.name,
        model=pretrain_before.model,
        pretrain_before=pretrain_before,
        post_train_before=post_train_before,
        post_train_after=post_train_after,
        config={
            "backend": backend,
            "modal_app_path": modal_app_path,
            "train_steps": train_steps,
            "vllm_endpoint": vllm_endpoint,
            "run_post_train": run_post_train,
        },
        vllm_endpoint=vllm_endpoint,
    )


def _invoke_modal(modal_app_path: str, train_steps: int) -> str:
    """Invoke the nanogpt training loop on Modal."""
    cmd = [
        "modal", "run", f"{modal_app_path}::train",
        "--train-steps", str(train_steps),
    ]
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  Modal exited {result.returncode}: {result.stderr[:500]}")
        for line in result.stdout.splitlines():
            if "run_" in line:
                return line.strip()
        return "modal-run-unknown"
    except FileNotFoundError:
        print("  WARNING: modal CLI not found.")
        return "modal-not-found"
    except subprocess.TimeoutExpired:
        print("  WARNING: timed out. Use --detach.")
        return "modal-timeout"


def _serve_vllm(modal_app_path: str, run_id: str, checkpoint_path: str | None = None) -> str:
    """Serve the pretrained checkpoint via vLLM on Modal.

    Deploys a vLLM server on Modal that loads the checkpoint from the
    runs volume and exposes an OpenAI-compatible endpoint. The endpoint
    is persistent (deployed, not ``modal run``).

    Args:
        modal_app_path: path to the nanogpt modal_app.py (unused — the
            serve app is in thomas)
        run_id: the Modal run ID (for logging)
        checkpoint_path: optional explicit checkpoint path on the runs
            volume. If None, uses the default (latest/final).

    Returns:
        The OpenAI-compatible endpoint URL.
    """
    from .serve_vllm import serve
    return serve(checkpoint_path)


def _baseline_from_endpoint(
    task: Task, endpoint: str, *, target: float = 0.95, model_name: str = "pretrained"
) -> BaselineResult:
    """Sample from a vLLM OpenAI-compatible endpoint and score.

    Uses the OpenAI client to sample, then scores with the task's
    ``score_text``. Same scoring path as thomas.baseline.

    Args:
        task: the thomas Task
        endpoint: vLLM endpoint URL (e.g. ``https://...--serve.modal.run/v1``)
        model_name: the model name as served by vLLM (check ``/v1/models``)
    """
    import os
    from openai import OpenAI

    api_key = os.environ.get("MODAL_API_KEY", "no-key-needed")
    client = OpenAI(base_url=endpoint, api_key=api_key)

    outputs: dict[str, str] = {}
    for case in task.cases:
        cid = task.case_id(case)
        messages = [{"role": "system", "content": task.system_prompt}]
        messages += task.render_messages(case)
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        outputs[cid] = content
        reward, detail = task.score_text(case, content)
        status = "✓" if reward >= 0.99 else " "
        print(f"  {status} {cid:30s} r={reward:.3f}")

    return baseline_from_outputs(task, outputs, target=target)


def _run_trl_grpo(
    task: Task,
    vllm_endpoint: str,
    *,
    model_path: str | None = None,
    lora_rank: int = 32,
    group_size: int = 4,
    max_steps: int = 5,
    learning_rate: float = 1e-5,
    max_tokens: int = 512,
    temperature: float = 1.0,
    output_dir: str = "./logs/thomas-grpo",
) -> Any:
    """Run TRL GRPO LoRA RL against a vLLM endpoint.

    Uses TRL's GRPOTrainer with vLLM in server mode. The vLLM server
    generates rollouts; TRL trains a LoRA adapter on the policy gradient.
    The reward function is ``task.score_text`` — the same function the
    baseline and the Tinker env use.

    Requires TRL, transformers, peft, and datasets installed. Runs on
    GPU (Modal or local).
    """
    from .trl_grpo import run_grpo
    return run_grpo(
        task,
        vllm_endpoint,
        model_path=model_path,
        lora_rank=lora_rank,
        group_size=group_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        temperature=temperature,
        output_dir=output_dir,
    )
