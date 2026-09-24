"""Pretrain: baseline → invoke training → re-baseline → compare.

The pretrain path trains on a corpus (the nanogpt loop on Modal), not on
the thomas cases. The cases are the *eval set* — the baseline measures the
model before training, the after-baseline measures it after. The training
itself runs on a GPU via Modal (or whatever compute backend is configured).

This is different from post-train, where the cases ARE the training data
(Case → Tinker Env → LoRA RL). Pretrain trains on a corpus; the cases
just measure whether the register was installed.

The pretrain path installs the *register* — the distribution the model
speaks in. Post-train RL then amplifies the *deliberation* — when to ask,
when to tell, when to withhold. These are different objectives and need
different machinery (RL_STRATEGY.md §1).

The compute backend is pluggable. The default is Modal (the nanogpt
repo's ``modal_app.py::train``). To use a different backend, pass a
``backend`` string or replace this module's ``_invoke`` function.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .task import Task
from .baseline import BaselineResult, baseline


@dataclass
class PretrainResult:
    """The output of a pretrain run."""

    task_name: str
    model: str
    before: BaselineResult
    after: BaselineResult | None = None
    config: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    backend: str = "modal"


def pretrain(
    task: Task,
    model: str | None = None,
    *,
    target: float = 0.95,
    backend: str = "modal",
    modal_app_path: str | None = None,
    train_steps: int = 3250,
    max_tokens: int = 256,
    **train_kwargs: Any,
) -> PretrainResult:
    """Run the pretrain path: baseline → train on corpus → re-baseline.

    The training runs on a GPU via the configured backend (default: Modal).
    The cases are the eval set, not the training data — the corpus is
    already sharded on the backend's volume.

    **Ask before spending credits.** GPU time costs money.
    """
    if modal_app_path is None:
        modal_app_path = str(Path.home() / "git/modded-nanogpt/modal_app.py")

    # 1. Baseline
    print("--- Baseline (before pretraining) ---")
    before = baseline(task, model=model, max_tokens=max_tokens, target=target)

    # 2. Train
    print(f"\n--- Pretrain ({backend}) ---")
    print(f"  Modal app:   {modal_app_path}")
    print(f"  Train steps: {train_steps}")
    print(f"  Model:        {before.model}")
    print(f"  Cases (eval): {task.n}")

    if backend == "modal":
        run_id = _invoke_modal(modal_app_path, train_steps, **train_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Supported: 'modal'.")

    print(f"\n  Training complete. Run ID: {run_id}")

    # 3. After: re-sample the trained model
    # The trained model is a new checkpoint on the runs volume.
    # For the after-baseline, the checkpoint needs to be served (Modal
    # endpoint, local GGUF, etc.) and sampled against the same cases.
    # That's a separate integration — the thomas baseline samples from
    # a Tinker sampling client, and the nanogpt checkpoint isn't on
    # Tinker. When the serving path is wired, this becomes:
    #   after = baseline(task, model=served_model_name, ...)
    print("\n--- Baseline (after pretraining) ---")
    print("  (After-baseline requires serving the trained checkpoint.)")
    print("  The training run is on Modal; fetch the checkpoint and serve it")
    print("  to run the after-baseline. See modded-nanogpt/MODAL.md.")

    return PretrainResult(
        task_name=task.name,
        model=before.model,
        before=before,
        config={
            "backend": backend,
            "modal_app_path": modal_app_path,
            "train_steps": train_steps,
            **train_kwargs,
        },
        run_id=run_id,
    )


def _invoke_modal(
    modal_app_path: str,
    train_steps: int,
    **kwargs: Any,
) -> str:
    """Invoke the nanogpt training loop on Modal.

    Runs ``modal run modal_app.py::train --train-steps N`` and returns
    the run ID from Modal's output.

    For long runs, use ``--detach`` (pass ``detach=True`` in kwargs) so
    the run survives a local disconnect.
    """
    cmd = [
        "modal", "run", f"{modal_app_path}::train",
        "--train-steps", str(train_steps),
    ]
    for k, v in kwargs.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])

    print(f"  Running: {' '.join(cmd)}")
    print("  (Use --detach for long runs.)")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  Modal exited with code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
        for line in result.stdout.splitlines():
            if "run_" in line and "modal" in line.lower():
                return line.strip()
        return "modal-run-unknown"
    except FileNotFoundError:
        print("  WARNING: modal CLI not found. Is Modal installed?")
        return "modal-not-found"
    except subprocess.TimeoutExpired:
        print("  WARNING: Modal run timed out. Use --detach for long runs.")
        return "modal-timeout"
