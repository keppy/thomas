"""TRL GRPO LoRA RL — post-train via TRL + vLLM on Modal.

Runs Group Relative Policy Optimization (GRPO) with a LoRA adapter against
a vLLM-served model. The vLLM server generates rollouts; TRL trains the
LoRA adapter on the policy gradient. This is the open-source post-train
path that doesn't require Tinker.

Two modes:
  1. **Local** (GPU machine): ``run_grpo(task, vllm_endpoint, model_path=...)``
  2. **Modal** (remote GPU): ``run_grpo_modal(task, vllm_endpoint, model_path=...)``
     — runs the training on Modal, saves the adapter to a Modal volume.

Usage::

    # Local (GPU machine):
    from thomas.trl_grpo import run_grpo
    result = run_grpo(task, vllm_endpoint="https://...", model_path="Qwen/Qwen3.5-4B")

    # Modal (remote GPU, no local GPU needed):
    from thomas.trl_grpo import run_grpo_modal
    result = run_grpo_modal(task, vllm_endpoint="https://...", model_path="Qwen/Qwen3.5-4B")

The reward function is ``task.score_text`` — the same function the
baseline and the Tinker env use. One reward function, three consumers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .task import Task


@dataclass
class GRPOResult:
    """The output of a TRL GRPO run."""

    task_name: str
    model_path: str
    vllm_endpoint: str
    adapter_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


def _build_dataset(task: Task):
    """Build a TRL-compatible dataset from a Task's cases."""
    from datasets import Dataset

    rows = []
    for case in task.cases:
        messages = [{"role": "system", "content": task.system_prompt}]
        messages += task.render_messages(case)
        rows.append({"prompt": messages})
    return Dataset.from_list(rows)


def _make_reward_fn(task: Task):
    """Create a reward function that calls task.score_text on completions."""

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            user_msg = next(
                (m["content"] for m in prompt if m["role"] == "user"), ""
            )
            case = next(
                (c for c in task.cases
                 if any(m.get("content") == user_msg
                        for m in task.render_messages(c)
                        if m.get("role") == "user")),
                None,
            )
            if case is None:
                rewards.append(0.0)
            else:
                reward, _ = task.score_text(case, completion)
                rewards.append(float(reward))
        return rewards

    return reward_fn


def run_grpo(
    task: Task,
    vllm_endpoint: str,
    *,
    model_path: str = "Qwen/Qwen3.5-4B",
    lora_rank: int = 32,
    group_size: int = 4,
    max_steps: int = 5,
    learning_rate: float = 1e-5,
    max_tokens: int = 512,
    temperature: float = 1.0,
    output_dir: str = "./logs/thomas-grpo",
) -> GRPOResult:
    """Run TRL GRPO LoRA RL locally (requires GPU + TRL + transformers + peft).

    The vLLM server generates rollouts; TRL trains a LoRA adapter on the
    policy gradient. The reward function is ``task.score_text``.

    **Ask before spending credits.** This runs on GPU.
    """
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    dataset = _build_dataset(task)
    reward_fn = _make_reward_fn(task)

    config = GRPOConfig(
        output_dir=output_dir,
        num_generations=group_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        max_completion_length=max_tokens,
        temperature=temperature,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=group_size,
        save_steps=max_steps,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=vllm_endpoint,
    )

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"  Base model: {model_path}")
    print(f"  vLLM endpoint: {vllm_endpoint}")
    print(f"  LoRA rank: {lora_rank}, group_size: {group_size}, steps: {max_steps}")
    print(f"  Cases: {task.n}")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print(f"\n  Training {max_steps} steps...")
    trainer.train()
    print(f"\n  Saving adapter to {output_dir}")
    trainer.save_model(output_dir)

    log_history = trainer.state.log_history if hasattr(trainer.state, "log_history") else {}

    return GRPOResult(
        task_name=task.name,
        model_path=model_path,
        vllm_endpoint=vllm_endpoint,
        adapter_path=output_dir,
        metrics=log_history,
        config={
            "lora_rank": lora_rank,
            "group_size": group_size,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )


# --- Modal mode -------------------------------------------------------------
#
# Modal requires @app.function at global scope, so we define a module-level
# Modal app + function, and pass parameters via a serialized config dict.
# The function runs on a remote GPU, loads the model from HuggingFace,
# builds the dataset + reward from the serialized cases, and trains.

import modal

_grpo_app = modal.App("thomas-grpo")

_grpo_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "torch>=2.5.0",
        "transformers>=4.46.0",
        "peft>=0.12.0",
        "trl>=0.12.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "vllm>=0.19.1,<0.29.0",
        "wandb>=0.18.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

_grpo_vol = modal.Volume.from_name("k3mini-runs", create_if_missing=True)
_grpo_secret = modal.Secret.from_name("wandb")


@_grpo_app.function(
    image=_grpo_image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={"/root/thomas-grpo": _grpo_vol},
    secrets=[_grpo_secret],
)
def _grpo_train(
    cases_json: str,
    score_fn_path: str,
    render_fn_path: str,
    case_id_fn_path: str,
    system_prompt: str,
    task_name: str,
    model_path: str,
    vllm_endpoint: str,
    lora_rank: int,
    group_size: int,
    max_steps: int,
    learning_rate: float,
    max_tokens: int,
    temperature: float,
    output_path: str,
    wandb_project: str = "thomas",
    wandb_name: str | None = None,
) -> dict:
    """Run GRPO training on Modal. Called via .remote()."""
    import importlib
    import json as _json
    from types import SimpleNamespace

    # Deserialize cases
    raw_cases = _json.loads(cases_json)
    cases = [SimpleNamespace(**d) if isinstance(d, dict) else d for d in raw_cases]

    # Import score/render/case_id functions
    def _import(dotted):
        mod, _, name = dotted.rpartition(".")
        return getattr(importlib.import_module(mod), name)

    score_fn = _import(score_fn_path)
    render_fn = _import(render_fn_path) if render_fn_path else None
    case_id_fn = _import(case_id_fn_path) if case_id_fn_path else None

    # Build dataset
    from datasets import Dataset

    def _render(case):
        if render_fn:
            return render_fn(case)
        return [{"role": "user", "content": getattr(case, "prompt", str(case))}]

    rows = [{"prompt": [{"role": "system", "content": system_prompt}] + _render(c)}
            for c in cases]
    dataset = Dataset.from_list(rows)

    # Reward function
    def _completion_text(completion):
        # TRL passes conversational completions as a list of messages;
        # plain completions as a string. Score on the assistant text.
        if isinstance(completion, str):
            text = completion
        else:
            text = "\n".join(
                m.get("content", "") for m in completion
                if isinstance(m, dict) and m.get("role") == "assistant"
            )
        # Judge the visible reply, not hidden reasoning: a thinking model's
        # <think> block can contain the answer without the student seeing it.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            user_msg = next(
                (m["content"] for m in prompt if m["role"] == "user"), ""
            )
            case = next(
                (c for c in cases
                 if any(m.get("content") == user_msg
                        for m in _render(c) if m.get("role") == "user")),
                None,
            )
            if case is None:
                rewards.append(0.0)
            else:
                r, _ = score_fn(case, _completion_text(completion))
                rewards.append(float(r))
        return rewards

    # GRPO config
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # W&B config via env vars (GRPOConfig/TrainingArguments uses these)
    import os
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_name:
        os.environ["WANDB_RUN_NAME"] = wandb_name

    config = GRPOConfig(
        output_dir=output_path,
        num_generations=group_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        max_completion_length=max_tokens,
        temperature=temperature,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=group_size,
        save_steps=max_steps,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_max_model_length=4096,
        # Colocate shares one GPU between the trainer and the vLLM engine.
        # vLLM's budget = total_gpu * utilization MINUS what the process
        # already holds (the training model!), so utilization must cover
        # trainer weights + vLLM weights + KV cache. 0.25 left KV = 0 ->
        # "No available memory for the cache blocks".
        vllm_gpu_memory_utilization=0.75,
        report_to="wandb",
    )

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"  Base model: {model_path}")
    print(f"  vLLM endpoint: {vllm_endpoint}")
    print(f"  LoRA rank: {lora_rank}, group_size: {group_size}, steps: {max_steps}")
    print(f"  Cases: {len(cases)}")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # --- Qwen3.5/VLM weight-name remap (huggingface/trl#5269) -------------
    # vLLM serves Qwen3.5 through the ForConditionalGeneration wrapper, so
    # HF param prefixes (model.language_model.*, model.visual.*, lm_head.*)
    # must be rewritten to vLLM's names before colocate weight sync.
    import trl.generation.vllm_generation as _vg

    _VLLM_PREFIX_MAPS = {
        "Qwen3_5ForConditionalGeneration": {
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        },
        "Qwen3VLForConditionalGeneration": {
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        },
        "Qwen3_5ForCausalLM": {
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    }
    _arch_map = {}
    _archs = list(getattr(model.config, "architectures", None) or []) + [type(model).__name__]
    print(f"  Model class: {type(model).__name__}, architectures: {getattr(model.config, 'architectures', None)}")
    for _arch in _archs:
        if _arch in _VLLM_PREFIX_MAPS:
            _arch_map = _VLLM_PREFIX_MAPS[_arch]
            break
    if _arch_map:
        _orig_fix = _vg.VLLMGeneration._fix_param_name_to_vllm

        def _fix_param_name_to_vllm(self, name, extra_prefixes=None):
            name = _orig_fix(self, name, extra_prefixes)
            for _p, _np in _arch_map.items():
                if name.startswith(_p):
                    return name.replace(_p, _np, 1)
            return name

        _vg.VLLMGeneration._fix_param_name_to_vllm = _fix_param_name_to_vllm
        print(f"  Patched vLLM weight-name prefixes for {model.config.architectures}")

    print(f"\n  Training {max_steps} steps...")
    trainer.train()
    print(f"\n  Saving adapter to {output_path}")
    trainer.save_model(output_path)

    _grpo_vol.commit()

    log_history = trainer.state.log_history if hasattr(trainer.state, "log_history") else {}
    return {
        "adapter_path": output_path,
        "metrics": log_history,
    }


def run_grpo_modal(
    task: Task,
    vllm_endpoint: str,
    *,
    model_path: str = "Qwen/Qwen3.5-4B",
    lora_rank: int = 32,
    group_size: int = 4,
    max_steps: int = 5,
    learning_rate: float = 1e-5,
    max_tokens: int = 512,
    temperature: float = 1.0,
    output_path: str = "/root/thomas-grpo",
    gpu: str = "A100-40GB",
    wandb_project: str = "thomas",
    wandb_name: str | None = None,
) -> GRPOResult:
    """Run TRL GRPO LoRA RL on Modal (remote GPU, no local GPU needed).

    The task's cases are serialized as JSON and sent to Modal. The vLLM
    endpoint must be accessible from Modal (use a deployed endpoint URL).
    The LoRA adapter is saved to the k3mini-runs volume.

    The task must have ``_score_fn_path``, ``_render_fn_path``, and
    ``_case_id_fn_path`` set to dotted import paths so Modal can resolve
    them at runtime.

    **Ask before spending credits.** This runs on Modal GPU.
    """
    score_fn_path = getattr(task, "_score_fn_path", None)
    if score_fn_path is None:
        raise ValueError(
            "task._score_fn_path must be set to a dotted path (e.g. "
            "'thomas.domains.tutor.score') for Modal mode."
        )

    render_fn_path = getattr(task, "_render_fn_path", "")
    case_id_fn_path = getattr(task, "_case_id_fn_path", "")

    cases_json = json.dumps([
        c.__dict__ if hasattr(c, "__dict__") else dict(c)
        for c in task.cases
    ])

    print(f"  Launching GRPO on Modal ({gpu})...")
    print(f"  Base model: {model_path}")
    print(f"  vLLM endpoint: {vllm_endpoint}")
    print(f"  LoRA rank: {lora_rank}, group_size: {group_size}, steps: {max_steps}")
    print(f"  Cases: {task.n}")

    result = _grpo_train.remote(
        cases_json=cases_json,
        score_fn_path=score_fn_path,
        render_fn_path=render_fn_path,
        case_id_fn_path=case_id_fn_path,
        system_prompt=task.system_prompt,
        task_name=task.name,
        model_path=model_path,
        vllm_endpoint=vllm_endpoint,
        lora_rank=lora_rank,
        group_size=group_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        temperature=temperature,
        output_path=output_path,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    return GRPOResult(
        task_name=task.name,
        model_path=model_path,
        vllm_endpoint=vllm_endpoint,
        adapter_path=result.get("adapter_path", output_path),
        metrics=result.get("metrics", {}),
        config={
            "lora_rank": lora_rank,
            "group_size": group_size,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "gpu": gpu,
        },
    )


# --- CLI entrypoint (modal run) ---------------------------------------------

@_grpo_app.local_entrypoint()
def cli(
    vllm_endpoint: str = "",
    model_path: str = "Qwen/Qwen3.5-4B",
    lora_rank: int = 32,
    group_size: int = 4,
    max_steps: int = 5,
    learning_rate: float = 1e-5,
    max_tokens: int = 512,
    temperature: float = 1.0,
    output_path: str = "/root/thomas-grpo",
    # Domain: "tutor" or "eerie"
    domain: str = "tutor",
    wandb_project: str = "thomas",
    wandb_name: str = "",
):
    """Run GRPO LoRA RL on Modal via `modal run`.

    Usage::

        modal run -m thomas.trl_grpo --vllm-endpoint https://... --max-steps 5
    """
    import json as _json

    if domain == "tutor":
        from .domains.tutor import score, render, case_id, TUTOR_SYSTEM_PROMPT
        from .domains.tutor_scenarios import CASES
        score_fn_path = "thomas.domains.tutor.score"
        render_fn_path = "thomas.domains.tutor.render"
        case_id_fn_path = "thomas.domains.tutor.case_id"
        system_prompt = TUTOR_SYSTEM_PROMPT
        cases = CASES
    elif domain == "eerie":
        from .domains.eerie import score, render, case_id, SYSTEM_PROMPT, load_cases
        score_fn_path = "thomas.domains.eerie.score"
        render_fn_path = "thomas.domains.eerie.render"
        case_id_fn_path = "thomas.domains.eerie.case_id"
        system_prompt = SYSTEM_PROMPT
        cases = load_cases(n_cases=18)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    cases_json = _json.dumps([
        c.__dict__ if hasattr(c, "__dict__") else dict(c) for c in cases
    ])

    wb_name = wandb_name or f"thomas-{domain}-{model_path.split('/')[-1]}"

    print(f"  Domain: {domain}")
    print(f"  vLLM endpoint: {vllm_endpoint}")
    print(f"  Model: {model_path}")
    print(f"  LoRA rank: {lora_rank}, group_size: {group_size}, steps: {max_steps}")
    print(f"  W&B: {wandb_project}/{wb_name}")
    print(f"  Cases: {len(cases)}")
    print()

    result = _grpo_train.remote(
        cases_json=cases_json,
        score_fn_path=score_fn_path,
        render_fn_path=render_fn_path,
        case_id_fn_path=case_id_fn_path,
        system_prompt=system_prompt,
        task_name=domain,
        model_path=model_path,
        vllm_endpoint=vllm_endpoint,
        lora_rank=lora_rank,
        group_size=group_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        temperature=temperature,
        output_path=output_path,
        wandb_project=wandb_project,
        wandb_name=wb_name,
    )

    print(f"\n=== GRPO RESULT ===")
    print(f"  Adapter: {result.get('adapter_path')}")
    print(f"  Metrics: {result.get('metrics', {})}")
