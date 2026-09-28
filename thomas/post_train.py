"""Post-train: Case → TinkerEnv → LoRA RL → checkpoint → re-baseline.

Uses the Tinker cookbook's RL training loop with ``ThomasDataset``.
Config: smallest Qwen, LoRA rank 32, ``group_size=4``, short ``max_steps``.

**Before running: report the config and stop. The operator decides whether
to spend credits** (AGENTS.md rule 5). When approved, run it, then re-sample the trained
sampler and report before/after.

The training loop saves a ``sampler_path`` in ``checkpoints.jsonl``. After
the run, we read it with ``get_last_checkpoint`` and pass it to
``create_sampling_client(model_path=sampler_path)`` for the after-baseline.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .task import Task
from .baseline import BaselineResult, baseline, _resolve_renderer


def _validate_eval_split(train_task: Task, eval_task: Task) -> None:
    """Reject obvious training/eval leakage before any sampling or training."""
    if eval_task.score_text is not train_task.score_text:
        raise ValueError("eval_task must use the same score_text function as training")
    train_ids = [train_task.case_id(c) for c in train_task.cases]
    eval_ids = [eval_task.case_id(c) for c in eval_task.cases]
    if len(set(train_ids)) != len(train_ids) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("case ids must be unique within each split")
    if set(train_ids) & set(eval_ids):
        raise ValueError("training and eval case ids overlap")
    def prompts(task):
        return {json.dumps(task.render_messages(c), sort_keys=True, ensure_ascii=False)
                for c in task.cases}
    if prompts(train_task) & prompts(eval_task):
        raise ValueError("training and eval prompts overlap (even under different ids)")


def _new_log_path(task_name: str) -> str:
    """Unique per paid run; never read a checkpoint left by an older attempt."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", task_name).strip(".-") or "task"
    return f"./logs/thomas-{safe_name}-{uuid.uuid4().hex}"


@dataclass
class PostTrainResult:
    """The output of a post-train run."""

    task_name: str
    model: str
    before: BaselineResult
    after: BaselineResult
    config: dict[str, Any] = field(default_factory=dict)
    sampler_path: str | None = None


def post_train(
    task: Task,
    model: str | None = None,
    *,
    eval_task: Task | None = None,
    lora_rank: int = 32,
    group_size: int = 4,
    max_steps: int = 5,
    learning_rate: float = 1e-5,
    max_tokens: int = 512,
    temperature: float = 1.0,
    target: float = 0.95,
    base_url: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    # Dotted paths to the domain's functions, so the chz builder can
    # resolve them at deserialization time. Required: score_text_fn.
    score_text_fn: str = "",
    oracle_reply_fn: str = "",
    render_messages_fn: str = "",
    case_id_fn: str = "",
    from_json_fn: str = "thomas.post_train._default_from_json",
) -> PostTrainResult:
    """Run a LoRA RL training loop on a Task.

    1. Baseline: sample the base model, get the gonogo card.
    2. Train: LoRA RL on the Task's cases via Tinker.
    3. After: re-sample the trained model (from the saved sampler checkpoint),
       get the new card.
    4. Return before/after for comparison.

    **Ask before spending credits.** This makes API calls that cost money.
    """
    if eval_task is None:
        raise ValueError("eval_task is required: before/after must use untouched cases")
    _validate_eval_split(task, eval_task)
    if not score_text_fn or not callable(task.score_text):
        raise ValueError("score_text_fn must identify task.score_text by dotted import path")
    module, dot, attr = score_text_fn.rpartition(".")
    if not dot or getattr(importlib.import_module(module), attr) is not task.score_text:
        raise ValueError("score_text_fn must resolve to the same function as task.score_text")
    _validate_builder_roundtrip(task, from_json_fn, oracle_reply_fn,
                                render_messages_fn, case_id_fn)
    log_path = _new_log_path(task.name)

    # --- 1. Baseline (untouched eval cases only) ---
    print("--- Baseline (before training; untouched eval set) ---")
    before = baseline(eval_task, model=model, max_tokens=max_tokens, target=target)

    # --- 2. Train ---
    from dotenv import load_dotenv
    if not os.environ.get("TINKER_API_KEY"):
        for p in [Path.cwd() / ".env", Path.home() / ".env"]:
            if p.exists():
                load_dotenv(p)
                break

    import tinker
    from tinker_cookbook.rl import train as rl_train
    from tinker_cookbook import checkpoint_utils
    from .tinker_env import ThomasDatasetBuilder

    model_name = before.model
    renderer_name, renderer = _resolve_renderer(model_name)

    cases_json = _serialize_cases(task)

    dataset_builder = ThomasDatasetBuilder(
        batch_size=min(group_size * 4, task.n),
        group_size=group_size,
        num_batches=max_steps,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        cases_json=cases_json,
        from_json_fn=from_json_fn,
        system_prompt=task.system_prompt,
        task_name=task.name,
        score_text_fn=score_text_fn,
        oracle_reply_fn=oracle_reply_fn,
        render_messages_fn=render_messages_fn,
        case_id_fn=case_id_fn,
    )

    train_config = rl_train.Config(
        log_path=log_path,
        model_name=model_name,
        recipe_name=f"thomas-{task.name}",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        max_tokens=max_tokens,
        temperature=temperature,
        max_steps=max_steps,
        save_every=max_steps,
        eval_every=0,
        base_url=base_url,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    print(f"\n--- RL Training Config ---")
    print(f"  Model:          {model_name}")
    print(f"  LoRA rank:      {lora_rank}")
    print(f"  Group size:     {group_size}")
    print(f"  Max steps:      {max_steps}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Cases:          {task.n}")
    print(f"  Log path:       {log_path}")
    print(f"\n  This will take {max_steps} gradient steps on {model_name}.")

    asyncio.run(rl_train.main(train_config))

    # --- Read the saved checkpoint ---
    ckpt = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
    sampler_path = ckpt.sampler_path if ckpt else None
    if sampler_path:
        print(f"\n  Trained sampler: {sampler_path}")
    else:
        raise RuntimeError(f"no trained sampler checkpoint found in {log_path}; cannot report after-eval")

    # --- 3. After: re-sample the trained model ---
    print("\n--- Baseline (after training) ---")
    after = _baseline_with_sampler(eval_task, model_name, sampler_path, target, max_tokens=max_tokens)

    return PostTrainResult(
        task_name=task.name,
        model=model_name,
        before=before,
        after=after,
        config={
            "lora_rank": lora_rank,
            "group_size": group_size,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        sampler_path=sampler_path,
    )


async def _sample_with_client(task, sampling_client, renderer, stop, max_tokens=512):
    """Sample each case once and score."""
    import tinker
    from tinker_cookbook import renderers

    per_case = []
    outputs_by_id = {}
    for i, case in enumerate(task.cases):
        cid = task.case_id(case)
        messages = [{"role": "system", "content": task.system_prompt}]
        messages += task.render_messages(case)
        prompt = renderer.build_generation_prompt(messages)
        resp = await sampling_client.sample_async(
            prompt=prompt, num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens, temperature=0.0, stop=stop,
            ),
        )
        msg, termination = renderer.parse_response(list(resp.sequences[0].tokens))
        content = renderers.get_text_content(msg)
        outputs_by_id[cid] = content
        reward, detail = task.score_text(case, content)
        per_case.append({"case_id": cid, "output": content, "reward": reward, "detail": detail})
        status = "✓" if reward >= 0.99 else " "
        snippet = content[:120].replace("\n", " ")
        print(f"  {status} [{i+1:2d}/{task.n}] {cid:30s} r={reward:.3f}  {snippet}...")
    return per_case, outputs_by_id


def _baseline_with_sampler(
    task: Task, model_name: str, sampler_path: str, target: float, *, max_tokens: int = 512
) -> BaselineResult:
    """Run untouched eval against the trained sampler only."""
    from dotenv import load_dotenv
    if not os.environ.get("TINKER_API_KEY"):
        for p in [Path.cwd() / ".env", Path.home() / ".env"]:
            if p.exists():
                load_dotenv(p)
                break

    import tinker
    from tinker_cookbook import renderers

    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(model_path=sampler_path)

    renderer_name, renderer = _resolve_renderer(model_name)
    stop = renderer.get_stop_sequences()

    per_case, outputs_by_id = asyncio.run(
        _sample_with_client(task, sampling_client, renderer, stop, max_tokens=max_tokens)
    )

    # Gonogo card
    report = None
    try:
        from gonogo import Case as GCase, evaluate

        gcases = [GCase(input={"output": outputs_by_id.get(task.case_id(c), "")},
                        expected={"reward": 1.0}, id=task.case_id(c))
                  for c in task.cases]

        def _agent(gcase):
            return outputs_by_id.get(gcase.id, "")

        def _scorer(output, expected, gcase):
            orig = next((c for c in task.cases if task.case_id(c) == gcase.id), None)
            if orig is None:
                return False, 0.0, f"case {gcase.id} not found"
            reward, detail = task.score_text(orig, output)
            return reward >= 0.99, reward, str(detail)

        report = evaluate(_agent, gcases, scorer=_scorer,
                          task=f"thomas-after/{task.name}", target=target)
        print(f"\n{report.summary() if hasattr(report, 'summary') else report}")
    except ImportError:
        pass

    return BaselineResult(
        task_name=task.name, model=model_name,
        per_case=per_case, report=report,
        console_url=sc.get_console_url(),
    )


def _validate_builder_roundtrip(task: Task, from_json_fn: str, oracle_reply_fn: str,
                                render_messages_fn: str, case_id_fn: str) -> None:
    """Check the deserialized Tinker case/scorer/prompt before any API call."""
    from .task import default_case_id, default_render_messages

    def resolve(path):
        module, dot, name = path.rpartition(".")
        if not dot:
            raise ValueError(f"expected dotted callable path: {path!r}")
        fn = getattr(importlib.import_module(module), name)
        if not callable(fn):
            raise ValueError(f"not callable: {path!r}")
        return fn

    serialized = _serialize_cases(task)
    original_payload = json.loads(serialized)
    restored = resolve(from_json_fn)(serialized)
    if len(restored) != task.n:
        raise ValueError("Tinker case round-trip changed case count")
    render = resolve(render_messages_fn) if render_messages_fn else default_render_messages
    cid = resolve(case_id_fn) if case_id_fn else default_case_id
    if oracle_reply_fn and resolve(oracle_reply_fn) is not task.oracle_reply:
        raise ValueError("oracle_reply_fn differs from Task.oracle_reply")
    for original, case, payload in zip(task.cases, restored, original_payload):
        try:
            restored_payload = case.__dict__ if hasattr(case, "__dict__") else dict(case)
            # Check *all* serialized fields, not just the oracle reply. A
            # deserializer can preserve the oracle reward while changing how
            # wrong answers score (e.g. dropping an expected-label field).
            if json.loads(json.dumps(restored_payload)) != payload:
                raise ValueError("serialized case fields changed")
            replies = [task.oracle_reply(original), "", "__thomas_invalid_reply__"]
            same = (str(cid(case)) == str(task.case_id(original))
                    and render(case) == task.render_messages(original)
                    and all(task.score_text(case, text) == task.score_text(original, text)
                            for text in replies))
        except Exception as exc:
            raise ValueError(f"Tinker case round-trip cannot score/render: {exc}") from exc
        if not same:
            raise ValueError(f"Tinker case round-trip changed reward, id or prompt: "
                             f"{task.case_id(original)}")


def _serialize_cases(task: Task) -> str:
    """Serialize a Task's cases to JSON for the chz builder."""
    import json
    return json.dumps([
        c.__dict__ if hasattr(c, "__dict__") else dict(c)
        for c in task.cases
    ])


def _default_from_json(s: str) -> list:
    """Generic deserializer — assumes cases were serialized as dicts.

    Reconstructs as SimpleNamespace objects so attribute access (``case.id``,
    ``case.prompt``, etc.) works just like the original dataclass.
    """
    import json
    from types import SimpleNamespace
    raw = json.loads(s)
    return [SimpleNamespace(**d) if isinstance(d, dict) else d for d in raw]
