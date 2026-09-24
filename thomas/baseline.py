"""Baseline: sample the base model on a Task's cases, score, report.

This is the cheap step — no training, no gradient steps. It proves the
harness works end to end and establishes the baseline the training loop is
measured against.

The same ``score_text`` that drives the baseline is the one that will drive
the RL loop. One reward function, three consumers (gonogo, Tinker, compare).

If gonogo is installed, the baseline also produces a gonogo card. If not,
it prints the raw per-case breakdown — matching ``eerie_rl/smoke.py``'s
shape.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .task import Task


@dataclass
class BaselineResult:
    """The output of a baseline run.

    ``report`` is the gonogo.Report if gonogo scored it; ``per_case`` is
    always present for inspection and for comparison with a trained model.
    """

    task_name: str
    model: str
    per_case: list[dict[str, Any]] = field(default_factory=list)
    report: Any = None  # gonogo.Report or None
    console_url: str | None = None

    @property
    def pass_rate(self) -> float:
        n = len(self.per_case)
        return sum(1 for r in self.per_case if r["reward"] >= 0.99) / n if n else 0.0

    def summary(self) -> str:
        if self.report is not None and hasattr(self.report, "summary"):
            return self.report.summary()
        lines = [
            f"Task: {self.task_name}",
            f"Model: {self.model}",
            f"Cases: {len(self.per_case)}",
            f"Pass rate: {self.pass_rate:.1%}",
        ]
        if self.console_url:
            lines.append(f"Console: {self.console_url}")
        return "\n".join(lines)


def _pick_smallest_qwen(models: list) -> str:
    """Pick the smallest Qwen from supported_models."""
    import re

    qwen = [m for m in models if "qwen" in m.model_name.lower()]
    if not qwen:
        raise RuntimeError("No Qwen models in supported_models")
    pat = re.compile(r"(\d+(?:\.\d+)?)\s*B", re.IGNORECASE)

    def _b(m):
        hits = pat.findall(m.model_name)
        return float(hits[0]) if hits else 999

    return min(qwen, key=_b).model_name


def _resolve_renderer(model_name: str):
    """Resolve a renderer for the model, the cookbook's way."""
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    try:
        from tinker_cookbook.checkpoint_utils import (
            resolve_renderer_name_from_checkpoint_or_default,
        )
        name = resolve_renderer_name_from_checkpoint_or_default(model_name)
    except Exception:
        name = "qwen3" if "qwen3" in model_name.lower() else "role_colon"

    tokenizer = get_tokenizer(model_name)
    return name, renderers.get_renderer(name, tokenizer=tokenizer)


def baseline_from_outputs(
    task: Task,
    outputs: dict[str, str],
    *,
    target: float = 0.95,
) -> BaselineResult:
    """Score already-sampled outputs through the harness.

    Use this when you have the model's replies from another source (a file,
    a previous run, an API) and just need the card. No Tinker dependency.
    """
    per_case: list[dict[str, Any]] = []
    for case in task.cases:
        cid = task.case_id(case)
        text = outputs.get(cid, "")
        reward, detail = task.score_text(case, text)
        per_case.append({
            "case_id": cid,
            "output": text,
            "reward": reward,
            "detail": detail,
        })

    report = None
    try:
        from gonogo import Case as GCase, evaluate

        gcases = []
        for case in task.cases:
            cid = task.case_id(case)
            gcases.append(GCase(
                input={"output": outputs.get(cid, "")},
                expected={"reward": 1.0},
                id=cid,
            ))

        def _agent(gcase):
            return outputs.get(gcase.id, "")

        def _scorer(output, expected, gcase):
            orig = next((c for c in task.cases if task.case_id(c) == gcase.id), None)
            if orig is None:
                return False, 0.0, f"case {gcase.id} not found"
            reward, detail = task.score_text(orig, output)
            return reward >= 0.99, reward, str(detail)

        report = evaluate(
            _agent, gcases, scorer=_scorer,
            task=f"thomas-baseline/{task.name}", target=target,
        )
    except ImportError:
        pass

    return BaselineResult(
        task_name=task.name,
        model="external",
        per_case=per_case,
        report=report,
    )


def baseline(
    task: Task,
    model: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    target: float = 0.95,
) -> BaselineResult:
    """Sample the base model on a Task's cases and return the baseline.

    If ``model`` is None, picks the smallest available Qwen.
    Sampling-only; no training, no gradient steps.

    Mirrors ``eerie_rl/smoke.py``: load cases, verify oracle, sample once
    per case through the env's scoring path, print per-case breakdown.
    If gonogo is installed, also produce a gonogo card.
    """
    return asyncio.run(_baseline_async(task, model, temperature, max_tokens, target))


async def _baseline_async(task, model, temperature, max_tokens, target) -> BaselineResult:
    from dotenv import load_dotenv

    # Load API key: env var first, then .env in cwd, then ~/.env
    if not os.environ.get("TINKER_API_KEY"):
        for p in [Path.cwd() / ".env", Path.home() / ".env"]:
            if p.exists():
                load_dotenv(p)
                break
    assert os.environ.get("TINKER_API_KEY"), (
        "TINKER_API_KEY not found. Set it in .env or as an environment variable."
    )

    import tinker
    from tinker_cookbook import renderers

    sc = tinker.ServiceClient()
    caps = await sc.get_server_capabilities_async()
    model_name = model or _pick_smallest_qwen(caps.supported_models)
    print(f"Model: {model_name}")

    renderer_name, renderer = _resolve_renderer(model_name)
    print(f"Renderer: {renderer_name}")
    print(f"Cases: {task.n}")

    # --- Oracle check: every case's oracle reply must score 1.0 ---
    all_oracle = True
    for case in task.cases:
        reward, _ = task.score_text(case, task.oracle_reply(case))
        if reward < 0.99:
            cid = task.case_id(case)
            print(f"  ORACLE FAIL: {cid} reward={reward:.4f}")
            all_oracle = False
    if all_oracle:
        print(f"Oracle check: all {task.n} oracles score 1.0 ✓")
    else:
        print("ORACLE FAILURE — env is asking for something its grader cannot accept")
    print()

    # --- Sample the base model ---
    sampling_client = sc.create_sampling_client(base_model=model_name)
    stop = renderer.get_stop_sequences()

    per_case: list[dict[str, Any]] = []
    outputs_by_id: dict[str, str] = {}

    for i, case in enumerate(task.cases):
        cid = task.case_id(case)
        messages = [{"role": "system", "content": task.system_prompt}]
        messages += task.render_messages(case)

        prompt = renderer.build_generation_prompt(messages)
        resp = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            ),
        )
        msg, termination = renderer.parse_response(list(resp.sequences[0].tokens))
        content = renderers.get_text_content(msg)
        outputs_by_id[cid] = content

        reward, detail = task.score_text(case, content)
        per_case.append({
            "case_id": cid,
            "output": content,
            "reward": reward,
            "detail": detail,
        })

        status = "✓" if reward >= 0.99 else " "
        snippet = content[:120].replace("\n", " ")
        print(f"  {status} [{i+1:2d}/{task.n}] {cid:30s} r={reward:.3f}  {snippet}...")

    # --- Summary ---
    avg_reward = sum(r["reward"] for r in per_case) / len(per_case)
    n_pass = sum(1 for r in per_case if r["reward"] >= 0.99)
    print(f"\n{'='*60}")
    print(f"  Pass: {n_pass}/{task.n} ({100*n_pass/task.n:.1f}%)")
    print(f"  Avg reward: {avg_reward:.4f}")
    print(f"{'='*60}")

    # --- Gonogo card (if installed) ---
    report = None
    try:
        from gonogo import Case as GCase, evaluate

        # Wrap domain cases as gonogo cases for the card
        gcases = []
        for case in task.cases:
            cid = task.case_id(case)
            gcase = GCase(
                input={"output": outputs_by_id.get(cid, "")},
                expected={"reward": 1.0},
                id=cid,
            )
            gcases.append(gcase)

        def _agent(gcase):
            return outputs_by_id.get(gcase.id, "")

        def _scorer(output, expected, gcase):
            # Find the original case by id
            cid = gcase.id
            orig = next((c for c in task.cases if task.case_id(c) == cid), None)
            if orig is None:
                return False, 0.0, f"case {cid} not found"
            reward, detail = task.score_text(orig, output)
            return reward >= 0.99, reward, str(detail)

        report = evaluate(
            _agent, gcases, scorer=_scorer,
            task=f"thomas-baseline/{task.name}", target=target,
        )
        print(f"\nGonogo card:")
        print(report.summary() if hasattr(report, "summary") else report)
    except ImportError:
        pass  # gonogo not installed — raw breakdown is enough

    console_url = sc.get_console_url()
    print(f"\nConsole: {console_url}")

    return BaselineResult(
        task_name=task.name,
        model=model_name,
        per_case=per_case,
        report=report,
        console_url=console_url,
    )
