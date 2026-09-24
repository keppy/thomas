"""thomas Tinker Env — one Task Case as one Tinker RL episode.

The shared spine of every thomas RL domain:

* One Case → one ``Env``.
* ``initial_observation()`` renders the case messages with the model's
  renderer, plus the Task's system prompt.
* ``step(action)`` decodes tokens → text → ``task.score_text(case, text)``
  → reward. Single step, ``episode_done=True``.
* ``CaseGroupBuilder`` yields ``group_size`` copies (GRPO needs groups).
* ``CaseDataset`` / ``CaseDatasetBuilder`` iterate over the case list.

The reward is ``task.score_text`` — the same function any offline scorer
calls, so the baseline and the RL loop see identical numbers.

Built against **tinker-cookbook 0.5.x / tinker 0.29.x** — ``Action``,
``ActionExtra``, ``StepResult``, ``EnvGroupBuilder``, ``RLDataset`` are
that version's shapes. Read ``tinker_cookbook/rl/types.py`` and one
worked recipe before trusting this; do not invent the interface.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .task import Task


class ThomasEnv(Env):
    """Single-turn env over one Case, scored by ``task.score_text``."""

    def __init__(
        self,
        case: Any,
        task: Task,
        renderer: renderers.Renderer,
    ) -> None:
        self.case = case
        self.task = task
        self.renderer = renderer

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _messages(self) -> list[renderers.Message]:
        msgs: list[renderers.Message] = [
            {"role": "system", "content": self.task.system_prompt}
        ]
        msgs += self.task.render_messages(self.case)
        return msgs

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return (
            self.renderer.build_generation_prompt(self._messages()),
            self.stop_condition,
        )

    async def step(
        self, action: Action, *, extra: ActionExtra | None = None
    ) -> StepResult:
        message, termination = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        reward, detail = self.task.score_text(self.case, content)

        cid = self.task.case_id(self.case)
        metrics: Metrics = {
            "reward": reward,
            "clean_stop": float(termination.is_clean),
        }
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
            logs={
                "case_id": cid,
                "detail": str(detail),
                "response": content,
            },
        )


# --- Group builder (GRPO) ----------------------------------------------------


@dataclass(frozen=True)
class ThomasGroupBuilder(EnvGroupBuilder):
    """``group_size`` copies of one case — GRPO centers rewards across them."""

    case: Any
    task: Task
    renderer: renderers.Renderer
    group_size: int
    dataset_name: str = "thomas"

    async def make_envs(self) -> Sequence[Env]:
        return [
            ThomasEnv(self.case, self.task, self.renderer)
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name, self.task.name]


# --- Dataset -----------------------------------------------------------------


class ThomasDataset(RLDataset):
    """Batches of ``ThomasGroupBuilder``, one builder per case.

    ``batch_size`` is capped at the number of cases so a batch never
    repeats a case.
    """

    def __init__(
        self,
        task: Task,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
    ) -> None:
        if not task.cases:
            raise ValueError("no cases")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.task = task
        self.batch_size = min(batch_size, len(task.cases))
        self.group_size = group_size
        self.renderer = renderer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self.task.cases)
        rows = [
            self.task.cases[(start + i) % len(self.task.cases)]
            for i in range(self.batch_size)
        ]
        return [
            ThomasGroupBuilder(case, self.task, self.renderer, self.group_size)
            for case in rows
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.task.cases) / self.batch_size)


# --- Dataset builder (chz) ---------------------------------------------------


@chz.chz
class ThomasDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    # Cases serialized as JSON — chz can't hold arbitrary objects.
    # The Task's cases must be (de)serializable via the domain's own
    # to_json / from_json functions, passed in here.
    cases_json: str
    from_json_fn: str  # dotted path to a callable: str -> list[case]
    system_prompt: str = ""
    render_messages_fn: str = ""  # dotted path to a render fn (optional)
    case_id_fn: str = ""  # dotted path to a case_id fn (optional)
    score_text_fn: str = ""  # dotted path to score_text (optional)
    oracle_reply_fn: str = ""  # dotted path to oracle_reply (optional)
    task_name: str = "thomas"

    async def __call__(self) -> tuple[ThomasDataset, None]:
        from .task import Task, default_render_messages, default_case_id

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(
            self.renderer_name, tokenizer=tokenizer
        )
        cases = _import_callable(self.from_json_fn)(self.cases_json)

        score_text = _import_callable(self.score_text_fn) if self.score_text_fn else None
        oracle_reply = _import_callable(self.oracle_reply_fn) if self.oracle_reply_fn else None
        render_messages = _import_callable(self.render_messages_fn) if self.render_messages_fn else default_render_messages
        case_id = _import_callable(self.case_id_fn) if self.case_id_fn else default_case_id

        task = Task(
            name=self.task_name,
            cases=cases,
            score_text=score_text,
            oracle_reply=oracle_reply,
            system_prompt=self.system_prompt,
            render_messages=render_messages,
            case_id=case_id,
        )
        return (
            ThomasDataset(
                task=task,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
            ),
            None,
        )


def _import_callable(dotted: str):
    """Import a callable from a dotted path like 'eerie_rl.case.load_cases_from_json'."""
    import importlib

    mod, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), name)
