"""thomas — a training harness.

Case→reward→train: take a Case set and a ``score_text`` function, get a
baseline card, run a training loop, compare before/after. The domain
supplies the Case shape and the reward; thomas supplies the plumbing.

Two paths share one Case→reward interface:

    Case set ──► thomas.baseline   (sample base model, score, card)
             ──► thomas.post_train (Case → TinkerEnv → LoRA RL → checkpoint)
             ──► thomas.pretrain   (Case → shards → nanogpt loop → checkpoint)

Architecture:

    Case ─► Env ─► initial_observation() ─► sample ─► step() ─► reward
                 ↑                                          │
                 └── score_text() ────────────────────────┘

One reward function, used by both the env's ``step`` and any offline scorer,
so the baseline and the training loop see identical numbers.
"""

from __future__ import annotations

from .task import Task, ScoreFn
from .baseline import baseline, baseline_from_outputs, BaselineResult
from .compare import compare, Comparison

__version__ = "0.2.0"

# post_train requires tinker; import lazily
def post_train(*args, **kwargs):
    from .post_train import post_train as _pt
    return _pt(*args, **kwargs)

# pretrain invokes Modal; import lazily
def pretrain(*args, **kwargs):
    from .pretrain import pretrain as _pt
    return _pt(*args, **kwargs)

# handoff: pretrain (Modal) → post-train (Tinker)
def handoff(*args, **kwargs):
    from .handoff import handoff as _h
    return _h(*args, **kwargs)

# vLLM serving on Modal
def serve_vllm(*args, **kwargs):
    from .serve_vllm import serve as _s
    return _s(*args, **kwargs)

# TRL GRPO LoRA RL
def trl_grpo(*args, **kwargs):
    from .trl_grpo import run_grpo as _g
    return _g(*args, **kwargs)

def trl_grpo_modal(*args, **kwargs):
    from .trl_grpo import run_grpo_modal as _g
    return _g(*args, **kwargs)

__all__ = [
    "Task",
    "ScoreFn",
    "baseline",
    "baseline_from_outputs",
    "BaselineResult",
    "post_train",
    "pretrain",
    "handoff",
    "serve_vllm",
    "trl_grpo",
    "trl_grpo_modal",
    "compare",
    "Comparison",
]
