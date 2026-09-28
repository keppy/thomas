# thomas

Give thomas a case set and a reward function. It gives you a baseline card, a
training run, and a before/after comparison, with
[gonogo](https://github.com/keppy/gonogo) making the ship / don't-ship call at
each end. The domain supplies the case shape and the reward; thomas supplies
the plumbing.

*`thomas.train()`: named after Thomas the Tank Engine (it's a train) and Thomas
Barrow of Downton Abbey (it's the undercrogue that earns its place).*

```
                    ┌── encoder SFT ── (text, label) → Modal L4 → calibrated classifier
  Case set ──► thomas ┼── RL (Tinker) ── Case → TinkerEnv → LoRA RL → checkpoint
                    └── RL (Modal) ─── Case → TRL GRPO + vLLM colocate → LoRA adapter
                                  │
                                  └──► gonogo: verdict, interval, operating point
```

One reward function, three consumers: the gonogo card, the RL gradient, and the
before/after comparison all call the same `score_text`, so they see the same
numbers.

[![Fine-tuning an encoder and getting a go/no-go verdict — thomas + gonogo](https://i.ytimg.com/vi/ozWITnaJtf4/maxresdefault.jpg)](https://youtu.be/ozWITnaJtf4)

**Video (40:33):** the whole pipeline worked live: plan, contract, a subagent building the thomas training path, two dead runs and one false alarm, and the Modal fine-tune of [ModernBERT-small-v2](https://huggingface.co/johnnyboycurtis/ModernBERT-small-v2) on an L4 for under $1, scored on the gonogo Banking77 canary. [Writeup](https://www.keppylab.com/blog/2026/09/21/banking77-canary-872-pass-two-dead-runs-one-false-alarm/).

- 87.2% [82.5%, 90.8%] pass rate against a 95% target → **AUTOMATE WITH REVIEW**
- at confidence ≥ 0.91: 98.3% precision [95.1%, 99.4%] on 71% of cases, the rest routed to a human
- calibration error 0.03 after temperature scaling; +10.0 points over the TF-IDF baseline on the same 250 cases (p < 0.001)

## Install

```bash
pip install -e .                    # core + gonogo
pip install -e ".[encoder]"         # + encoder fine-tune on Modal (torch, transformers, modal)
pip install -e ".[post_train]"      # + Tinker RL
```

Not on PyPI yet; install from a clone.

## Try it

```bash
python examples/banking77_encoder.py --dry-run     # free: prints the config and the split, spends nothing
python examples/banking77_predict.py --model-dir examples/artifacts/banking77-enc \
    "I lost my card" "why was I charged twice"   # the published artifact, on CPU
```

## Encoder fine-tune (Modal)

The cheapest path, and the one behind the Banking77 result above: a small
encoder with a classification head, temperature-scaled on a held-out split so
its confidences mean something to gonogo.

```bash
# (text, label) JSONL in, artifact dir out
modal run -m thomas.encoder_train --model johnnyboycurtis/ModernBERT-small-v2 --data cases.jsonl

# the Banking77 canary end to end (dry run first, it's free)
python examples/banking77_encoder.py --dry-run
python examples/banking77_encoder.py
```

The artifact is a HuggingFace `save_pretrained` dir plus `label2id.json`,
`temperature.json` and `metrics.json` (see [CONTRACT.md §4](docs/CONTRACT.md#4-encoder-artifact)).
`examples/banking77_predict.py` runs it interactively on CPU.

## RL on a domain

```python
import thomas
from thomas.domains import tutor, tutor_scenarios

task = thomas.Task(
    name="tutor",
    cases=tutor_scenarios.CASES,
    score_text=tutor.score,           # (case, text) -> (reward, detail)
    oracle_reply=tutor.oracle,        # a reply that scores 1.0
    system_prompt=tutor.TUTOR_SYSTEM_PROMPT,
    render_messages=tutor.render,
    case_id=tutor.case_id,
)
assert task.oracle_check()            # the env is winnable by construction

before = thomas.baseline(task)        # sample the base model, gonogo card (Tinker)
run = thomas.post_train(task, max_steps=5, score_text_fn="thomas.domains.tutor.score")
print(thomas.compare(before, run.after).summary())
```

`post_train` and every Modal entry point spend money. Each one prints its
config before launching.

### Example domains

- `thomas.domains.tutor`: a tutoring trace scored by deterministic hard checks.
  The tutor must not assert unverified correctness or leak the answer, must
  keep the problem's conditions visible, and gets no reward for solving it for
  the student. Ten single-turn math scenarios in `tutor_scenarios`.
- `thomas.domains.skill_classify`: label a problem with a skill id, exact match.
- `thomas.domains.eerie`: literary continuation scored by n-gram overlap (the
  pretrain-path eval).

## Modal GRPO (custom checkpoints)

```bash
modal run -m thomas.trl_grpo --domain tutor \
    --model-path Qwen/Qwen3-4B --max-steps 5 \
    --wandb-name my-run
```

Saves a LoRA adapter to a Modal volume; metrics go to the
W&B project `thomas` (Modal secret `wandb`).

Known constraints (each cost a run to learn):

- **Colocate memory:** vLLM's budget is `total_gpu × vllm_gpu_memory_utilization`
  *minus* everything already resident in the process (the trainer model). The
  utilization must cover trainer weights + vLLM weights + KV cache. 0.75 on an
  A100-40GB works for 4B models; set `vllm_max_model_length` (TRL 1.13 kwarg —
  not `max_model_len`) to something small, or vLLM sizes KV for the model's
  native context.
- **Load in bf16** (`torch_dtype="auto"`): a silent fp32 load doubles the
  resident model and breaks the budget.
- **Reward functions receive chat format:** conversational completions arrive
  as a list of `{"role", "content"}` messages, not a string. `trl_grpo`
  unwraps them; custom reward fns must too.
- **VLM-class models (e.g. Qwen3.5) are broken upstream** in TRL colocate —
  weight-sync name remapping for `*ForConditionalGeneration` architectures
  (huggingface/trl#5269). thomas ships a monkeypatch prefix remap, but prefer
  text-only `ForCausalLM` checkpoints on this path.
- Server mode (`vllm_mode="server"`) hangs on Modal's kernel 4.19; colocate
  is the supported mode.

## The contract with gonogo

thomas and gonogo agree on six things: case ids, reward → `passed` / `score`,
the confidence definition (`max(softmax(logits / T))` over the full label set),
the artifact a training run leaves on disk, eval splits, and the gonogo calls
thomas depends on. They're written down and versioned in **[docs/CONTRACT.md](docs/CONTRACT.md)**. It's contract version 1,
tested against gonogo-eval `>=0.2,<0.3`.

```python
from thomas.contract import read_artifact   # stdlib only, no torch needed
art = read_artifact("examples/artifacts/banking77-enc")
art["temperature"], len(art["label2id"])    # 0.78, 77
```

## Works with

| | |
| --- | --- |
| [gonogo](https://github.com/keppy/gonogo) | Turns a score into a ship / don't-ship decision, with intervals |
| [thomas](https://github.com/keppy/thomas) | Trains the model: encoder fine-tune on Modal, or RL on Tinker / Modal |
| [hermes-plugin-gonogo](https://github.com/keppy/hermes-plugin-gonogo) | gonogo as Hermes agent tools |
| [hermes-plugin-thomas](https://github.com/keppy/hermes-plugin-thomas) | thomas's encoder path as Hermes agent tools, GPU launches behind the approval gate |

The thomas ↔ gonogo contract is [above](#the-contract-with-gonogo).

## What thomas is not

- Not gonogo (that's the scoring/decision layer; thomas imports it)
- Not a model (that's Tinker, Modal or nanogpt; thomas drives them)
- Not a case format (that's the domain)
- Not an eval framework (that's gonogo + the domain's checks)

## Changelog

[CHANGELOG.md](CHANGELOG.md).

## License

MIT
