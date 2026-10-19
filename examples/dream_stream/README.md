# Dream-stream experiment

Can a *divergent* auxiliary token stream — a dream — improve a model's
verified final answer, the way convergent reasoning (CoT) does? The dream
is accountable to nothing except its causal effect on the answer: the
verifier scores only the answer, never the stream.

## Prior art (grounding)

**Quiet-STaR** (Zelikman et al. 2024, arXiv:2403.09629). Think/talk/learn:
after every token, generate a rationale between learnable
`<|startthought|>` / `<|endthought|>` meta-tokens (think); a learned
mixing head blends the next-token log-probs predicted with and without
the thought (talk); REINFORCE increases the likelihood of thoughts that
raised the log-prob of the *next n_true ground-truth tokens* (learn —
non-myopic, beyond single next token). Continued pretraining of Mistral-7B
on OpenWebMath/C4 gave zero-shot GSM8K 5.9%→10.9% and CommonsenseQA
36.3%→47.2% with no task fine-tuning. Key findings: thoughts
disproportionately help hard-to-predict tokens; the mixing head is load-
bearing — ablate it and the model learns to ignore thoughts; reward is
unstable because the thought→contribution mapping is itself learned.

**Coconut** (Hao et al. 2024, arXiv:2412.06769). Latent mode between
`<bot>`/`<eot>`: the last hidden state is fed back directly as the next
input embedding — reasoning in continuous space, never decoded to text.
Fully differentiable (no REINFORCE). Multi-stage curriculum: stage k
replaces the first k language CoT steps with k×c continuous thoughts
(c=1–2 stable; c=3 spikes the loss). Emergent BFS-like behavior — the
continuous thought encodes several candidate next steps at once; beats
CoT on ProsQA/logical tasks needing backtracking, with fewer inference
tokens.

**Pause / filler tokens** (Goyal et al. 2023; Pfau et al. 2024). Extra
non-semantic tokens add compute but no content — the control condition
that separates "more compute" from "stream content matters".

## What "dream" adds

All of the above optimize the stream to *predict or justify* — convergent.
The dream arm is deliberately divergent: high-temperature, free-associative,
not required to be faithful or even relevant. Hypothesis: associative noise
sometimes surfaces structure that convergent CoT prunes too early. The
experiment exists to falsify that cheaply before any training spend.

## Phase ladder

### Phase 0 — inference only, no training (cheap, do first)

Four arms over the *identical* case set, same base model, same
`max_tokens` budget, answer-only scoring:

| Arm | Stream | Purpose |
|---|---|---|
| A | none — direct answer | floor |
| B | convergent CoT, then answer | the incumbent |
| C | dream stream (temp ≥ 1.0, free-association instruction), then answer | the hypothesis |
| D | filler stream (`... ... ...`, matched length), then answer | compute-only control |

Verdict via gonogo cards (Wilson CIs), pass@1 on the answer only.

- Proceed iff C > A, C ≥ B, and C > D. C > D is the interesting bar:
  it says the *content* of the divergent stream matters, not just the
  extra forward passes.
- If C ≤ D: extra compute is doing all the work; negative result, write
  it up, stop.

### Phase 1 — RL over the dream (STaR-style, GRPO)

Completion = `<dream>...</dream>` + `ANSWER: x`. Reward comes from the
stripped answer only; gradient reaches dream tokens through the shared
completion log-probs — exactly the STaR mechanism, answer-conditioned
(Quiet-STaR conditions on future ground-truth text instead). Train arm C
and arm B identically (same steps, LR, budget); compare post-train deltas
with thomas `compare()`.

### Phase 2 — architecture-level (deferred, not Tinker-reachable)

The real Quiet-STaR/Coconut versions need what LoRA services can't do:
- Quiet-STaR: new meta-token embeddings + a mixing head on hidden states.
- Coconut: hidden-state feedback into the embedding input.
Both are full-parameter, custom-forward work — that's the Modal nanogpt
path (`thomas.pretrain` stub), not `post_train`. Phase 0/1 deliberately
stay inside what thomas can drive today.

## Why thomas needs no changes

The dream is a domain-side concern:

- `render_messages` adds the dream instruction to the prompt (arm-specific).
- `score_text` is wrapped: strip everything up to the `ANSWER:` marker,
  then call the domain's real `score_text`. One reward function preserved —
  baseline and RL loop see identical numbers (thomas rule 2).
- `ThomasEnv` is single-step; dream + answer fit in one completion.

## Cost gates

Phase 0 is sampling-only — may run without approval, but report token
counts. Phase 1 is a training run — report the resolved config and stop
for approval (thomas rule 5).

## Open questions to answer in the writeup

- Does dream-stream benefit concentrate on hard cases, the way Quiet-STaR
  thoughts concentrate on hard tokens?
- Does the dream arm's advantage survive temp-matching (C at temp 1.0 vs
  B sampled at temp 1.0)? If not, it's just sampling noise.
- Counterfactual dream (dream about a *different but related* problem):
  if that also helps, the mechanism is association-priming, not rehearsal.

## Phase 0b result (2026-10-01, receipt in the domain repo)

Qwen3.5-4B, 18 foundermath fluency cases, max_tokens=768, hot arms n=16
(temp 1.0), greedy arms n=1:

| arm | temp | rate | 95% Wilson |
|---|---|---|---|
| direct | 0.0 | 0.78 | [0.55, 0.91] |
| cot | 0.0 | 0.78 | [0.55, 0.91] |
| cot_hot | 1.0 | 0.78 | [0.73, 0.83] |
| dream | 1.0 | 0.72 | [0.66, 0.76] |
| filler | 0.0 | 0.83 | [0.61, 0.94] |

Verdict on math fluency: **negative**. Dream ≤ temp-matched CoT, and the
Phase 0 filler surprise did not replicate at a larger token budget. The
dream hypothesis is falsified for short-horizon verified math with an
already-competent model. The open door is open-ended generation, where
there is no single right answer to converge on — that's the K3-mini /
eerie-register arm (option 3), which needs a few-shot continuation design
(base model, no instruct tuning) rather than this instruction protocol.
