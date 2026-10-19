"""Dream-stream Phase 0 runner — 4 arms × 18 foundermath fluency cases.

Arms (see README.md): direct / cot / dream / filler, identical case set,
identical max_tokens, answer-only scoring via foundermath's own
``score_text`` — the last ``Answer:`` line wins there, so the stream is
invisible to the verifier by construction.

Usage (sampling-only, cheap; still prints token counts):

    cd <synthetic repo>   # so ./.env and foundermath resolve
    .venv/Scripts/python.exe <thomas repo>/examples/dream_stream/run_phase0_tinker.py

Prints per-arm pass counts, per-case breakdown, truncation counts, token
usage, and the Tinker console URL. Receipt JSON goes to
``docs/receipts/`` under the cwd.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(".env"))

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from foundermath.sources.fluency_bank import bank_to_cases
from foundermath.tinker_env import DEFAULT_BANK, score_text, system_line_for

MODEL_NAME = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5_disable_thinking"  # thinking blocks break a grader check
MAX_TOKENS = 768  # identical budget across arms; 512 truncated the direct arm

ARM_INSTRUCTIONS = {
    "direct": "",  # system_line_for alone
    "cot": " Think step by step, briefly, then give the Answer line.",
    "cot_hot": " Think step by step, briefly, then give the Answer line.",
    "dream": (
        " Before answering, dream: between <dream> and </dream>,"
        " free-associate — images, half-related facts, whatever surfaces."
        " It does not need to be relevant or correct. Then wake up and"
        " give the Answer line."
    ),
    "filler": " Before answering, output one line of dots ('... ... ...'), then the Answer line.",
}
# Greedy arms are deterministic: n=1 replicates them exactly, more samples
# buy nothing. Hot arms get n=16 so dream-vs-cot_hot is temp-matched.
ARM_TEMP = {"direct": 0.0, "cot": 0.0, "cot_hot": 1.0, "dream": 1.0, "filler": 0.0}
ARM_N = {"direct": 1, "cot": 1, "cot_hot": 16, "dream": 16, "filler": 1}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return center - half, center + half


async def run_arm(sc, arm: str, cases, renderer) -> dict:
    sampling_client = sc.create_sampling_client(base_model=MODEL_NAME)
    stop = renderer.get_stop_sequences()

    async def one(case):
        system = system_line_for(case) + ARM_INSTRUCTIONS[arm]
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": m.role, "content": m.content} for m in case.input.messages]
        prompt = renderer.build_generation_prompt(msgs)
        result = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=ARM_N[arm],
            sampling_params=tinker.SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=ARM_TEMP[arm],
                stop=stop,
            ),
        )
        rows = []
        for seq in result.sequences:
            message, _term = renderer.parse_response(list(seq.tokens))
            content = renderers.get_text_content(message)
            reward, extracted, parse_ok, detail = score_text(case, content)
            rows.append({
                "case_id": case.id,
                "reward": reward,
                "extracted": str(extracted),
                "parse_ok": bool(parse_ok),
                "stop_reason": seq.stop_reason,
                "sampled_tokens": len(seq.tokens),
                "prompt_tokens": prompt.length,
                "reply": content,
            })
        return rows

    nested = await asyncio.gather(*[one(c) for c in cases])
    rows = [r for case_rows in nested for r in case_rows]
    n_pass = sum(r["reward"] for r in rows)
    return {
        "arm": arm,
        "temperature": ARM_TEMP[arm],
        "n": len(rows),
        "pass": n_pass,
        "pass_rate": n_pass / len(rows),
        "truncated": sum(1 for r in rows if r["stop_reason"] == "length"),
        "sampled_tokens": sum(r["sampled_tokens"] for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "rows": rows,
    }


async def main() -> None:
    sc = tinker.ServiceClient()
    caps = await sc.get_server_capabilities_async()
    names = [m.model_name for m in caps.supported_models]
    assert any(MODEL_NAME in n for n in names), f"{MODEL_NAME} not in {names}"

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tokenizer)

    cases = bank_to_cases(DEFAULT_BANK)
    print(f"model={MODEL_NAME} renderer={RENDERER_NAME} cases={len(cases)} "
          f"max_tokens={MAX_TOKENS}")

    arms = {}
    for arm in ARM_INSTRUCTIONS:
        res = await run_arm(sc, arm, cases, renderer)
        arms[arm] = res
        lo, hi = wilson(res["pass"], res["n"])
        print(f"arm {arm:8s} temp={res['temperature']}  n={res['n']:3d}  "
              f"pass {int(res['pass']):3d}  rate {res['pass_rate']:.2f} "
              f"[{lo:.2f}, {hi:.2f}]  truncated={res['truncated']}  "
              f"sampled_tok={res['sampled_tokens']}")

    # Per-case cross-arm breakdown: mean reward per case (hot arms) or the
    # single greedy result.
    print("\ncase_id                          direct cot  cot_hot dream filler")
    for case in cases:
        marks = []
        for arm in ARM_INSTRUCTIONS:
            rs = [r["reward"] for r in arms[arm]["rows"] if r["case_id"] == case.id]
            m = sum(rs) / len(rs)
            marks.append("  .    " if m == 0 else ("  1    " if m == 1 else f" {m:.2f}  "))
        print(f"{case.id:32s} {''.join(marks)}")

    receipt = {
        "date": str(date.today()),
        "experiment": "dream-stream phase 0b (5 arms, temp-matched, n=16 hot)",
        "model": MODEL_NAME,
        "renderer": RENDERER_NAME,
        "max_tokens": MAX_TOKENS,
        "arm_n": ARM_N,
        "console_url": sc.get_console_url(),
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in arms.items()},
        "rows": {k: v["rows"] for k, v in arms.items()},
    }
    out = Path("docs/receipts") / f"{date.today()}-dream-stream-phase0b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(f"\nreceipt: {out}")
    print(f"console: {receipt['console_url']}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
