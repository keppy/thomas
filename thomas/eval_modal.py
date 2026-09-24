"""Modal-side evaluation for thomas: sample base / base+LoRA on a GPU.

`thomas.baseline` needs Tinker and cannot load custom adapters, so the
Modal (TRL GRPO) path evaluates through here instead: an offline vLLM
engine samples k completions per case, the domain's own `score_text`
scores them, and the results feed `thomas.baseline_from_outputs` +
`thomas.compare` locally. Same reward function on both sides of the
comparison — the "one reward function" rule holds.

Usage (local)::

    python -m thomas.eval_modal --model Qwen/Qwen3-4B-Instruct-2507 \
        --adapter /root/thomas-grpo   # omit adapter for the base model
"""

from __future__ import annotations

import json

import modal

from .trl_grpo import _grpo_image, _grpo_vol  # same image + volume as training

_eval_app = modal.App("thomas-eval")


@_eval_app.function(
    image=_grpo_image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/root/thomas-grpo": _grpo_vol},
)
def _eval_remote(
    cases_json: str,
    score_fn_path: str,
    render_fn_path: str,
    case_id_fn_path: str,
    system_prompt: str,
    model_path: str,
    adapter_path: str | None = None,
    k: int = 4,
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> str:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    import importlib

    score_mod_path, score_name = score_fn_path.rsplit(".", 1)
    score_fn = getattr(importlib.import_module(score_mod_path), score_name)
    render_mod_path, render_name = render_fn_path.rsplit(".", 1)
    render_fn = getattr(importlib.import_module(render_mod_path), render_name)
    cid_mod_path, cid_name = case_id_fn_path.rsplit(".", 1)
    case_id_fn = getattr(importlib.import_module(cid_mod_path), cid_name)

    cases = [dict(c) if not hasattr(c, "__dict__") else vars(c) for c in json.loads(cases_json)]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, enable_lora=adapter_path is not None,
              max_lora_rank=64 if adapter_path is not None else 16,
              max_model_len=4096, gpu_memory_utilization=0.85)
    sp = SamplingParams(n=k, max_tokens=max_tokens, temperature=temperature)

    lora_req = None
    if adapter_path is not None:
        lora_req = LoRARequest("adapter", 1, adapter_path)

    results: dict[str, list[dict]] = {}
    for c in cases:
        case = type("C", (), c)()  # attribute access like the dataclass
        msgs = [{"role": "system", "content": system_prompt}] + render_fn(case)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out = llm.generate([prompt], sp, lora_request=lora_req)[0].outputs
        rows = []
        for o in out:
            text = o.text
            reward, detail = score_fn(case, text)
            rows.append({"text": text, "reward": float(reward),
                         "detail": {k: v for k, v in detail.items() if k != "checks"},
                         "checks": detail.get("checks", {}), "case_id": case_id_fn(case)})
        results[case_id_fn(case)] = rows
        print(f"  {case_id_fn(case)}: pass {sum(r['reward'] for r in rows)}/{len(rows)}")

    return json.dumps(results)


def run_eval():
    """Local drivers should call the CLI (`modal run -m thomas.eval_modal`)
    and build their BaselineResult from the JSON it writes via
    ``thomas.baseline_from_outputs`` — keeps this module free of Task
    plumbing."""
    raise NotImplementedError("use the CLI entrypoint")


# --- CLI ---------------------------------------------------------------------
# modal run -m thomas.eval_modal --model Qwen/Qwen3-4B-Instruct-2507 \
#     [--adapter /root/thomas-grpo] [--k 4]

@_eval_app.local_entrypoint()
def cli(
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507",
    adapter: str = "",
    k: int = 4,
    max_tokens: int = 512,
    temperature: float = 1.0,
    domain: str = "tutor",
    out: str = "",
):
    if domain == "tutor":
        from .domains.tutor import score, render, case_id, TUTOR_SYSTEM_PROMPT
        from .domains.tutor_scenarios import CASES
        fn_paths = ("thomas.domains.tutor.score",
                    "thomas.domains.tutor.render",
                    "thomas.domains.tutor.case_id")
        system_prompt, cases = TUTOR_SYSTEM_PROMPT, CASES
    else:
        raise ValueError(f"Unknown domain: {domain}")

    result_json = _eval_remote.remote(
        cases_json=json.dumps([vars(c) for c in cases]),
        score_fn_path=fn_paths[0],
        render_fn_path=fn_paths[1],
        case_id_fn_path=fn_paths[2],
        system_prompt=system_prompt,
        model_path=model_path,
        adapter_path=adapter or None,
        k=k, max_tokens=max_tokens, temperature=temperature,
    )
    results = json.loads(result_json)
    print(f"\n=== EVAL {model_path}{' +adapter ' + adapter if adapter else ''} (k={k}) ===")
    for cid, rows in results.items():
        print(f"  {cid}: pass {sum(r['reward'] for r in rows)}/{len(rows)}"
              f"  sample0: {rows[0]['text'][:80]!r}")
    total = sum(r["reward"] for rows in results.values() for r in rows)
    n = sum(len(rows) for rows in results.values())
    print(f"  overall pass rate: {total}/{n} = {total/n:.0%}")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {out}")
