"""Modal vLLM serve — serve a pretrained checkpoint on Modal via vLLM.

This is a Modal app that loads a checkpoint from the runs volume and
serves it via vLLM in OpenAI-compatible mode. Deploy with::

    modal deploy serve_vllm.py

or invoke from thomas::

    from thomas.serve_vllm import serve
    endpoint = serve(checkpoint_path="/root/modded-nanogpt/runs/<run_id>")

The endpoint is OpenAI-compatible: ``POST /v1/chat/completions`` with
``model="pretrained"``.

Based on the Modal vLLM inference example
(https://modal.com/docs/examples/vllm_inference).
"""

from __future__ import annotations

import os

import modal

app = modal.App("thomas-vllm-serve")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_LOG_STATS_INTERVAL": "1",
    })
)

# Volumes — the runs volume has the checkpoints, the HF cache caches weights.
# Adjust volume names to match the nanogpt repo's Modal setup.
runs_vol = modal.Volume.from_name("k3mini-runs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

VLLM_PORT = 8000
N_GPU = 1


@app.server(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * 60,
    startup_timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/modded-nanogpt/runs": runs_vol,
    },
    port=VLLM_PORT,
    unauthenticated=True,
)
class VLLMServer:
    """Serve a pretrained checkpoint via vLLM in OpenAI-compatible mode.

    Set ``CHECKPOINT_PATH`` env var to the checkpoint directory on the
    runs volume. Defaults to the latest run's final checkpoint.
    """

    @modal.enter()
    def start(self):
        import os
        import subprocess

        ckpt = os.environ.get(
            "CHECKPOINT_PATH",
            "/root/modded-nanogpt/runs/latest/final",
        )
        print(f"Loading checkpoint: {ckpt}")

        cmd = [
            "vllm", "serve", ckpt,
            "--served-model-name", "pretrained",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--uvicorn-log-level", "info",
            "--enforce-eager",  # faster startup for short-lived serve
        ]
        print(f"vLLM command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


def serve(checkpoint_path: str | None = None) -> str:
    """Deploy the vLLM server on Modal and return the endpoint URL.

    Args:
        checkpoint_path: path to the checkpoint on the runs volume.
            If None, uses the default (latest/final).

    Returns:
        The OpenAI-compatible endpoint URL.
    """
    import os
    import subprocess

    if checkpoint_path:
        os.environ["CHECKPOINT_PATH"] = checkpoint_path

    # Deploy the app — this creates a persistent endpoint
    result = subprocess.run(
        ["modal", "deploy", "--app", "thomas-vllm-serve", __file__],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"Deploy failed: {result.stderr[:500]}")
    else:
        print(f"Deployed: {result.stdout[:500]}")

    # The endpoint URL follows Modal's convention:
    # https://<workspace>--<app>.modal.run
    workspace = os.environ.get("MODAL_WORKSPACE", "<workspace>")
    return f"https://{workspace}--thomas-vllm-serve.modal.run"


if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    endpoint = serve(ckpt)
    print(f"\nvLLM endpoint: {endpoint}")
    print(f"\nTest with:")
    print(f"  curl {endpoint}/v1/chat/completions \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"model\": \"pretrained\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello\"}}]}}'")
