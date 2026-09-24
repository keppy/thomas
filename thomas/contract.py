"""The thomas contract surface, versioned. See docs/CONTRACT.md.

Stdlib only, so readers (a gonogo eval script, the Hermes plugin, a serving
process) can check an artifact without importing torch or modal.

Versioning rule: adding an optional field or file keeps the version;
removing, renaming, or changing the meaning of anything bumps it and gets a
CHANGELOG entry. Readers treat a missing ``contract_version`` as 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACT_VERSION = 1

# Files every encoder artifact dir carries, beyond HF's save_pretrained output.
ARTIFACT_SIDECARS = ("label2id.json", "temperature.json", "metrics.json")

# gonogo versions this release of thomas is tested against (see pyproject).
GONOGO_RANGE = ">=0.2,<0.3"


def read_artifact(model_dir: str | Path) -> dict[str, Any]:
    """Validate an encoder artifact dir and return its contract fields.

    Returns ``{"contract_version", "label2id", "temperature", "metrics"}``.
    Raises ``ValueError`` with the specific problem if the dir does not meet
    the contract, including a ``contract_version`` newer than this reader.
    """
    d = Path(model_dir)
    missing = [f for f in ("config.json", *ARTIFACT_SIDECARS) if not (d / f).is_file()]
    if missing:
        raise ValueError(f"{d} is not a thomas encoder artifact (missing {', '.join(missing)})")
    label2id = json.loads((d / "label2id.json").read_text(encoding="utf-8"))
    temperature = float(json.loads((d / "temperature.json").read_text(encoding="utf-8"))["temperature"])
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    version = int(metrics.get("contract_version", 1))
    if version > CONTRACT_VERSION:
        raise ValueError(f"{d} has contract_version {version}; this thomas reads <= {CONTRACT_VERSION}")
    if not temperature > 0:
        raise ValueError(f"{d}: temperature must be > 0, got {temperature}")
    ids = sorted(label2id.values())
    if ids != list(range(len(ids))):
        raise ValueError(f"{d}: label2id ids must be 0..{len(ids) - 1} with no gaps")
    return {"contract_version": version, "label2id": label2id,
            "temperature": temperature, "metrics": metrics}
