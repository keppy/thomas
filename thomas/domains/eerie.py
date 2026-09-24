"""Eerie literary domain for thomas.

The pretrain-path domain: passages from the K3-mini register corpus
(weird & eerie Gutenberg books). The score is n-gram overlap (mechanically
checkable, judge-free, the anti-mode-collapse design from
RL_STRATEGY.md §1).

This domain is the base for an eventual alien-personality model: the
register is installed at pretrain time, and RL later amplifies the
deliberation/disposition on top of it (RL_STRATEGY.md §2).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Task(str, Enum):
    continuation = "continuation"
    restoration = "restoration"
    infilling = "infilling"


@dataclass(frozen=True)
class EerieCase:
    """One passage from the register, one task, one target."""
    id: str
    task: Task
    prompt: str
    target: str
    source_title: str
    source_author: str


SYSTEM_PROMPT = (
    "You are a literary model fluent in the weird and eerie register — "
    "prose in the tradition of Poe, Machen, Blackwood, Dunsany, and the "
    "early weird tale. Continue, restore, or fill the text in that voice. "
    "Do not explain or comment. Produce only the passage."
)


# --- n-gram score (mirrors eerie_rl/case.py) ---

def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]

def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def _word_set(text: str) -> set[str]:
    return {w.strip(".,;:!?'\"()[]—–").lower() for w in _tokenize(text) if w}

def ngram_precision(gen: str, target: str, n: int = 3) -> float:
    g, t = _ngrams(_tokenize(gen), n), _ngrams(_tokenize(target), n)
    return len(g & t) / len(g) if g else 0.0

def ngram_recall(gen: str, target: str, n: int = 3) -> float:
    g, t = _ngrams(_tokenize(gen), n), _ngrams(_tokenize(target), n)
    return len(g & t) / len(t) if t else 0.0

def ngram_f1(gen: str, target: str, n: int = 3) -> float:
    p, r = ngram_precision(gen, target, n), ngram_recall(gen, target, n)
    return 2 * p * r / (p + r) if p + r else 0.0

def word_overlap(gen: str, target: str) -> float:
    g, t = _word_set(gen), _word_set(target)
    return len(g & t) / len(g | t) if g and t else 0.0


def score(case: EerieCase, text: str) -> tuple[float, dict[str, Any]]:
    """(reward, detail) — the single scoring path.

    Blends sequence match (n-gram) with diction match (word overlap), 50/50.
    Mechanically checkable, judge-free (RL_STRATEGY.md §1).
    """
    seq_p = ngram_precision(text, case.target)
    seq_r = ngram_recall(text, case.target)
    seq_f = ngram_f1(text, case.target)
    word = word_overlap(text, case.target)

    if case.task == Task.continuation:
        seq = seq_p
    elif case.task == Task.restoration:
        seq = seq_r
    else:
        seq = seq_f

    reward = 0.5 * seq + 0.5 * word
    return reward, {
        "task": case.task.value,
        "precision": seq_p,
        "recall": seq_r,
        "f1": seq_f,
        "word_overlap": word,
        "reward": reward,
    }


def oracle(case: EerieCase) -> str:
    """The target itself — scores 1.0 by construction."""
    return case.target


def render(case: EerieCase) -> list[dict[str, str]]:
    return [{"role": "user", "content": case.prompt}]


def case_id(case: EerieCase) -> str:
    return case.id


# --- Case generation from the corpus ---

def _extract_passages(text: str, min_words: int = 80, max_words: int = 200) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)
             if p.strip() and len(p.strip().split()) >= 10]
    passages: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for para in paras:
        w = len(para.split())
        if buf_words + w <= max_words:
            buf.append(para)
            buf_words += w
        else:
            if buf and buf_words >= min_words:
                passages.append(" ".join(buf))
            buf = [para]
            buf_words = w
    if buf and buf_words >= min_words:
        passages.append(" ".join(buf))
    return passages


def make_continuation(passage: str, cid: str, title: str, author: str) -> EerieCase:
    tokens = passage.split()
    split = max(int(len(tokens) * 0.6), 20)
    return EerieCase(
        id=cid, task=Task.continuation,
        prompt=" ".join(tokens[:split]),
        target=" ".join(tokens[split:]),
        source_title=title, source_author=author,
    )


def make_infilling(passage: str, cid: str, title: str, author: str) -> EerieCase:
    tokens = passage.split()
    n = len(tokens)
    return EerieCase(
        id=cid, task=Task.infilling,
        prompt=f"{' '.join(tokens[:n//3])} [FILL] {' '.join(tokens[2*n//3:])}",
        target=" ".join(tokens[n//3:2*n//3]),
        source_title=title, source_author=author,
    )


def make_restoration(passage: str, cid: str, title: str, author: str) -> EerieCase:
    return EerieCase(
        id=cid, task=Task.restoration,
        prompt=passage, target=passage,
        source_title=title, source_author=author,
    )


def load_cases(
    manifest_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
    n_cases: int = 18,
    seed: int = 1899,
) -> list[EerieCase]:
    """Build cases from the K3-mini register corpus.

    Falls back to a small synthetic passage set if the corpus is not found.
    """
    if manifest_path is None:
        manifest_path = Path.home() / "git/modded-nanogpt/data/manifest.jsonl"
    if raw_dir is None:
        raw_dir = Path.home() / "git/modded-nanogpt/data/gutenberg/raw"

    manifest_path, raw_dir = Path(manifest_path), Path(raw_dir)

    if not manifest_path.exists():
        return _synthetic_fallback(n_cases)

    entries = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    register = [e for e in entries if e.get("slice") == "register"]
    if not register:
        return _synthetic_fallback(n_cases)

    rng = random.Random(seed)
    rng.shuffle(register)

    cases: list[EerieCase] = []
    per_book: dict[int, int] = {}
    task_fns = [make_continuation, make_infilling, make_restoration]
    task_weights = [0.5, 0.25, 0.25]

    for entry in register:
        if len(cases) >= n_cases:
            break
        gid = entry["gutenberg_id"]
        if per_book.get(gid, 0) >= 2:
            continue
        raw_path = raw_dir / f"{gid}.txt"
        if not raw_path.exists():
            continue
        text = raw_path.read_text(encoding="utf-8", errors="replace")
        passages = _extract_passages(text)
        rng.shuffle(passages)
        for passage in passages:
            if len(cases) >= n_cases:
                break
            if per_book.get(gid, 0) >= 2:
                break
            fn = rng.choices(task_fns, weights=task_weights)[0]
            cid = f"{gid}-{len(cases):03d}"
            cases.append(fn(passage, cid, entry.get("title", "?"), entry.get("authors", ["?"])[0]))
            per_book[gid] = per_book.get(gid, 0) + 1

    return cases[:n_cases] if cases else _synthetic_fallback(n_cases)


def _synthetic_fallback(n: int) -> list[EerieCase]:
    passages = [
        ("The house had been empty for years, and in the gathering dusk its "
         "windows stared like blind eyes across the moor. Something had "
         "happened there, something that the villagers still spoke of only "
         "in whispers, and never after dark."),
        ("In the hollow beneath the hill, where the ancient oaks bent "
         "toward the earth as though listening, there was a silence that "
         "was not silence. It pressed against the ears, heavy and "
         "expectant, as if the very air remembered what had transpired."),
    ]
    cases: list[EerieCase] = []
    for i, p in enumerate(passages * (n // len(passages) + 1)):
        if len(cases) >= n:
            break
        cases.append(make_continuation(p, f"synth-{i:03d}", "Synthetic", "thomas"))
    return cases[:n]


def cases_to_json(cases: list[EerieCase]) -> str:
    return json.dumps([
        {"id": c.id, "task": c.task.value, "prompt": c.prompt,
         "target": c.target, "source_title": c.source_title,
         "source_author": c.source_author}
        for c in cases
    ])


def cases_from_json(s: str) -> list[EerieCase]:
    raw = json.loads(s)
    return [EerieCase(
        id=c["id"], task=Task(c["task"]), prompt=c["prompt"],
        target=c["target"], source_title=c["source_title"],
        source_author=c["source_author"],
    ) for c in raw]
