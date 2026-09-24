"""Compare two baseline runs (before vs after training).

If gonogo is installed, uses ``gonogo.compare`` for the paired McNemar
test. Otherwise, reports the raw pass-rate difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .baseline import BaselineResult


@dataclass
class Comparison:
    """Before/after comparison of two baseline runs."""

    task_name: str
    before: BaselineResult
    after: BaselineResult
    report: Any = None  # gonogo.Comparison or None

    def summary(self) -> str:
        lines = [
            f"Task: {self.task_name}",
            f"Before: {self.before.pass_rate:.1%} ({sum(1 for r in self.before.per_case if r['reward'] >= 0.99)}/{len(self.before.per_case)})",
            f"After:  {self.after.pass_rate:.1%} ({sum(1 for r in self.after.per_case if r['reward'] >= 0.99)}/{len(self.after.per_case)})",
        ]
        if self.report is not None and hasattr(self.report, "summary"):
            lines.append("")
            lines.append(self.report.summary())
        return "\n".join(lines)


def compare(before: BaselineResult, after: BaselineResult) -> Comparison:
    """Compare two baseline runs.

    If both have gonogo reports, uses ``gonogo.compare`` for the paired
    McNemar test on shared case ids. Otherwise, reports the raw
    pass-rate difference.
    """
    report = None
    if before.report is not None and after.report is not None:
        try:
            from gonogo import compare as gcompare

            report = gcompare(before.report, after.report)
        except Exception:
            pass  # fall through to raw comparison

    return Comparison(
        task_name=before.task_name,
        before=before,
        after=after,
        report=report,
    )
