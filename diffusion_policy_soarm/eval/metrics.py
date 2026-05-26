"""Evaluation metrics: success rate, failure breakdown, mode-balance score.

Phase 8 implementation target:
- Success rate per tier.
- Failure category distribution (first-cause priority breakdown).
- Mode-balance score: |P(left) - 0.5| among diffusion successes — lower is
  better, 0 means perfect 50/50 split matching the training data.
- Summary tables suitable for inclusion in the writeup.
"""

from __future__ import annotations

from pathlib import Path

from diffusion_policy_soarm.eval.harness import TrialResult


class EvalMetrics:
    """Aggregate and format evaluation metrics.

    Phase 8 implementation target.
    """

    def __init__(self, results: list[TrialResult]) -> None:
        raise NotImplementedError("Phase 8")

    def success_rate(self, tier: str) -> float:
        """Fraction of successful trials for the given tier."""
        raise NotImplementedError("Phase 8")

    def failure_breakdown(self, tier: str) -> dict[str, float]:
        """Per-category failure fractions for the given tier."""
        raise NotImplementedError("Phase 8")

    def mode_balance_score(self) -> float:
        """Absolute deviation from 50/50 cube-choice among successes."""
        raise NotImplementedError("Phase 8")

    def to_markdown(self) -> str:
        """Render all metrics as a Markdown table."""
        raise NotImplementedError("Phase 8")

    def save(self, output_dir: Path) -> None:
        """Write metrics JSON and Markdown table to *output_dir*."""
        raise NotImplementedError("Phase 8")
