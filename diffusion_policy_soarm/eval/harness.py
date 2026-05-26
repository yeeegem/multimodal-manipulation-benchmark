"""Evaluation harness: three-tier rollout runner and failure logger.

Phase 8 implementation target:
- Tier A: cube positions from the training distribution.
- Tier B: cube positions shifted outside training range.
- Tier C: training positions + distractor objects.
- For each trial: record binary success, failure category (first-cause priority),
  and — for diffusion successes — which cube was chosen (left/right).
- Layouts are randomly sampled within each tier definition.
- Output raw results CSV and summary tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FailureCategory(str, Enum):
    GRABBED_NOTHING = "grabbed_nothing"
    GRASPED_WRONG_OBJECT = "grasped_wrong_object"
    GRASP_SLIP = "grasp_slip"
    MISSED_CUP = "missed_cup"
    KNOCKED_CUP_OVER = "knocked_cup_over"
    COLLISION_UNSAFE = "collision_unsafe"
    FROZE_NO_ATTEMPT = "froze_no_attempt"
    TIMEOUT = "timeout"


@dataclass
class TrialResult:
    """Result record for one evaluation rollout."""

    tier: str
    trial_idx: int
    success: bool
    failure_category: FailureCategory | None = None
    # "left" or "right" for diffusion successes, None otherwise.
    cube_chosen: str | None = None
    notes: str = ""


class EvalHarness:
    """Runs tiered evaluation trials and accumulates results.

    Phase 8 implementation target.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Phase 8")

    def run_tier(self, tier: str, num_trials: int) -> list[TrialResult]:
        """Execute *num_trials* rollouts for the given tier."""
        raise NotImplementedError("Phase 8")

    def save_results(self, results: list[TrialResult], output_dir: Path) -> None:
        """Write results CSV and summary table to *output_dir*."""
        raise NotImplementedError("Phase 8")
