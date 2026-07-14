"""Validate phase-progress labels produced by LIBERO semantic traces.

The check is intentionally independent of rollout collection. It verifies the
per-frame labels for every episode before they are used for training or shown
in a demo video.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NUM_PHASES = 4
SUCCESS_PHASE = 3
REQUIRED_COLUMNS = {
    "episode_index",
    "frame_index",
    "phase",
    "phase_progress",
    "global_progress",
    "is_success",
}


@dataclass(frozen=True)
class ValidationReport:
    """Summary of phase-progress validation for one dataset."""

    episodes: int
    successes: int
    frames: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether every checked label follows the phase-progress rule."""
        return not self.failures


def infer_annotation_name(dataset_path: Path) -> str:
    """Infer the semantic trace label name from a ``taskN`` dataset folder."""
    match = re.fullmatch(r"task(\d+)", dataset_path.name)
    if match:
        return f"phase_progress_semantic_trace_task{match.group(1)}"
    return "phase_progress_semantic"


def _completed_phase_progress(length: int) -> np.ndarray:
    """Return the required linear progress for a completed phase."""
    if length == 1:
        return np.ones(1, dtype=np.float32)
    return np.linspace(0.0, 1.0, length, dtype=np.float32)


def _failed_terminal_progress(length: int) -> np.ndarray:
    """Return the required ramp-and-cap progress for a failed terminal phase."""
    if length == 1:
        return np.zeros(1, dtype=np.float32)
    ramp_length = max(2, (length + 1) // 2)
    progress = np.full(length, 0.5, dtype=np.float32)
    progress[:ramp_length] = np.linspace(
        0.0, 0.5, ramp_length, dtype=np.float32
    )
    return progress


def validate_phase_progress_labels(labels: pd.DataFrame) -> ValidationReport:
    """Validate all successful and failed episode labels.

    Successful episodes must linearly complete every observed phase, including
    phase zero. Failed episodes must complete previous phases and ramp only the
    final observed phase to 0.5 before remaining there.
    """
    missing = REQUIRED_COLUMNS - set(labels.columns)
    if missing:
        raise ValueError(f"Phase label file missing columns: {sorted(missing)}")

    labels = labels.sort_values(["episode_index", "frame_index"])
    failures: list[str] = []
    successes = 0

    for episode_index, episode in labels.groupby("episode_index", sort=True):
        phase = episode["phase"].to_numpy(dtype=np.int64)
        progress = episode["phase_progress"].to_numpy(dtype=np.float32)
        global_progress = episode["global_progress"].to_numpy(dtype=np.float32)
        success_values = episode["is_success"].to_numpy(dtype=bool)

        if len(phase) == 0:
            failures.append(f"ep{episode_index}: no frames")
            continue
        if not np.all(success_values == success_values[-1]):
            failures.append(f"ep{episode_index}: is_success changes within episode")
            continue
        if not np.isfinite(progress).all() or not np.isfinite(global_progress).all():
            failures.append(f"ep{episode_index}: non-finite progress value")
            continue
        if np.any(np.diff(phase) < 0):
            failures.append(f"ep{episode_index}: phase is not monotonic")
            continue

        is_success = bool(success_values[-1])
        successes += int(is_success)
        terminal_phase = int(phase[-1])
        expected_global = (phase.astype(np.float32) + progress) / NUM_PHASES
        if not np.allclose(global_progress, expected_global, atol=1e-6):
            failures.append(f"ep{episode_index}: global_progress formula mismatch")

        for phase_id in np.unique(phase):
            observed = progress[phase == phase_id]
            expected = (
                _failed_terminal_progress(len(observed))
                if not is_success and phase_id == terminal_phase
                else _completed_phase_progress(len(observed))
            )
            if not np.allclose(observed, expected, atol=1e-6):
                failures.append(
                    f"ep{episode_index}: phase {phase_id} does not follow "
                    "the expected progress schedule"
                )

        if is_success and (
            terminal_phase != SUCCESS_PHASE
            or not np.isclose(global_progress[-1], 1.0, atol=1e-6)
        ):
            failures.append(f"ep{episode_index}: success does not end at 1.0")
        if np.any(np.diff(global_progress) < -1e-6):
            failures.append(f"ep{episode_index}: global progress is not monotonic")

    return ValidationReport(
        episodes=int(labels["episode_index"].nunique()),
        successes=successes,
        frames=len(labels),
        failures=tuple(failures),
    )


def print_validation_report(dataset_name: str, report: ValidationReport) -> None:
    """Print a compact, human-readable validation result."""
    print(
        f"{dataset_name}: episodes={report.episodes}, successes={report.successes}, "
        f"failures={report.episodes - report.successes}, frames={report.frames}, "
        f"checks={'PASSED' if report.passed else 'FAILED'}"
    )
    for failure in report.failures:
        print(failure)


def main() -> None:
    """Run validation for one task dataset from the command line."""
    parser = argparse.ArgumentParser(description="Validate phase-progress labels.")
    parser.add_argument("--dataset_path", required=True, help="Task dataset root.")
    parser.add_argument(
        "--annotation_name",
        default=None,
        help="Label parquet basename under meta/; inferred for a taskN directory.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    annotation_name = args.annotation_name or infer_annotation_name(dataset_path)
    labels_path = dataset_path / "meta" / f"{annotation_name}.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Phase labels not found: {labels_path}")

    report = validate_phase_progress_labels(pd.read_parquet(labels_path))
    print_validation_report(dataset_path.name, report)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
