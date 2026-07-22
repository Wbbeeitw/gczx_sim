"""Build fixed-size per-task policy pools from rollouts and expert episodes.

All successful rollout episodes are retained. Expert episodes replace the
lowest-progress rollout failures, so every output task dataset has exactly the
requested number of episodes and remains directly usable by later ReCap stages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

if __package__:
    from examples.recap.process.merge_lerobot_multitask_datasets import (
        merge_multitask_datasets,
    )
else:
    from merge_lerobot_multitask_datasets import merge_multitask_datasets


@dataclass(frozen=True)
class EpisodeCandidate:
    """One episode available for a fixed policy-training pool."""

    dataset_path: Path
    position: int
    episode_index: int
    source_type: str
    is_success: bool
    max_phase: int
    max_global_progress: float
    max_phase_progress: float

    @property
    def score(self) -> tuple[int, float, float, int]:
        return (
            self.max_phase,
            self.max_global_progress,
            self.max_phase_progress,
            -self.position,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _parse_expert(value: str) -> tuple[str, Path, tuple[int, int] | None]:
    if "=" not in value:
        raise ValueError(f"Expected TASK=PATH[::START:END], got {value!r}")
    task, raw_input = value.split("=", 1)
    if "::" not in raw_input:
        return task, Path(raw_input), None
    raw_path, raw_range = raw_input.rsplit("::", 1)
    try:
        raw_start, raw_end = raw_range.split(":", 1)
        start, end = int(raw_start), int(raw_end)
    except ValueError as exc:
        raise ValueError(
            f"Invalid expert slice {value!r}; expected TASK=PATH::START:END"
        ) from exc
    if start < 0 or end <= start:
        raise ValueError(f"Invalid expert slice {value!r}")
    return task, Path(raw_path), (start, end)


def _phase_path(dataset_path: Path, task: str) -> Path:
    preferred = (
        dataset_path
        / "meta"
        / f"phase_progress_semantic_trace_{task}.parquet"
    )
    if preferred.exists():
        return preferred
    multitask = dataset_path / "meta" / "phase_progress_multitask.parquet"
    if multitask.exists():
        return multitask
    candidates = sorted(
        path
        for path in (dataset_path / "meta").glob("phase_progress_*.parquet")
        if path.name != "phase_progress_multitask.parquet"
    )
    if len(candidates) != 1:
        raise ValueError(
            f"{dataset_path}: cannot identify phase labels for {task}; "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


def _episode_candidates(
    dataset_path: Path,
    task: str,
    source_type: str,
    episode_range: tuple[int, int] | None = None,
) -> list[EpisodeCandidate]:
    episodes = _read_jsonl(dataset_path / "meta" / "episodes.jsonl")
    start, end = episode_range or (0, len(episodes))
    if end > len(episodes):
        raise ValueError(
            f"{dataset_path}: slice {start}:{end} exceeds "
            f"total episodes {len(episodes)}"
        )
    selected = list(enumerate(episodes))[start:end]
    selected_indices = {int(record["episode_index"]) for _, record in selected}
    phase_path = _phase_path(dataset_path, task)
    labels = pd.read_parquet(
        phase_path,
        columns=[
            "episode_index",
            "phase",
            "phase_progress",
            "global_progress",
            "is_success",
        ],
    )
    labels = labels[
        labels["episode_index"].astype("int64").isin(selected_indices)
    ].copy()
    stats = labels.groupby("episode_index", sort=False).agg(
        is_success=("is_success", "last"),
        max_phase=("phase", "max"),
        max_global_progress=("global_progress", "max"),
        max_phase_progress=("phase_progress", "max"),
    )
    missing = selected_indices.difference(int(index) for index in stats.index)
    if missing:
        raise ValueError(
            f"{dataset_path}: phase labels missing episodes {sorted(missing)}"
        )
    return [
        EpisodeCandidate(
            dataset_path=dataset_path,
            position=position,
            episode_index=int(record["episode_index"]),
            source_type=source_type,
            is_success=bool(stats.loc[int(record["episode_index"]), "is_success"]),
            max_phase=int(stats.loc[int(record["episode_index"]), "max_phase"]),
            max_global_progress=float(
                stats.loc[int(record["episode_index"]), "max_global_progress"]
            ),
            max_phase_progress=float(
                stats.loc[int(record["episode_index"]), "max_phase_progress"]
            ),
        )
        for position, record in selected
    ]


def _select_task_pool(
    rollout: list[EpisodeCandidate],
    experts: list[EpisodeCandidate],
    episodes_per_task: int,
    require_successful_experts: bool,
) -> tuple[list[EpisodeCandidate], dict[str, Any]]:
    if require_successful_experts:
        failed_experts = [item.episode_index for item in experts if not item.is_success]
        if failed_experts:
            raise ValueError(
                f"expert input contains failed episodes {failed_experts}; "
                "use --allow-failed-expert to keep them intentionally"
            )
    rollout_slots = episodes_per_task - len(experts)
    if rollout_slots < 0:
        raise ValueError(
            f"expert episodes ({len(experts)}) exceed pool size {episodes_per_task}"
        )
    successes = [item for item in rollout if item.is_success]
    failures = [item for item in rollout if not item.is_success]
    if len(successes) > rollout_slots:
        raise ValueError(
            f"cannot preserve {len(successes)} rollout successes with "
            f"{len(experts)} experts in a {episodes_per_task}-episode pool"
        )
    failures_needed = rollout_slots - len(successes)
    if failures_needed > len(failures):
        raise ValueError(
            f"need {failures_needed} rollout failures, only {len(failures)} available"
        )
    kept_failures = sorted(failures, key=lambda item: item.score, reverse=True)[
        :failures_needed
    ]
    selected_rollout = sorted(
        [*successes, *kept_failures], key=lambda item: item.position
    )
    selected = [*selected_rollout, *experts]
    report = {
        "episodes": len(selected),
        "rollout_available": len(rollout),
        "rollout_successes_preserved": len(successes),
        "rollout_failures_kept": len(kept_failures),
        "rollout_failures_dropped": len(failures) - len(kept_failures),
        "expert_episodes": len(experts),
    }
    return selected, report


def _select_successful_experts(
    candidates: list[EpisodeCandidate],
    *,
    task: str,
    dataset_path: Path,
    count: int,
) -> list[EpisodeCandidate]:
    successful = [candidate for candidate in candidates if candidate.is_success]
    if len(successful) < count:
        raise ValueError(
            f"{task}: requested {count} successful experts, but only found "
            f"{len(successful)} in {dataset_path}"
        )
    return successful[:count]


def _write_pool_provenance(
    output_path: Path,
    task: str,
    selected: list[EpisodeCandidate],
) -> tuple[Path, Path, list[int]]:
    provenance_path = output_path / "meta" / "episode_provenance.jsonl"
    with provenance_path.open("w", encoding="utf-8") as file:
        for output_index, item in enumerate(selected):
            record = {
                "episode_index": output_index,
                "task": task,
                "source_type": item.source_type,
                "source_dataset": str(item.dataset_path),
                "source_episode_position": item.position,
                "source_episode_index": item.episode_index,
                "is_success": item.is_success,
                "max_phase": item.max_phase,
                "max_global_progress": item.max_global_progress,
                "max_phase_progress": item.max_phase_progress,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    full_positive_episodes = [
        output_index
        for output_index, item in enumerate(selected)
        if item.source_type == "expert"
    ]
    full_positive_path = output_path / "meta" / "full_positive_episodes.json"
    full_positive_path.write_text(
        json.dumps(full_positive_episodes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return provenance_path, full_positive_path, full_positive_episodes


def build_fixed_pools(
    *,
    tasks: list[str],
    rollout_pattern: str,
    output_pattern: str,
    episodes_per_task: int,
    expert_inputs: dict[str, list[tuple[Path, tuple[int, int] | None]]],
    expert_pattern: str | None,
    experts_per_task: int | None,
    overwrite: bool,
    require_successful_experts: bool,
) -> dict[str, Any]:
    """Materialize one fixed-size LeRobot dataset per task."""
    result: dict[str, Any] = {"episodes_per_task": episodes_per_task, "tasks": {}}
    prepared: list[
        tuple[str, Path, Path, list[EpisodeCandidate], dict[str, Any]]
    ] = []
    for task in tasks:
        rollout_path = Path(rollout_pattern.format(task=task))
        output_path = Path(output_pattern.format(task=task))
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_path}; "
                "use --overwrite to replace it"
            )
        rollout = _episode_candidates(rollout_path, task, "rollout")
        experts = [
            candidate
            for path, episode_range in expert_inputs.get(task, [])
            for candidate in _episode_candidates(
                path, task, "expert", episode_range
            )
        ]
        if expert_pattern is not None:
            if experts:
                raise ValueError(
                    f"{task}: --expert-pattern cannot be combined with "
                    "explicit --expert inputs"
                )
            expert_path = Path(expert_pattern.format(task=task))
            available_experts = _episode_candidates(
                expert_path,
                task,
                "expert",
            )
            requested_experts = int(experts_per_task or 0)
            experts = _select_successful_experts(
                available_experts,
                task=task,
                dataset_path=expert_path,
                count=requested_experts,
            )
        selected, task_report = _select_task_pool(
            rollout,
            experts,
            episodes_per_task,
            require_successful_experts,
        )
        prepared.append((task, rollout_path, output_path, selected, task_report))

    for task, rollout_path, output_path, selected, task_report in prepared:
        merge_multitask_datasets(
            [item.dataset_path for item in selected],
            output_path,
            overwrite=overwrite,
            episode_ranges=[(item.position, item.position + 1) for item in selected],
        )
        provenance_path, full_positive_path, full_positive_episodes = (
            _write_pool_provenance(output_path, task, selected)
        )
        task_report.update(
            {
                "rollout_dataset": str(rollout_path),
                "output_dataset": str(output_path),
                "provenance_path": str(provenance_path),
                "full_positive_episodes_path": str(full_positive_path),
                "full_positive_episodes": full_positive_episodes,
            }
        )
        result["tasks"][task] = task_report
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--rollout-pattern", required=True)
    parser.add_argument("--output-pattern", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--expert", action="append", default=[])
    parser.add_argument("--expert-pattern")
    parser.add_argument("--experts-per-task", type=int)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-failed-expert", action="store_true")
    args = parser.parse_args()

    expert_inputs: dict[str, list[tuple[Path, tuple[int, int] | None]]] = {}
    for value in args.expert:
        task, path, episode_range = _parse_expert(value)
        if task not in args.tasks:
            raise ValueError(f"expert task {task!r} is not listed in --tasks")
        expert_inputs.setdefault(task, []).append((path, episode_range))
    if args.expert_pattern is not None:
        if args.experts_per_task is None or args.experts_per_task <= 0:
            raise ValueError(
                "--expert-pattern requires --experts-per-task to be positive"
            )
        if args.expert:
            raise ValueError(
                "--expert-pattern cannot be combined with explicit --expert inputs"
            )
    elif args.experts_per_task is not None:
        raise ValueError("--experts-per-task requires --expert-pattern")
    result = build_fixed_pools(
        tasks=args.tasks,
        rollout_pattern=args.rollout_pattern,
        output_pattern=args.output_pattern,
        episodes_per_task=args.episodes_per_task,
        expert_inputs=expert_inputs,
        expert_pattern=args.expert_pattern,
        experts_per_task=args.experts_per_task,
        overwrite=args.overwrite,
        require_successful_experts=not args.allow_failed_expert,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
