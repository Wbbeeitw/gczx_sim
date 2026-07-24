"""Create a paper-ready raw-vs-fused critic trajectory visualization.

The figure combines simulator keyframes, target/raw/fused return trajectories,
semantic phase boundaries, and an optional absolute-error panel. It also writes
the aligned frame data and plotting metadata for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch
from matplotlib.text import Text


TARGET_COLOR = "#263238"
RAW_COLOR = "#E76F51"
FUSED_COLOR = "#2364AA"
BOUNDARY_COLOR = "#415A77"
PHASE_COLORS = ("#DCEAF7", "#E8F3E8", "#FFF1D6", "#F7E1EA")
SUCCESS_COLOR = "#2A9D8F"
FAILURE_COLOR = "#C44536"


@dataclass(frozen=True)
class ComparisonPaths:
    """Resolved return-comparison inputs and value scales."""

    advantages: Path
    predictions: Path
    return_min: float
    return_max: float
    value_min: float
    value_max: float


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidate = base_dir / path
    return candidate if candidate.exists() else path


def _resolve_comparison_inputs(args: argparse.Namespace) -> ComparisonPaths:
    report: dict[str, Any] = {}
    report_dir = Path.cwd()
    if args.comparison is not None:
        comparison_path = Path(args.comparison).expanduser()
        if not comparison_path.is_file():
            raise FileNotFoundError(f"Comparison report not found: {comparison_path}")
        with comparison_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        report_dir = comparison_path.parent

    advantages_raw = args.advantages or report.get("advantages_path")
    predictions_raw = args.predictions or report.get("predictions_path")
    if not advantages_raw or not predictions_raw:
        raise ValueError(
            "Provide --comparison, or provide both --advantages and --predictions."
        )

    scales = {
        "return_min": args.return_min,
        "return_max": args.return_max,
        "value_min": args.value_min,
        "value_max": args.value_max,
    }
    defaults = {
        "return_min": -900.0,
        "return_max": 0.0,
        "value_min": -1.0,
        "value_max": 0.0,
    }
    for key, value in scales.items():
        if value is None:
            scales[key] = float(report.get(key, defaults[key]))

    paths = ComparisonPaths(
        advantages=_resolve_path(str(advantages_raw), report_dir),
        predictions=_resolve_path(str(predictions_raw), report_dir),
        return_min=float(scales["return_min"]),
        return_max=float(scales["return_max"]),
        value_min=float(scales["value_min"]),
        value_max=float(scales["value_max"]),
    )
    for name, path in (
        ("advantages", paths.advantages),
        ("predictions", paths.predictions),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name.title()} parquet not found: {path}")
    return paths


def _load_episode_returns(
    paths: ComparisonPaths,
    episode_index: int,
    smooth_window: int,
) -> pd.DataFrame:
    from rlinf.revalue.value_scale import map_values_to_return_scale

    advantages = pd.read_parquet(paths.advantages)
    predictions = pd.read_parquet(paths.predictions)
    required_advantages = {
        "episode_index",
        "frame_index",
        "return",
        "value_current",
    }
    required_predictions = {"episode_index", "frame_index", "value_fused"}
    missing_advantages = required_advantages - set(advantages.columns)
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_advantages:
        raise ValueError(
            f"Advantages parquet is missing columns: {sorted(missing_advantages)}"
        )
    if missing_predictions:
        raise ValueError(
            f"Predictions parquet is missing columns: {sorted(missing_predictions)}"
        )

    advantages = advantages[advantages["episode_index"] == episode_index]
    predictions = predictions[predictions["episode_index"] == episode_index]
    prediction_columns = ["episode_index", "frame_index", "value_fused"]
    advantage_columns = [
        "episode_index",
        "frame_index",
        "return",
        "value_current",
    ]
    frame = predictions[prediction_columns].merge(
        advantages[advantage_columns],
        on=["episode_index", "frame_index"],
        how="inner",
        validate="one_to_one",
    )
    if frame.empty:
        raise ValueError(
            f"No overlapping critic rows for comparison episode {episode_index}."
        )

    frame = frame.sort_values("frame_index").reset_index(drop=True)
    frame["target_return"] = frame["return"].astype(float)
    frame["raw_critic"] = map_values_to_return_scale(
        frame["value_current"].to_numpy(dtype=np.float64),
        return_min=paths.return_min,
        return_max=paths.return_max,
        value_min=paths.value_min,
        value_max=paths.value_max,
    )
    frame["fused_critic"] = map_values_to_return_scale(
        frame["value_fused"].to_numpy(dtype=np.float64),
        return_min=paths.return_min,
        return_max=paths.return_max,
        value_min=paths.value_min,
        value_max=paths.value_max,
    )
    frame["raw_abs_error"] = (frame["raw_critic"] - frame["target_return"]).abs()
    frame["fused_abs_error"] = (
        frame["fused_critic"] - frame["target_return"]
    ).abs()
    frame["raw_critic_plot"] = frame["raw_critic"].rolling(
        smooth_window, center=True, min_periods=1
    ).mean()
    frame["fused_critic_plot"] = frame["fused_critic"].rolling(
        smooth_window, center=True, min_periods=1
    ).mean()
    return frame


def _discover_phase_path(dataset_path: Path, explicit_path: str | None) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Phase-label parquet not found: {path}")
        return path

    meta_dir = dataset_path / "meta"
    patterns = (
        "phase_progress_semantic_trace_*.parquet",
        "phase_progress_multitask.parquet",
        "*phase_progress*.parquet",
    )
    for pattern in patterns:
        matches = sorted(meta_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _phase_column(frame: pd.DataFrame) -> str:
    for column in ("phase", "phase_index", "semantic_phase"):
        if column in frame.columns:
            return column
    raise ValueError(
        "Phase-label parquet must contain phase, phase_index, or semantic_phase."
    )


def _align_phases(
    trajectory: pd.DataFrame,
    phase_path: Path | None,
    phase_episode: int,
    fallback_episode: int,
) -> tuple[pd.DataFrame, str | None, int | None]:
    trajectory = trajectory.copy()
    trajectory["phase"] = pd.NA
    trajectory["phase_progress"] = np.nan
    trajectory["is_success"] = pd.NA
    if phase_path is None:
        trajectory["is_boundary"] = False
        return trajectory, None, None

    phases = pd.read_parquet(phase_path)
    required = {"episode_index", "frame_index"}
    missing = required - set(phases.columns)
    if missing:
        raise ValueError(f"Phase-label parquet is missing columns: {sorted(missing)}")
    selected_episode = phase_episode
    episode_phases = phases[phases["episode_index"] == selected_episode].copy()
    if episode_phases.empty and fallback_episode != selected_episode:
        selected_episode = fallback_episode
        episode_phases = phases[phases["episode_index"] == selected_episode].copy()
    if episode_phases.empty:
        trajectory["is_boundary"] = False
        return trajectory, str(phase_path), None

    phase_column = _phase_column(episode_phases)
    keep_columns = ["frame_index", phase_column]
    for optional in ("phase_progress", "is_success"):
        if optional in episode_phases.columns:
            keep_columns.append(optional)
    episode_phases = (
        episode_phases[keep_columns]
        .sort_values("frame_index")
        .drop_duplicates("frame_index", keep="last")
        .set_index("frame_index")
    )
    aligned = episode_phases.reindex(trajectory["frame_index"].to_numpy())
    aligned = aligned.ffill().bfill()
    trajectory["phase"] = aligned[phase_column].to_numpy()
    if "phase_progress" in aligned.columns:
        trajectory["phase_progress"] = aligned["phase_progress"].to_numpy()
    if "is_success" in aligned.columns:
        trajectory["is_success"] = aligned["is_success"].to_numpy()
    trajectory["is_boundary"] = trajectory["phase"].ne(
        trajectory["phase"].shift(1)
    )
    trajectory.loc[trajectory.index[0], "is_boundary"] = False
    return trajectory, str(phase_path), selected_episode


def _phase_display_name(phase: Any, phase_names: list[str]) -> str:
    if pd.isna(phase):
        return "Unknown phase"
    try:
        index = int(phase)
    except (TypeError, ValueError):
        return str(phase)
    if 0 <= index < len(phase_names):
        return phase_names[index]
    return f"Phase {index}"


def _choose_keyframes(
    trajectory: pd.DataFrame,
    num_keyframes: int,
    boundary_window: int,
) -> list[int]:
    frames = trajectory["frame_index"].astype(int).tolist()
    if not frames:
        return []
    mandatory = [frames[0]]
    boundaries = trajectory.loc[trajectory["is_boundary"], "frame_index"].astype(int)
    mandatory.extend(boundaries.tolist())
    mandatory.append(frames[-1])
    mandatory = list(dict.fromkeys(mandatory))
    selected = mandatory[:num_keyframes]
    frame_positions = {frame: index for index, frame in enumerate(frames)}
    minimum_margin = max(1, boundary_window)
    selected_positions = sorted(frame_positions[frame] for frame in selected)
    intervals = [
        (left, right)
        for left, right in zip(selected_positions, selected_positions[1:])
        if right - left > 1
    ]
    allocations = [0] * len(intervals)
    for _ in range(max(0, num_keyframes - len(selected))):
        if not intervals:
            break
        interval_index = max(
            range(len(intervals)),
            key=lambda index: (
                (intervals[index][1] - intervals[index][0])
                / (allocations[index] + 1),
                intervals[index][1] - intervals[index][0],
            ),
        )
        allocations[interval_index] += 1

    for (left, right), allocation in zip(intervals, allocations):
        span = right - left
        for slot in range(1, allocation + 1):
            position = int(round(left + span * slot / (allocation + 1)))
            lower = min(left + minimum_margin, right - 1)
            upper = max(right - minimum_margin, left + 1)
            position = min(max(position, lower), upper)
            candidate = frames[position]
            if candidate not in selected:
                selected.append(candidate)
    if len(selected) < num_keyframes:
        for candidate in frames:
            if len(selected) >= num_keyframes:
                break
            if candidate not in selected:
                selected.append(candidate)
    return sorted(selected)


def _to_image_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (
        1,
        3,
        4,
    ):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image)
        if image.max(initial=0.0) > 1.0 or image.min(initial=0.0) < 0.0:
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0.0, 1.0)
    return image


def _detect_image_keys(sample: dict[str, Any]) -> list[str]:
    preferred = (
        "image",
        "front_image",
        "observation.images.image",
        "observation.images.front",
        "wrist_image",
        "left_image",
        "right_image",
    )
    keys = []
    for key, value in sample.items():
        if not (
            key in preferred
            or key.startswith("observation.images.")
            or key.endswith("_image")
        ):
            continue
        try:
            if _to_image_array(value).ndim >= 2:
                keys.append(key)
        except (TypeError, ValueError):
            continue
    return sorted(
        keys, key=lambda key: preferred.index(key) if key in preferred else 99
    )


def _episode_indices(dataset: Any, episode_index: int) -> list[int]:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is not None and episode_index < len(
        episode_data_index["from"]
    ):
        start = int(episode_data_index["from"][episode_index].item())
        end = int(episode_data_index["to"][episode_index].item())
        return list(range(start, end))
    indices = []
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_episode = int(np.asarray(sample["episode_index"]).item())
        if sample_episode == episode_index:
            indices.append(index)
    return indices


def _task_description(dataset_path: Path, sample: dict[str, Any]) -> str | None:
    for key in ("task", "task_name", "language_instruction", "instruction"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())

    task_index = sample.get("task_index")
    if task_index is None:
        return None
    try:
        task_index = int(np.asarray(task_index).item())
    except (TypeError, ValueError):
        return None

    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        return None
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if int(entry.get("task_index", -1)) != task_index:
                continue
            task = entry.get("task") or entry.get("task_name")
            if isinstance(task, str) and task.strip():
                return " ".join(task.split())
    return None


def _load_keyframe_images(
    dataset_path: Path,
    episode_index: int,
    requested_frames: list[int],
    trajectory_frames: list[int],
    image_key: str | None,
) -> tuple[dict[int, np.ndarray], str, str | None]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.datasets.recap.utils import decode_image_struct_batch

    dataset = LeRobotDataset(
        dataset_path.name,
        root=dataset_path,
        delta_timestamps=None,
        download_videos=False,
    )
    dataset.hf_dataset.set_transform(decode_image_struct_batch)
    indices = _episode_indices(dataset, episode_index)
    if not indices:
        raise ValueError(f"Dataset episode {episode_index} has no frames.")

    first_sample = dataset[indices[0]]
    detected_keys = _detect_image_keys(first_sample)
    selected_key = image_key or (detected_keys[0] if detected_keys else None)
    if selected_key is None:
        raise ValueError(
            "No simulator image field was detected. Pass --image-key explicitly."
        )
    if selected_key not in first_sample:
        raise ValueError(
            f"Image key {selected_key!r} is absent. Detected keys: {detected_keys}"
        )

    trajectory_array = np.asarray(trajectory_frames)
    estimated: dict[int, int] = {}
    for frame in requested_frames:
        position = int(np.argmin(np.abs(trajectory_array - frame)))
        position = min(position, len(indices) - 1)
        estimated[frame] = indices[position]

    samples: dict[int, dict[str, Any]] = {}
    mismatched = False
    for requested_frame, dataset_index in estimated.items():
        sample = dataset[dataset_index]
        sample_frame = int(np.asarray(sample["frame_index"]).item())
        samples[requested_frame] = sample
        mismatched = mismatched or sample_frame != requested_frame
    if mismatched:
        frame_lookup: dict[int, dict[str, Any]] = {}
        for dataset_index in indices:
            sample = dataset[dataset_index]
            sample_frame = int(np.asarray(sample["frame_index"]).item())
            frame_lookup[sample_frame] = sample
        available = np.asarray(sorted(frame_lookup))
        samples = {
            frame: frame_lookup[int(available[np.argmin(np.abs(available - frame))])]
            for frame in requested_frames
        }

    images = {
        frame: _to_image_array(samples[frame][selected_key])
        for frame in requested_frames
    }
    return images, selected_key, _task_description(dataset_path, first_sample)


def _phase_segments(trajectory: pd.DataFrame) -> list[tuple[int, int, Any]]:
    frames = trajectory["frame_index"].astype(int).to_numpy()
    if len(frames) == 0 or trajectory["phase"].isna().all():
        return []
    phase_values = trajectory["phase"].to_numpy()
    starts = [0]
    starts.extend((np.flatnonzero(phase_values[1:] != phase_values[:-1]) + 1).tolist())
    ends = starts[1:] + [len(frames)]
    return [
        (int(frames[start]), int(frames[end - 1]), phase_values[start])
        for start, end in zip(starts, ends)
    ]


def _add_phase_context(
    axis: plt.Axes,
    trajectory: pd.DataFrame,
    phase_names: list[str],
    show_labels: bool,
) -> None:
    segments = _phase_segments(trajectory)
    for index, (start, end, phase) in enumerate(segments):
        axis.axvspan(
            start,
            end,
            color=PHASE_COLORS[index % len(PHASE_COLORS)],
            alpha=0.28,
            linewidth=0,
            zorder=0,
        )
        if show_labels:
            axis.text(
                (start + end) / 2,
                0.96,
                _phase_display_name(phase, phase_names),
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#52606D",
            )
    boundaries = trajectory[trajectory["is_boundary"]]
    for _, row in boundaries.iterrows():
        frame = int(row["frame_index"])
        axis.axvline(
            frame,
            color=BOUNDARY_COLOR,
            linestyle=(0, (4, 3)),
            linewidth=1.25,
            alpha=0.75,
            zorder=2,
        )


def _draw_phase_strip(
    axis: plt.Axes,
    trajectory: pd.DataFrame,
    phase_names: list[str],
    show_axis_label: bool = True,
) -> None:
    frames = trajectory["frame_index"].astype(int)
    axis.set_xlim(int(frames.iloc[0]), int(frames.iloc[-1]))
    axis.set_ylim(0.0, 1.0)
    for index, (start, end, phase) in enumerate(_phase_segments(trajectory)):
        axis.axvspan(
            start,
            end,
            ymin=0.08,
            ymax=0.92,
            color=PHASE_COLORS[index % len(PHASE_COLORS)],
            linewidth=0,
        )
        axis.text(
            (start + end) / 2,
            0.5,
            _phase_display_name(phase, phase_names),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="semibold",
            color="#334E68",
        )
    for frame in trajectory.loc[trajectory["is_boundary"], "frame_index"]:
        axis.axvline(
            int(frame),
            ymin=0.08,
            ymax=0.92,
            color=BOUNDARY_COLOR,
            linewidth=1.25,
        )
    if show_axis_label:
        axis.text(
            -0.012,
            0.5,
            "Semantic phases",
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color="#52606D",
        )
    axis.set_axis_off()


def _episode_outcome(trajectory: pd.DataFrame) -> tuple[str, str]:
    values = trajectory["is_success"].dropna()
    if values.empty:
        return "TRAJECTORY", BOUNDARY_COLOR
    value = values.iloc[-1]
    if isinstance(value, str):
        success = value.strip().lower() in {"true", "1", "yes"}
    else:
        success = bool(value)
    if success:
        return "SUCCESS", SUCCESS_COLOR
    return "FAILURE / INCOMPLETE", FAILURE_COLOR


def _style_plot_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#FCFDFE")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#90A4AE")
    axis.spines["bottom"].set_color("#90A4AE")
    axis.tick_params(colors="#455A64", labelsize=9)


def _metric_summary(trajectory: pd.DataFrame) -> dict[str, float]:
    raw_error = trajectory["raw_critic"] - trajectory["target_return"]
    fused_error = trajectory["fused_critic"] - trajectory["target_return"]
    raw_mae = float(raw_error.abs().mean())
    fused_mae = float(fused_error.abs().mean())
    improvement = 0.0 if raw_mae == 0.0 else 100.0 * (1.0 - fused_mae / raw_mae)
    return {
        "raw_mae": raw_mae,
        "fused_mae": fused_mae,
        "raw_bias": float(raw_error.mean()),
        "fused_bias": float(fused_error.mean()),
        "mae_improvement_pct": float(improvement),
    }


def _draw_compact_metric_strip(
    axis: plt.Axes,
    metrics: dict[str, float],
) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    entries = (
        (
            0.012,
            f"RAW CRITIC   MAE {metrics['raw_mae']:.2f}   "
            f"BIAS {metrics['raw_bias']:+.2f}",
            RAW_COLOR,
            "#FFF4EF",
        ),
        (
            0.385,
            f"FUSED CRITIC   MAE {metrics['fused_mae']:.2f}   "
            f"BIAS {metrics['fused_bias']:+.2f}",
            FUSED_COLOR,
            "#EFF6FC",
        ),
        (
            0.755,
            f"MAE IMPROVEMENT   {metrics['mae_improvement_pct']:.1f}%",
            SUCCESS_COLOR,
            "#F1F8F5",
        ),
    )
    for x_position, text, edge_color, face_color in entries:
        axis.text(
            x_position,
            0.5,
            text,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=9.2,
            color="#243B53",
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": face_color,
                "edgecolor": edge_color,
                "linewidth": 0.9,
                "alpha": 0.98,
            },
        )


def _plot_trajectory(
    trajectory: pd.DataFrame,
    images: dict[int, np.ndarray],
    image_key: str,
    output_base: Path,
    title: str,
    task_description: str,
    dataset_episode: int,
    phase_names: list[str],
    show_error_panel: bool,
    dpi: int,
    font_family: str | None,
    font_scale: float,
    font_weight: str | None,
    compact_paper: bool,
    hide_legend: bool,
) -> None:
    keyframes = sorted(images)
    columns = max(1, len(keyframes))
    figure_width = max(14.8, 2.45 * columns)
    if compact_paper:
        show_error_panel = False
        row_heights = [2.25, 0.18, 0.34, 2.75]
        figure_height = 6.1
        grid_vertical_space = 0.10
        grid_horizontal_space = 0.07
    else:
        row_heights = (
            [2.45, 0.22, 3.25, 1.15]
            if show_error_panel
            else [2.45, 0.22, 3.65]
        )
        figure_height = 8.8 if show_error_panel else 7.5
        grid_vertical_space = 0.20
        grid_horizontal_space = 0.08
    figure = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    grid = figure.add_gridspec(
        len(row_heights),
        columns,
        height_ratios=row_heights,
        hspace=grid_vertical_space,
        wspace=grid_horizontal_space,
    )
    image_axes = [figure.add_subplot(grid[0, index]) for index in range(columns)]
    phase_axis = figure.add_subplot(grid[1, :])
    if compact_paper:
        metric_axis = figure.add_subplot(grid[2, :])
        trajectory_axis = figure.add_subplot(grid[3, :])
        error_axis = None
    else:
        metric_axis = None
        trajectory_axis = figure.add_subplot(grid[2, :])
        error_axis = (
            figure.add_subplot(grid[3, :], sharex=trajectory_axis)
            if show_error_panel
            else None
        )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.835 if compact_paper else 0.875,
    )

    trajectory_lookup = trajectory.set_index("frame_index")
    outcome, outcome_color = _episode_outcome(trajectory)
    for image_number, (axis, frame) in enumerate(zip(image_axes, keyframes), start=1):
        axis.imshow(images[frame])
        phase = trajectory_lookup.loc[frame, "phase"]
        label = _phase_display_name(phase, phase_names)
        is_boundary = bool(trajectory_lookup.loc[frame, "is_boundary"])
        if is_boundary:
            label = f"Transition to {label}"
        elif frame == keyframes[0]:
            label = "Start"
        elif frame == keyframes[-1]:
            label = "End" if outcome == "SUCCESS" else "End (incomplete)"
        axis.text(
            0.045,
            0.93,
            str(image_number),
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "circle,pad=0.28",
                "facecolor": FUSED_COLOR,
                "edgecolor": "white",
                "linewidth": 1.0,
            },
        )
        if compact_paper:
            if frame == keyframes[0]:
                compact_label = f"Start  |  t={frame}"
            elif frame == keyframes[-1]:
                endpoint = "Success" if outcome == "SUCCESS" else "Incomplete"
                compact_label = f"{endpoint}  |  t={frame}"
            else:
                compact_label = f"t={frame}"
            axis.set_xlabel(
                compact_label,
                fontsize=9.6,
                color="#334E68",
                labelpad=4,
            )
        else:
            axis.set_xlabel(
                f"{label}\nTime step {frame}",
                fontsize=9,
                color="#334E68",
                labelpad=6,
            )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(FUSED_COLOR if is_boundary else "#9FB3C8")
            spine.set_linewidth(2.0 if is_boundary else 1.0)

    frames = trajectory["frame_index"].to_numpy()
    _draw_phase_strip(
        phase_axis,
        trajectory,
        phase_names,
        show_axis_label=not compact_paper,
    )
    _add_phase_context(trajectory_axis, trajectory, phase_names, show_labels=False)
    trajectory_axis.plot(
        frames,
        trajectory["target_return"],
        color=TARGET_COLOR,
        linestyle="--",
        linewidth=2.15,
        label="Return-to-go" if compact_paper else "Target return",
        zorder=4,
    )
    trajectory_axis.plot(
        frames,
        trajectory["raw_critic_plot"],
        color=RAW_COLOR,
        linewidth=2.15,
        label="Raw Critic",
        zorder=3,
    )
    trajectory_axis.plot(
        frames,
        trajectory["fused_critic_plot"],
        color=FUSED_COLOR,
        linewidth=2.65,
        label="Fused Critic" if compact_paper else "Ours (Fused Critic)",
        zorder=5,
    )
    trajectory_axis.set_ylabel("Return prediction", fontsize=10.5, color="#334E68")
    trajectory_axis.grid(axis="y", color="#CFD8DC", alpha=0.55, linewidth=0.7)
    if not hide_legend:
        trajectory_axis.legend(
            loc="lower left",
            ncol=3,
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="#D9E2EC",
            fontsize=9,
        )
    trajectory_axis.margins(x=0.01)
    trajectory_axis.tick_params(labelbottom=not show_error_panel)
    _style_plot_axis(trajectory_axis)

    for image_number, (axis, frame) in enumerate(zip(image_axes, keyframes), start=1):
        anchor_y = float(trajectory_lookup.loc[frame, "fused_critic_plot"])
        connector = ConnectionPatch(
            xyA=(0.5, 0.0),
            coordsA=axis.transAxes,
            xyB=(frame, anchor_y),
            coordsB=trajectory_axis.transData,
            color="#829AB1",
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            alpha=0.72,
            zorder=1,
            clip_on=False,
        )
        figure.add_artist(connector)
        trajectory_axis.scatter(
            [frame],
            [anchor_y],
            s=90,
            color=FUSED_COLOR,
            zorder=7,
            edgecolor="white",
            linewidth=1.2,
        )
        trajectory_axis.text(
            frame,
            anchor_y,
            str(image_number),
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="white",
            zorder=8,
        )

    if error_axis is not None:
        _add_phase_context(error_axis, trajectory, phase_names, show_labels=False)
        error_axis.plot(
            frames,
            trajectory["raw_abs_error"],
            color=RAW_COLOR,
            linewidth=1.45,
            label="Raw absolute error",
        )
        error_axis.plot(
            frames,
            trajectory["fused_abs_error"],
            color=FUSED_COLOR,
            linewidth=1.7,
            label="Fused absolute error",
        )
        error_axis.fill_between(
            frames,
            trajectory["fused_abs_error"].to_numpy(),
            color=FUSED_COLOR,
            alpha=0.10,
        )
        error_axis.set_ylabel("Absolute error", fontsize=9.5, color="#334E68")
        error_axis.set_xlabel("Time step", fontsize=10.5, color="#334E68")
        error_axis.grid(axis="y", color="#CFD8DC", alpha=0.5, linewidth=0.65)
        error_axis.legend(
            loc="upper right",
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        error_axis.margins(x=0.01)
        _style_plot_axis(error_axis)
    else:
        trajectory_axis.set_xlabel("Time step", fontsize=10.5, color="#334E68")

    metrics = _metric_summary(trajectory)
    if compact_paper:
        if metric_axis is None:
            raise RuntimeError("Compact metric axis was not initialized.")
        _draw_compact_metric_strip(metric_axis, metrics)
    else:
        metric_text = (
            f"Raw MAE  {metrics['raw_mae']:.2f}\n"
            f"Raw bias  {metrics['raw_bias']:+.2f}\n"
            f"Fused MAE  {metrics['fused_mae']:.2f}\n"
            f"Fused bias  {metrics['fused_bias']:+.2f}\n"
            f"MAE improvement  {metrics['mae_improvement_pct']:.1f}%"
        )
        trajectory_axis.text(
            0.988,
            0.982,
            metric_text,
            transform=trajectory_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.7,
            linespacing=1.35,
            color="#243B53",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "white",
                "edgecolor": "#BCCCDC",
                "linewidth": 0.9,
                "alpha": 0.96,
            },
            zorder=10,
        )

    figure.text(
        0.065,
        0.972 if compact_paper else 0.965,
        title,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#102A43",
    )
    description = textwrap.fill(
        task_description,
        width=125 if compact_paper else 105,
    )
    figure.text(
        0.065,
        0.938 if compact_paper else 0.944,
        description,
        ha="left",
        va="top",
        fontsize=10.2,
        color="#486581",
    )
    if compact_paper:
        status_text = f"{outcome}   |   {len(trajectory)} STEPS"
    else:
        status_text = (
            f"{outcome}   |   EPISODE {dataset_episode}   |   "
            f"{len(trajectory)} STEPS"
        )
    figure.text(
        0.985,
        0.982,
        status_text,
        ha="right",
        va="top",
        fontsize=9.3,
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.48",
            "facecolor": outcome_color,
            "edgecolor": outcome_color,
        },
    )
    if not compact_paper:
        figure.text(
            0.985,
            0.944,
            f"Camera: {image_key}",
            ha="right",
            va="top",
            fontsize=8.3,
            color="#829AB1",
        )
    for text_artist in figure.findobj(match=Text):
        if font_family is not None:
            text_artist.set_fontfamily(font_family)
        text_artist.set_fontsize(text_artist.get_fontsize() * font_scale)
        if font_weight is not None:
            text_artist.set_fontweight(font_weight)
    figure.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _output_base(output: str, dataset_episode: int) -> Path:
    path = Path(output).expanduser()
    if path.exists() and path.is_dir():
        return path / f"critic_trajectory_episode{dataset_episode}"
    if path.suffix.lower() in {".png", ".pdf", ".csv", ".json"}:
        return path.with_suffix("")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize target return, Raw Critic, Fused Critic, semantic phase "
            "boundaries, and aligned simulator keyframes for one episode."
        )
    )
    parser.add_argument("--dataset", required=True, help="Child LeRobot dataset path.")
    parser.add_argument(
        "--episode", required=True, type=int, help="Local dataset episode."
    )
    parser.add_argument("--comparison", help="return_compare.json path.")
    parser.add_argument("--advantages", help="Raw advantages parquet path.")
    parser.add_argument("--predictions", help="Fused predictions parquet path.")
    parser.add_argument(
        "--comparison-episode",
        type=int,
        help="Episode index in merged critic files. Overrides --episode-offset.",
    )
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Merged episode offset added to --episode, e.g. task8 uses 240.",
    )
    parser.add_argument("--phase-labels", help="Semantic phase parquet path.")
    parser.add_argument(
        "--phase-episode",
        type=int,
        help="Episode index in phase parquet; defaults to local --episode.",
    )
    parser.add_argument(
        "--image-key", help="Simulator image field; auto-detected by default."
    )
    parser.add_argument(
        "--output", required=True, help="Output stem, file, or directory."
    )
    parser.add_argument("--title", help="Figure title.")
    parser.add_argument("--task-name", help="Task name shown in the default title.")
    parser.add_argument(
        "--task-description",
        help="Task instruction shown below the title; auto-detected by default.",
    )
    parser.add_argument(
        "--phase-names",
        default="",
        help="Comma-separated phase names in numeric phase order.",
    )
    parser.add_argument(
        "--num-keyframes",
        "--max-keyframes",
        dest="num_keyframes",
        type=int,
        default=6,
        help="Number of simulator keyframes; defaults to 6.",
    )
    parser.add_argument("--boundary-window", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-error-panel", action="store_true")
    parser.add_argument(
        "--font-family",
        help="Optional font family applied to all labels, e.g. Times New Roman.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to every text size; defaults to 1.0.",
    )
    parser.add_argument(
        "--font-weight",
        choices=("normal", "medium", "semibold", "bold"),
        help="Optional font weight applied to all labels.",
    )
    parser.add_argument(
        "--compact-paper",
        action="store_true",
        help=(
            "Keep keyframes, phases, the main return plot, and MAE metrics "
            "while removing redundant labels and the absolute-error subplot."
        ),
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Hide the return-curve legend, useful after the first stacked panel.",
    )
    parser.add_argument("--return-min", type=float)
    parser.add_argument("--return-max", type=float)
    parser.add_argument("--value-min", type=float)
    parser.add_argument("--value-max", type=float)
    return parser


def main() -> None:
    """Load one episode and write its critic trajectory figure and sidecars."""
    args = _build_parser().parse_args()
    if args.episode < 0:
        raise ValueError("--episode must be non-negative.")
    if args.num_keyframes < 2:
        raise ValueError("--num-keyframes must be at least 2.")
    if args.boundary_window < 0:
        raise ValueError("--boundary-window must be non-negative.")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be at least 1.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be positive.")

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    comparison_episode = (
        args.comparison_episode
        if args.comparison_episode is not None
        else args.episode + args.episode_offset
    )
    paths = _resolve_comparison_inputs(args)
    trajectory = _load_episode_returns(
        paths, comparison_episode, args.smooth_window
    )
    phase_path = _discover_phase_path(dataset_path, args.phase_labels)
    requested_phase_episode = (
        args.phase_episode if args.phase_episode is not None else args.episode
    )
    trajectory, phase_path_text, used_phase_episode = _align_phases(
        trajectory,
        phase_path,
        requested_phase_episode,
        comparison_episode,
    )
    keyframes = _choose_keyframes(
        trajectory, args.num_keyframes, args.boundary_window
    )
    trajectory["is_keyframe"] = trajectory["frame_index"].isin(keyframes)
    images, image_key, detected_task_description = _load_keyframe_images(
        dataset_path,
        args.episode,
        keyframes,
        trajectory["frame_index"].astype(int).tolist(),
        args.image_key,
    )

    phase_names = [name.strip() for name in args.phase_names.split(",") if name.strip()]
    task_name = args.task_name or dataset_path.name
    title = args.title or task_name
    task_description = (
        args.task_description
        or detected_task_description
        or "Robot manipulation trajectory with semantic phase annotations."
    )
    output_base = _output_base(args.output, args.episode)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    _plot_trajectory(
        trajectory,
        images,
        image_key,
        output_base,
        title,
        task_description,
        args.episode,
        phase_names,
        not args.no_error_panel,
        args.dpi,
        args.font_family,
        args.font_scale,
        args.font_weight,
        args.compact_paper,
        args.hide_legend,
    )

    export_columns = [
        "frame_index",
        "target_return",
        "raw_critic",
        "fused_critic",
        "raw_abs_error",
        "fused_abs_error",
        "phase",
        "phase_progress",
        "is_success",
        "is_boundary",
        "is_keyframe",
    ]
    trajectory[export_columns].to_csv(output_base.with_suffix(".csv"), index=False)
    metrics = _metric_summary(trajectory)
    boundaries = trajectory.loc[
        trajectory["is_boundary"], ["frame_index", "phase"]
    ]
    metadata = {
        "dataset": str(dataset_path),
        "dataset_episode": int(args.episode),
        "comparison_episode": int(comparison_episode),
        "advantages_path": str(paths.advantages),
        "predictions_path": str(paths.predictions),
        "phase_path": phase_path_text,
        "phase_episode": used_phase_episode,
        "image_key": image_key,
        "task_name": task_name,
        "task_description": task_description,
        "outcome": _episode_outcome(trajectory)[0],
        "keyframes": keyframes,
        "num_keyframes": int(args.num_keyframes),
        "phase_boundaries": [
            {"frame_index": int(row.frame_index), "phase": str(row.phase)}
            for row in boundaries.itertuples(index=False)
        ],
        "scales": {
            "return_min": paths.return_min,
            "return_max": paths.return_max,
            "value_min": paths.value_min,
            "value_max": paths.value_max,
        },
        "smooth_window": int(args.smooth_window),
        "font_family": args.font_family,
        "font_scale": float(args.font_scale),
        "font_weight": args.font_weight,
        "compact_paper": bool(args.compact_paper),
        "hide_legend": bool(args.hide_legend),
        "metrics": metrics,
        "outputs": {
            "png": str(output_base.with_suffix(".png")),
            "pdf": str(output_base.with_suffix(".pdf")),
            "csv": str(output_base.with_suffix(".csv")),
        },
    }
    with output_base.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
