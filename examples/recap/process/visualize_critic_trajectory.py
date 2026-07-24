"""Create a paper-ready raw-vs-fused critic trajectory visualization.

The figure combines simulator keyframes, target/raw/fused return trajectories,
semantic phase boundaries, and an optional absolute-error panel. It also writes
the aligned frame data and plotting metadata for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch


TARGET_COLOR = "#222222"
RAW_COLOR = "#D55E00"
FUSED_COLOR = "#0072B2"
BOUNDARY_COLOR = "#3A506B"
PHASE_COLORS = ("#DCEAF7", "#F2F2F2", "#E8F3E8", "#F8E6DD")


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
    max_keyframes: int,
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

    candidates: list[tuple[int, int]] = []
    frame_array = np.asarray(frames)
    for boundary in boundaries:
        for requested in (boundary - boundary_window, boundary + boundary_window):
            nearest = int(frame_array[np.argmin(np.abs(frame_array - requested))])
            candidates.append((0, nearest))
    if trajectory["phase"].notna().any():
        for _, phase_frame in trajectory.groupby("phase", sort=False, dropna=True):
            midpoint = int(phase_frame.iloc[len(phase_frame) // 2]["frame_index"])
            candidates.append((1, midpoint))
    evenly_spaced = np.linspace(0, len(frames) - 1, max_keyframes, dtype=int)
    candidates.extend((2, frames[index]) for index in evenly_spaced)

    selected = mandatory[:max_keyframes]
    minimum_gap = max(1, len(frames) // max(2, max_keyframes * 3))
    for _, candidate in sorted(candidates):
        if len(selected) >= max_keyframes:
            break
        if candidate in selected:
            continue
        if any(abs(candidate - existing) < minimum_gap for existing in selected):
            continue
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


def _load_keyframe_images(
    dataset_path: Path,
    episode_index: int,
    requested_frames: list[int],
    trajectory_frames: list[int],
    image_key: str | None,
) -> tuple[dict[int, np.ndarray], str]:
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
    return images, selected_key


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
                0.97,
                _phase_display_name(phase, phase_names),
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="#4A4A4A",
            )
    boundaries = trajectory[trajectory["is_boundary"]]
    previous_phase = trajectory["phase"].shift(1)
    for index, row in boundaries.iterrows():
        frame = int(row["frame_index"])
        axis.axvline(
            frame,
            color=BOUNDARY_COLOR,
            linestyle=(0, (5, 3)),
            linewidth=1.6,
            alpha=0.9,
            zorder=2,
        )
        if show_labels:
            before = _phase_display_name(previous_phase.loc[index], phase_names)
            after = _phase_display_name(row["phase"], phase_names)
            axis.annotate(
                f"{before} -> {after}",
                xy=(frame, 1.0),
                xycoords=axis.get_xaxis_transform(),
                xytext=(4, 7),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="bottom",
                fontsize=8,
                color=BOUNDARY_COLOR,
                clip_on=False,
            )


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


def _plot_trajectory(
    trajectory: pd.DataFrame,
    images: dict[int, np.ndarray],
    image_key: str,
    output_base: Path,
    title: str,
    phase_names: list[str],
    show_error_panel: bool,
    dpi: int,
) -> None:
    keyframes = sorted(images)
    columns = max(1, len(keyframes))
    figure_width = max(11.0, 2.0 * columns)
    row_heights = [1.65, 3.2, 1.35] if show_error_panel else [1.65, 3.5]
    figure = plt.figure(
        figsize=(figure_width, sum(row_heights) + 0.8), constrained_layout=True
    )
    grid = figure.add_gridspec(
        len(row_heights), columns, height_ratios=row_heights, hspace=0.08, wspace=0.05
    )
    image_axes = [figure.add_subplot(grid[0, index]) for index in range(columns)]
    trajectory_axis = figure.add_subplot(grid[1, :])
    error_axis = (
        figure.add_subplot(grid[2, :], sharex=trajectory_axis)
        if show_error_panel
        else None
    )

    trajectory_lookup = trajectory.set_index("frame_index")
    for axis, frame in zip(image_axes, keyframes):
        axis.imshow(images[frame])
        phase = trajectory_lookup.loc[frame, "phase"]
        label = _phase_display_name(phase, phase_names)
        if bool(trajectory_lookup.loc[frame, "is_boundary"]):
            label = f"Boundary | {label}"
        axis.set_title(f"t = {frame}\n{label}", fontsize=9, pad=4)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(BOUNDARY_COLOR if "Boundary" in label else "#A0A0A0")
            spine.set_linewidth(2.2 if "Boundary" in label else 0.8)

    frames = trajectory["frame_index"].to_numpy()
    _add_phase_context(trajectory_axis, trajectory, phase_names, show_labels=True)
    trajectory_axis.plot(
        frames,
        trajectory["target_return"],
        color=TARGET_COLOR,
        linestyle="--",
        linewidth=2.0,
        label="Target return",
        zorder=4,
    )
    trajectory_axis.plot(
        frames,
        trajectory["raw_critic_plot"],
        color=RAW_COLOR,
        linewidth=2.0,
        label="Raw Critic",
        zorder=3,
    )
    trajectory_axis.plot(
        frames,
        trajectory["fused_critic_plot"],
        color=FUSED_COLOR,
        linewidth=2.3,
        label="Ours (Fused Critic)",
        zorder=5,
    )
    trajectory_axis.set_ylabel("Predicted return", fontsize=11)
    trajectory_axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    trajectory_axis.legend(loc="lower right", ncol=3, frameon=True, fontsize=9)
    trajectory_axis.margins(x=0.01)
    trajectory_axis.tick_params(labelbottom=not show_error_panel)

    for axis, frame in zip(image_axes, keyframes):
        target_y = float(trajectory_lookup.loc[frame, "target_return"])
        connector = ConnectionPatch(
            xyA=(0.5, 0.0),
            coordsA=axis.transAxes,
            xyB=(frame, target_y),
            coordsB=trajectory_axis.transData,
            color="#6B7280",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
            zorder=1,
            clip_on=False,
        )
        figure.add_artist(connector)
        trajectory_axis.scatter(
            [frame], [target_y], s=24, color=TARGET_COLOR, zorder=6, edgecolor="white"
        )

    if error_axis is not None:
        _add_phase_context(error_axis, trajectory, phase_names, show_labels=False)
        error_axis.plot(
            frames,
            trajectory["raw_abs_error"],
            color=RAW_COLOR,
            linewidth=1.6,
            label="Raw absolute error",
        )
        error_axis.plot(
            frames,
            trajectory["fused_abs_error"],
            color=FUSED_COLOR,
            linewidth=1.8,
            label="Fused absolute error",
        )
        error_axis.fill_between(
            frames,
            trajectory["fused_abs_error"].to_numpy(),
            color=FUSED_COLOR,
            alpha=0.08,
        )
        error_axis.set_ylabel("Absolute error", fontsize=10)
        error_axis.set_xlabel("Episode frame", fontsize=11)
        error_axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        error_axis.legend(loc="upper right", ncol=2, fontsize=8)
        error_axis.margins(x=0.01)
    else:
        trajectory_axis.set_xlabel("Episode frame", fontsize=11)

    metrics = _metric_summary(trajectory)
    subtitle = (
        f"Raw MAE {metrics['raw_mae']:.2f}  |  Fused MAE {metrics['fused_mae']:.2f}  "
        f"|  improvement {metrics['mae_improvement_pct']:.1f}%  |  image: {image_key}"
    )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=13, fontweight="semibold")
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
        "--phase-names",
        default="",
        help="Comma-separated phase names in numeric phase order.",
    )
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--boundary-window", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-error-panel", action="store_true")
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
    if args.max_keyframes < 2:
        raise ValueError("--max-keyframes must be at least 2.")
    if args.boundary_window < 0:
        raise ValueError("--boundary-window must be non-negative.")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be at least 1.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")

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
        trajectory, args.max_keyframes, args.boundary_window
    )
    trajectory["is_keyframe"] = trajectory["frame_index"].isin(keyframes)
    images, image_key = _load_keyframe_images(
        dataset_path,
        args.episode,
        keyframes,
        trajectory["frame_index"].astype(int).tolist(),
        args.image_key,
    )

    phase_names = [name.strip() for name in args.phase_names.split(",") if name.strip()]
    task_name = args.task_name or dataset_path.name
    title = args.title or (
        f"{task_name} | dataset episode {args.episode} | "
        f"critic episode {comparison_episode}"
    )
    output_base = _output_base(args.output, args.episode)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    _plot_trajectory(
        trajectory,
        images,
        image_key,
        output_base,
        title,
        phase_names,
        not args.no_error_panel,
        args.dpi,
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
        "keyframes": keyframes,
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
