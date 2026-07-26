# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot task-wise Raw-versus-Full boundary MAE as a paper-ready radar chart."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_VARIANT = "Raw Critic"
FULL_VARIANT = "Full GLC-Critic"
RAW_COLOR = "#E76F51"
PACE_COLOR = "#2364AA"


def _task_order(num_tasks: int) -> list[str]:
    return [f"task{task_index}" for task_index in range(num_tasks)]


def _variant_values(
    frame: pd.DataFrame,
    *,
    variant: str,
    tasks: list[str],
) -> np.ndarray:
    required = {"critic_variant", "task", "boundary_mae"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"By-task CSV is missing columns: {sorted(missing)}")
    selected = frame[frame["critic_variant"] == variant].copy()
    if selected.empty:
        available = sorted(frame["critic_variant"].dropna().astype(str).unique())
        raise ValueError(f"Variant {variant!r} not found; available: {available}")
    if selected["task"].duplicated().any():
        raise ValueError(f"Variant {variant!r} contains duplicate task rows")
    values = selected.set_index("task")["boundary_mae"].reindex(tasks)
    if values.isna().any():
        missing_tasks = values.index[values.isna()].tolist()
        raise ValueError(f"Variant {variant!r} lacks tasks: {missing_tasks}")
    array = values.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"Variant {variant!r} has invalid Boundary MAE values")
    return array


def _overall_values(
    summary_path: Path,
    *,
    raw_variant: str,
    full_variant: str,
) -> tuple[float, float]:
    frame = pd.read_csv(summary_path)
    required = {"critic_variant", "boundary_mae"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing columns: {sorted(missing)}")

    def resolve(variant: str) -> float:
        values = frame.loc[
            frame["critic_variant"] == variant,
            "boundary_mae",
        ]
        if len(values) != 1:
            raise ValueError(
                f"Expected one aggregate row for {variant!r}, found {len(values)}"
            )
        return float(values.iloc[0])

    return resolve(raw_variant), resolve(full_variant)


def _closed(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[:1]])


def _nice_radial_max(maximum: float) -> float:
    if maximum <= 0.0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(maximum))
    normalized = maximum / magnitude
    step = 1.0 if normalized <= 5.0 else 2.0
    return math.ceil(normalized / step) * step * magnitude


def _plot_radar(
    *,
    tasks: list[str],
    raw_values: np.ndarray,
    pace_values: np.ndarray,
    raw_overall: float,
    pace_overall: float,
    output_path: Path,
    raw_label: str,
    pace_label: str,
    raw_color: str,
    pace_color: str,
    font_family: str,
    dpi: int,
) -> dict[str, Any]:
    angles = np.linspace(0.0, 2.0 * np.pi, len(tasks), endpoint=False)
    closed_angles = _closed(angles)
    closed_raw = _closed(raw_values)
    closed_pace = _closed(pace_values)
    radial_max = _nice_radial_max(float(max(raw_values.max(), pace_values.max())) * 1.08)
    radial_ticks = np.linspace(0.0, radial_max, 5)[1:]

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.weight": "semibold",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#90A4AE",
            "text.color": "#243B53",
        }
    )
    figure, axis = plt.subplots(
        figsize=(9.2, 8.2),
        subplot_kw={"projection": "polar"},
    )
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#FCFDFE")
    axis.set_theta_offset(np.pi / 2.0)
    axis.set_theta_direction(-1)
    axis.set_ylim(0.0, radial_max)
    axis.set_xticks(angles)
    axis.set_xticklabels(
        [task.replace("task", "Task ") for task in tasks],
        fontsize=12.5,
        fontweight="bold",
        color="#334E68",
    )
    axis.tick_params(axis="x", pad=13)
    axis.set_yticks(radial_ticks)
    axis.set_yticklabels(
        [f"{tick:.0f}" for tick in radial_ticks],
        fontsize=9.5,
        color="#607D8B",
    )
    axis.set_rlabel_position(18)
    axis.xaxis.grid(True, color="#B0BEC5", linewidth=0.75, alpha=0.55)
    axis.yaxis.grid(True, color="#B0BEC5", linewidth=0.75, alpha=0.55)
    axis.spines["polar"].set_color("#90A4AE")
    axis.spines["polar"].set_linewidth(0.9)

    axis.plot(
        closed_angles,
        closed_raw,
        color=raw_color,
        linewidth=2.35,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgecolor=raw_color,
        markeredgewidth=1.8,
        label=raw_label,
        zorder=4,
    )
    axis.fill(closed_angles, closed_raw, color=raw_color, alpha=0.045, zorder=2)
    axis.plot(
        closed_angles,
        closed_pace,
        color=pace_color,
        linewidth=2.7,
        marker="s",
        markersize=6.3,
        markerfacecolor="white",
        markeredgecolor=pace_color,
        markeredgewidth=1.9,
        label=pace_label,
        zorder=5,
    )
    axis.fill(closed_angles, closed_pace, color=pace_color, alpha=0.065, zorder=3)

    axis.set_title(
        "Task-wise Boundary MAE ↓",
        fontsize=18,
        pad=28,
        color="#102A43",
    )
    legend = axis.legend(
        loc="upper right",
        bbox_to_anchor=(1.18, 1.13),
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#CFD8DC",
        fontsize=10.5,
        handlelength=2.6,
    )
    for text in legend.get_texts():
        text.set_fontweight("semibold")

    improvement = (
        0.0
        if raw_overall == 0.0
        else 100.0 * (raw_overall - pace_overall) / raw_overall
    )
    figure.text(
        0.075,
        0.045,
        f"Overall Boundary MAE: {raw_overall:.2f} → {pace_overall:.2f}  "
        f"(−{improvement:.1f}%)",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#334E68",
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#F5F8FA",
            "edgecolor": "#B0BEC5",
            "linewidth": 0.9,
        },
    )
    figure.text(
        0.925,
        0.045,
        "Lower is better",
        ha="right",
        va="center",
        fontsize=10.5,
        fontstyle="italic",
        color="#607D8B",
    )
    figure.subplots_adjust(left=0.08, right=0.89, top=0.88, bottom=0.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "tasks": tasks,
        "raw_label": raw_label,
        "pace_label": pace_label,
        "raw_color": raw_color,
        "pace_color": pace_color,
        "raw_boundary_mae": raw_values.tolist(),
        "pace_boundary_mae": pace_values.tolist(),
        "raw_overall_boundary_mae": raw_overall,
        "pace_overall_boundary_mae": pace_overall,
        "overall_reduction_percent": improvement,
        "radial_max": radial_max,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--by-task-csv", type=Path, required=True)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Aggregate table; defaults to critic_ablation_table.csv beside by-task CSV.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--raw-variant", default=RAW_VARIANT)
    parser.add_argument("--full-variant", default=FULL_VARIANT)
    parser.add_argument("--raw-label", default="ReCAP / Raw Critic")
    parser.add_argument("--pace-label", default="PACE / Full GLC-Critic")
    parser.add_argument("--raw-color", default=RAW_COLOR)
    parser.add_argument("--pace-color", default=PACE_COLOR)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main() -> None:
    """Load fixed task metrics and render the radar chart."""
    args = _build_parser().parse_args()
    if args.num_tasks < 3:
        raise ValueError("--num-tasks must be at least 3 for a radar chart")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    by_task_path = args.by_task_csv.expanduser().resolve()
    summary_path = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv is not None
        else by_task_path.with_name("critic_ablation_table.csv")
    )
    for path in (by_task_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tasks = _task_order(args.num_tasks)
    by_task = pd.read_csv(by_task_path)
    raw_values = _variant_values(
        by_task,
        variant=args.raw_variant,
        tasks=tasks,
    )
    pace_values = _variant_values(
        by_task,
        variant=args.full_variant,
        tasks=tasks,
    )
    raw_overall, pace_overall = _overall_values(
        summary_path,
        raw_variant=args.raw_variant,
        full_variant=args.full_variant,
    )
    output_path = args.output.expanduser().resolve()
    metadata = _plot_radar(
        tasks=tasks,
        raw_values=raw_values,
        pace_values=pace_values,
        raw_overall=raw_overall,
        pace_overall=pace_overall,
        output_path=output_path,
        raw_label=args.raw_label,
        pace_label=args.pace_label,
        raw_color=args.raw_color,
        pace_color=args.pace_color,
        font_family=args.font_family,
        dpi=args.dpi,
    )
    metadata.update(
        {
            "by_task_csv": str(by_task_path),
            "summary_csv": str(summary_path),
            "metric": "Boundary MAE",
            "direction": "lower_is_better",
            "aggregate_note": "Overall values are frame-micro, not task-macro.",
        }
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PNG: {metadata['png']}")
    print(f"PDF: {metadata['pdf']}")
    print(f"JSON: {metadata_path}")
    print(
        "Overall Boundary MAE: "
        f"{raw_overall:.2f} -> {pace_overall:.2f} "
        f"(-{metadata['overall_reduction_percent']:.1f}%)"
    )


if __name__ == "__main__":
    main()
