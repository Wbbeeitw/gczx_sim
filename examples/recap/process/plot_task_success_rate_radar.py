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

"""Plot five-method task-wise LIBERO-Long success rates as a radar chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PI05_SFT = "pi0.5-SFT"
METHODS = (
    "OpenVLA",
    "Xiaomi-Robotics-0",
    PI05_SFT,
    "ReCAP",
    "PACE (ours)",
)
DISPLAY_LABELS = {
    "OpenVLA": "OpenVLA",
    "Xiaomi-Robotics-0": "Xiaomi-Robotics-0",
    PI05_SFT: r"$\pi_{0.5}$-SFT",
    "ReCAP": "ReCAP",
    "PACE (ours)": "PACE (ours)",
}
ALIASES = {
    "OpenVLA": ("OpenVLA",),
    "Xiaomi-Robotics-0": ("Xiaomi-Robotics-0", "XR-0"),
    PI05_SFT: (
        "π₀.₅-SFT",
        "π0.5-SFT",
        "pi0.5-SFT",
        "pi05-SFT",
        "π₀.₅ (SFT)",
    ),
    "ReCAP": ("ReCAP", "π0.5 (ReCAP)", "π₀.₅ (ReCAP)"),
    "PACE (ours)": ("PACE (ours)", "PACE", "Ours (PACE)"),
}
BASELINE_STYLES = {
    "OpenVLA": {
        "color": "#B8C4CE",
        "linestyle": (0, (2, 2)),
        "marker": "o",
    },
    "Xiaomi-Robotics-0": {
        "color": "#AFC9B6",
        "linestyle": (0, (5, 2)),
        "marker": "^",
    },
    PI05_SFT: {
        "color": "#D2B8C5",
        "linestyle": (0, (7, 2, 1, 2)),
        "marker": "D",
    },
}
RECAP_COLOR = "#E76F51"
PACE_COLOR = "#2364AA"


def _task_order(num_tasks: int) -> list[str]:
    return [f"task{task_index}" for task_index in range(num_tasks)]


def _resolve_task_column(frame: pd.DataFrame) -> str:
    for candidate in ("task", "Task", "task_name"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Input CSV must contain a task column")


def _resolve_method_column(frame: pd.DataFrame, method: str) -> str:
    matches = [column for column in ALIASES[method] if column in frame.columns]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one column for {method!r}; aliases={ALIASES[method]}, "
            f"matches={matches}"
        )
    return matches[0]


def _load_success_rates(
    path: Path,
    *,
    tasks: list[str],
) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    task_column = _resolve_task_column(frame)
    if frame[task_column].duplicated().any():
        raise ValueError("Input CSV contains duplicate task rows")
    indexed = frame.set_index(task_column)
    result: dict[str, np.ndarray] = {}
    for method in METHODS:
        column = _resolve_method_column(frame, method)
        values = indexed[column].reindex(tasks)
        if values.isna().any():
            missing = values.index[values.isna()].tolist()
            raise ValueError(f"Method {method!r} lacks task rows: {missing}")
        array = values.to_numpy(dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"Method {method!r} contains non-finite success rates")
        if (array < 0.0).any() or (array > 100.0).any():
            raise ValueError(
                f"Method {method!r} must use percentages in [0, 100], got "
                f"range [{array.min()}, {array.max()}]"
            )
        result[method] = array
    return result


def _closed(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[:1]])


def _plot_radar(
    *,
    tasks: list[str],
    values: dict[str, np.ndarray],
    output_path: Path,
    pace_color: str,
    recap_color: str,
    font_family: str,
    dpi: int,
) -> dict[str, Any]:
    angles = np.linspace(0.0, 2.0 * np.pi, len(tasks), endpoint=False)
    closed_angles = _closed(angles)
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.weight": "semibold",
            "mathtext.fontset": "stix",
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
    axis.set_ylim(0.0, 100.0)
    axis.set_xticks(angles)
    axis.set_xticklabels(
        [task.replace("task", "Task ") for task in tasks],
        fontsize=12.5,
        fontweight="bold",
        color="#334E68",
    )
    axis.tick_params(axis="x", pad=13)
    radial_ticks = np.asarray([20.0, 40.0, 60.0, 80.0, 100.0])
    axis.set_yticks(radial_ticks)
    axis.set_yticklabels(
        [f"{tick:.0f}%" for tick in radial_ticks],
        fontsize=9.5,
        color="#607D8B",
    )
    axis.set_rlabel_position(18)
    axis.xaxis.grid(True, color="#B0BEC5", linewidth=0.75, alpha=0.55)
    axis.yaxis.grid(True, color="#B0BEC5", linewidth=0.75, alpha=0.55)
    axis.spines["polar"].set_color("#90A4AE")
    axis.spines["polar"].set_linewidth(0.9)

    for method in METHODS[:3]:
        style = BASELINE_STYLES[method]
        axis.plot(
            closed_angles,
            _closed(values[method]),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            alpha=0.9,
            marker=style["marker"],
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.15,
            label=DISPLAY_LABELS[method],
            zorder=3,
        )

    axis.plot(
        closed_angles,
        _closed(values["ReCAP"]),
        color=recap_color,
        linewidth=2.35,
        linestyle="-",
        marker="o",
        markersize=6.2,
        markerfacecolor="white",
        markeredgecolor=recap_color,
        markeredgewidth=1.8,
        label="ReCAP",
        zorder=5,
    )
    axis.fill(
        closed_angles,
        _closed(values["ReCAP"]),
        color=recap_color,
        alpha=0.025,
        zorder=2,
    )
    axis.plot(
        closed_angles,
        _closed(values["PACE (ours)"]),
        color=pace_color,
        linewidth=3.1,
        linestyle="-",
        marker="s",
        markersize=6.5,
        markerfacecolor="white",
        markeredgecolor=pace_color,
        markeredgewidth=2.0,
        label="PACE (ours)",
        zorder=6,
    )
    axis.fill(
        closed_angles,
        _closed(values["PACE (ours)"]),
        color=pace_color,
        alpha=0.055,
        zorder=2,
    )

    legend = axis.legend(
        loc="upper right",
        bbox_to_anchor=(1.20, 1.13),
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#CFD8DC",
        fontsize=10.0,
        handlelength=2.7,
    )
    for text in legend.get_texts():
        text.set_fontweight("semibold")
    figure.text(
        0.925,
        0.045,
        r"Success rate (%)  $\uparrow$  Higher is better",
        ha="right",
        va="center",
        fontsize=10.5,
        fontstyle="italic",
        color="#607D8B",
    )
    figure.subplots_adjust(left=0.08, right=0.89, top=0.91, bottom=0.10)

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
        "radial_range_percent": [0.0, 100.0],
        "method_order": list(METHODS),
        "pace_color": pace_color,
        "recap_color": recap_color,
        "success_rates_percent": {
            method: method_values.tolist()
            for method, method_values in values.items()
        },
        "task_macro_success_rate_percent": {
            method: float(method_values.mean())
            for method, method_values in values.items()
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--pace-color", default=PACE_COLOR)
    parser.add_argument("--recap-color", default=RECAP_COLOR)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main() -> None:
    """Load task-wise success rates and render the five-method radar."""
    args = _build_parser().parse_args()
    if args.num_tasks < 3:
        raise ValueError("--num-tasks must be at least 3 for a radar chart")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    input_path = args.input_csv.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    tasks = _task_order(args.num_tasks)
    values = _load_success_rates(input_path, tasks=tasks)
    output_path = args.output.expanduser().resolve()
    metadata = _plot_radar(
        tasks=tasks,
        values=values,
        output_path=output_path,
        pace_color=args.pace_color,
        recap_color=args.recap_color,
        font_family=args.font_family,
        dpi=args.dpi,
    )
    metadata.update(
        {
            "input_csv": str(input_path),
            "metric": "Task success rate (%)",
            "direction": "higher_is_better",
        }
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PNG: {metadata['png']}")
    print(f"PDF: {metadata['pdf']}")
    print(f"JSON: {metadata_path}")
    for method in METHODS:
        macro = metadata["task_macro_success_rate_percent"][method]
        print(f"{method}: task-macro SR={macro:.2f}%")


if __name__ == "__main__":
    main()
