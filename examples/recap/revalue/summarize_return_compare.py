#!/usr/bin/env python
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

"""Render Revalue return comparison JSON as readable tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from rlinf.revalue.evaluation import summarize_return_report
except ModuleNotFoundError as exc:
    if exc.name != "omegaconf":
        raise
    omegaconf = types.ModuleType("omegaconf")

    class OmegaConf:
        @staticmethod
        def register_new_resolver(*args, **kwargs):
            return None

    omegaconf.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = omegaconf
    from rlinf.revalue.evaluation import summarize_return_report


def _load_report(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _safe_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _safe_number(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _line(values: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    parts = [_line(headers), _line(["-" * width for width in widths])]
    parts.extend(_line(row) for row in rows)
    return "\n".join(parts)


def _render_markdown(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _frame_table_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            str(row["split"]),
            str(row["rows"]),
            str(row["episodes"]),
            _fmt(row["base_mse"]),
            _fmt(row["fused_mse"]),
            _fmt(row["mse_improvement_pct"]),
            _fmt(row["base_mae"]),
            _fmt(row["fused_mae"]),
            _fmt(row["mae_improvement_pct"]),
            _fmt(row["base_bias"]),
            _fmt(row["fused_bias"]),
        ]
        for row in rows
    ]


def _episode_table_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            str(row["split"]),
            str(row["episodes"]),
            str(row["episodes_improved_mse"]),
            _fmt(row["episodes_improved_mse_pct"]),
            _fmt(row["mean_base_mse"]),
            _fmt(row["mean_fused_mse"]),
            _fmt(row["mean_mse_gain"]),
            _fmt(row["mean_base_mae"]),
            _fmt(row["mean_fused_mae"]),
            _fmt(row["mean_mae_gain"]),
        ]
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a Revalue return_compare.json into tables."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to return_compare.json produced by stage=compare_returns.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory for CSV/Markdown summaries. "
        "Defaults to the JSON parent directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _load_report(input_path)
    summary = summarize_return_report(report)
    frame_rows = summary["frame_rows"]
    episode_rows = summary["episode_rows"]

    frame_headers = [
        "split",
        "rows",
        "episodes",
        "base_mse",
        "fused_mse",
        "mse_gain_%",
        "base_mae",
        "fused_mae",
        "mae_gain_%",
        "base_bias",
        "fused_bias",
    ]
    episode_headers = [
        "split",
        "episodes",
        "improved_eps",
        "improved_eps_%",
        "mean_base_mse",
        "mean_fused_mse",
        "mean_mse_gain",
        "mean_base_mae",
        "mean_fused_mae",
        "mean_mae_gain",
    ]

    frame_table = _render_table(frame_headers, _frame_table_rows(frame_rows))
    episode_table = _render_table(episode_headers, _episode_table_rows(episode_rows))
    frame_md = _render_markdown(frame_headers, _frame_table_rows(frame_rows))
    episode_md = _render_markdown(episode_headers, _episode_table_rows(episode_rows))

    frame_csv_path = output_dir / "return_compare_frame_summary.csv"
    episode_csv_path = output_dir / "return_compare_episode_summary.csv"
    md_path = output_dir / "return_compare_summary.md"

    _write_csv(frame_csv_path, list(frame_rows[0].keys()) if frame_rows else [], frame_rows)
    _write_csv(
        episode_csv_path,
        list(episode_rows[0].keys()) if episode_rows else [],
        episode_rows,
    )
    md_path.write_text(
        "\n".join(
            [
                "# Revalue Return Comparison Summary",
                "",
                "## Frame-Level Summary",
                "",
                frame_md if frame_rows else "_No frame-level rows._",
                "",
                "## Episode-Level Summary",
                "",
                episode_md if episode_rows else "_No episode-level rows._",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Frame-Level Summary")
    print(frame_table if frame_rows else "No frame-level rows.")
    print()
    print("Episode-Level Summary")
    print(episode_table if episode_rows else "No episode-level rows.")
    print()
    print(f"Markdown summary: {md_path}")
    print(f"Frame CSV: {frame_csv_path}")
    print(f"Episode CSV: {episode_csv_path}")


if __name__ == "__main__":
    main()
