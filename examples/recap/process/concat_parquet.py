#!/usr/bin/env python
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Concatenate per-task prediction parquets into one joint predictions file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    """Concatenate and sort prediction parquets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = [pd.read_parquet(path) for path in args.inputs]
    merged = pd.concat(frames, ignore_index=True)
    sort_columns = [
        column
        for column in ("episode_index", "frame_index")
        if column in merged.columns
    ]
    if sort_columns:
        merged = merged.sort_values(sort_columns).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    print(
        f"wrote {output} rows={len(merged)} "
        f"episodes={merged['episode_index'].nunique()}"
    )


if __name__ == "__main__":
    main()
