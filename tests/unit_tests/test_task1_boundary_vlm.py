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

import numpy as np

from phase_split_script import task1_boundary_vlm


def test_coarse_boundaries_from_legacy_phase_success():
    legacy = np.array([0, 0, 1, 1, 2, 2, 4, 5, 5], dtype=int)
    boundaries = task1_boundary_vlm._coarse_boundaries_from_legacy(
        legacy,
        is_success=True,
    )

    assert boundaries == {"b1": 2, "b2": 4, "b3": 7}


def test_coarse_boundaries_from_legacy_phase_failure():
    legacy = np.array([0, 0, 0, 1, 1, 2, 2], dtype=int)
    boundaries = task1_boundary_vlm._coarse_boundaries_from_legacy(
        legacy,
        is_success=False,
    )

    assert boundaries == {"b1": 3, "b2": 5, "b3": None}


def test_window_sample_indices_stay_local():
    indices = task1_boundary_vlm._window_sample_indices(
        center_frame=50,
        episode_length=100,
        fps=10.0,
        window_seconds=4.0,
        num_samples=9,
    )

    assert len(indices) == 9
    assert indices[0] >= 30
    assert indices[-1] <= 70
    assert indices == sorted(indices)


def test_boundaries_to_phase_array_supports_skipping_phase_two():
    phase = task1_boundary_vlm.boundaries_to_phase_array(
        episode_length=8,
        boundaries={"b1": 2, "b2": None, "b3": 6},
    )

    assert phase.tolist() == [0, 0, 1, 1, 1, 1, 3, 3]


def test_build_phase_df_from_boundaries_freezes_failed_terminal_segment():
    df = task1_boundary_vlm.build_phase_df_from_boundaries(
        episode_index=7,
        episode_length=6,
        boundaries={"b1": 2, "b2": 4, "b3": None},
        is_success=False,
    )

    assert df["phase"].tolist() == [0, 0, 1, 1, 2, 2]
    assert df["phase_progress"].tolist()[-2:] == [0.0, 0.0]
    assert df["global_progress"].tolist()[-2:] == [2 / 4, 2 / 4]
