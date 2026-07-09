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
from PIL import Image

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


def test_b1_prompt_uses_earlier_grasp_semantics():
    spec = next(s for s in task1_boundary_vlm.BOUNDARY_SPECS if s.key == "b1")
    prompt = task1_boundary_vlm._build_boundary_prompt(
        spec=spec,
        sampled_indices=[0, 10, 20],
        fps=10.0,
        is_success=True,
        enable_reasoning=False,
    )

    assert "grasped, lifted, carried, or deliberately dragged" in prompt
    assert "do NOT wait for a long stable hold" in prompt


def test_refine_boundary_searches_without_coarse_for_success(monkeypatch):
    spec = next(s for s in task1_boundary_vlm.BOUNDARY_SPECS if s.key == "b3")
    frames = [Image.new("RGB", (8, 8), color="black") for _ in range(40)]
    seen = {}

    def fake_query(**kwargs):
        seen["sampled_indices"] = kwargs["sampled_indices"]
        return task1_boundary_vlm.BoundaryDecision(
            status="within",
            frame_index=kwargs["sampled_indices"][-1],
            reasoning="found near the tail",
        )

    monkeypatch.setattr(task1_boundary_vlm, "_query_boundary_window", fake_query)
    refinement = task1_boundary_vlm._refine_boundary(
        spec=spec,
        coarse_frame=None,
        main_frames=frames,
        wrist_frames=None,
        fps=10.0,
        episode_length=len(frames),
        model="dummy-model",
        is_success=True,
        enable_reasoning=False,
    )

    assert seen["sampled_indices"]
    assert refinement.status == "within"
    assert refinement.source == "vlm_seed_search"
    assert refinement.final_frame == seen["sampled_indices"][-1]


def test_finalize_boundaries_backfills_success_tail_and_b1():
    refinements = {
        "b1": task1_boundary_vlm.BoundaryRefinement(
            key="b1",
            coarse_frame=None,
            final_frame=None,
            status="absent",
            source="coarse_absent",
            attempts=[],
        ),
        "b2": task1_boundary_vlm.BoundaryRefinement(
            key="b2",
            coarse_frame=None,
            final_frame=None,
            status="absent",
            source="coarse_absent",
            attempts=[],
        ),
        "b3": task1_boundary_vlm.BoundaryRefinement(
            key="b3",
            coarse_frame=None,
            final_frame=None,
            status="absent",
            source="coarse_absent",
            attempts=[],
        ),
    }
    final = task1_boundary_vlm._finalize_boundaries(
        coarse={"b1": None, "b2": None, "b3": None},
        refinements=refinements,
        is_success=True,
        episode_length=50,
    )
    task1_boundary_vlm._sync_refinements_with_final(
        coarse={"b1": None, "b2": None, "b3": None},
        refinements=refinements,
        final=final,
        is_success=True,
        episode_length=50,
    )

    assert final["b3"] == 49
    assert final["b1"] is not None
    assert final["b1"] < final["b3"]
    assert refinements["b1"].source == "default_b1_fallback"
    assert refinements["b3"].source == "success_tail_fallback"
