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

from phase_split_script import task1_vlm


def test_build_prompt_mentions_combined_manipulation_and_optional_skips():
    prompt = task1_vlm._build_prompt(
        task_description=task1_vlm.TASK_DESCRIPTION,
        phase_definitions=task1_vlm.PHASE_DEFINITIONS,
        episode_length=520,
        duration_seconds=52.0,
        sample_fps=0.5,
        is_success=None,
        enable_reasoning=False,
    )

    assert "combine substeps" in prompt
    assert "drop one object near the other" in prompt
    assert "Not every phase must appear" in prompt
    assert "Phase 2 should be used only when a first stable single-object basket deposit" in prompt


def test_compose_multiview_frame_adds_header_and_wrist_panel():
    main = Image.new("RGB", (32, 24), color=(255, 0, 0))
    wrist = Image.new("RGB", (16, 12), color=(0, 255, 0))

    composed = task1_vlm._compose_multiview_frame(
        main_frame=main,
        wrist_frame=wrist,
        frame_index=10,
        timestamp_seconds=2.0,
    )

    assert composed.size[0] > main.size[0]
    assert composed.size[1] > main.size[1]
    assert composed.getpixel((1, 1)) == (0, 0, 0)


def test_build_sampled_images_preserves_sampling_order():
    frames = [
        Image.new("RGB", (20, 20), color=(i, i, i))
        for i in (10, 20, 30, 40)
    ]
    sampled = task1_vlm._build_sampled_images(
        main_frames=frames,
        wrist_frames=None,
        sampled_indices=[0, 2, 3],
        fps=10.0,
    )

    assert len(sampled) == 3
    assert all(img.size[1] > frames[0].size[1] for img in sampled)


def test_build_episode_df_keeps_nullable_success_flag():
    phase = np.array([0, 0, 1, 1], dtype=int)
    df = task1_vlm._build_episode_df(episode_index=3, phase=phase, is_success=False)

    assert "is_success" in df.columns
    assert df["is_success"].dtype.name == "boolean"
    assert df["is_success"].tolist() == [False, False, False, False]
