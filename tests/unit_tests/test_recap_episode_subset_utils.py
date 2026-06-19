import json

from examples.recap.process.episode_subset_utils import (
    compute_frame_indices_for_episodes,
    load_episode_subset_file,
    resolve_episode_subset_for_dataset,
    sample_balanced_episode_ids,
)


def test_compute_frame_indices_for_episodes() -> None:
    episode_to = [3, 5, 9]
    indices = compute_frame_indices_for_episodes(episode_to, [0, 2])
    assert indices == [0, 1, 2, 5, 6, 7, 8]


def test_sample_balanced_episode_ids_prefers_both_outcomes() -> None:
    episode_success = {
        0: True,
        1: True,
        2: True,
        3: False,
        4: False,
        5: False,
    }
    selected = sample_balanced_episode_ids(
        episode_success=episode_success,
        num_episodes=4,
        seed=123,
        success_ratio=0.5,
    )
    assert len(selected) == 4
    selected_success = sum(episode_success[ep] for ep in selected)
    assert selected_success == 2


def test_resolve_episode_subset_for_dataset_by_name(tmp_path) -> None:
    payload = {
        "libero10_task0_train": {
            "selected_episodes": [5, 1, 1, 3],
            "num_frames": 42,
        }
    }
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    raw = load_episode_subset_file(path)
    spec = resolve_episode_subset_for_dataset(
        raw,
        "/workspace/datasets/recap_libero10_task0/libero10_task0_train",
    )
    assert spec is not None
    assert spec.episodes == [1, 3, 5]
    assert spec.num_frames == 42
