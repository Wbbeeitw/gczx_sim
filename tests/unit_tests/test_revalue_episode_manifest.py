import json

import pandas as pd

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.data.episode_manifest import (  # noqa: E402
    EpisodeManifestConfig,
    build_episode_manifest,
    load_episode_manifest,
    resolve_episode_split_for_dataset,
)


def test_build_episode_manifest_samples_and_splits(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    rows = []
    for ep in range(10):
        max_phase = 4 if ep % 2 == 0 else 2
        for frame in range(3):
            rows.append(
                {
                    "episode_index": ep,
                    "frame_index": frame,
                    "phase": max_phase if frame == 2 else 0,
                }
            )
    pd.DataFrame(rows).to_parquet(meta / "phase_progress_semantic.parquet")

    out_path = tmp_path / "manifest.json"
    build_episode_manifest(
        EpisodeManifestConfig(
            dataset_path=str(dataset),
            output_path=str(out_path),
            num_episodes=6,
            success_ratio=0.5,
            val_episode_ratio=0.2,
            test_episode_ratio=0.2,
            seed=7,
        )
    )

    raw = load_episode_manifest(out_path)
    spec = resolve_episode_split_for_dataset(raw, dataset)

    assert spec is not None
    assert len(spec.selected_episodes) == 6
    assert set(spec.train_episodes).isdisjoint(spec.val_episodes)
    assert set(spec.train_episodes).isdisjoint(spec.test_episodes)
    assert set(spec.val_episodes).isdisjoint(spec.test_episodes)
    assert set(spec.train_episodes) | set(spec.val_episodes) | set(
        spec.test_episodes
    ) == set(spec.selected_episodes)

    payload = next(iter(raw.values()))
    assert payload["num_success"] == 3
    assert payload["num_failure"] == 3
    assert payload["num_frames"] == 18


def test_build_episode_manifest_does_not_overwrite_by_default(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    pd.DataFrame(
        [{"episode_index": 0, "frame_index": 0, "phase": 4}]
    ).to_parquet(meta / "phase_progress_semantic.parquet")
    out_path = tmp_path / "manifest.json"
    out_path.write_text(json.dumps({"sentinel": True}), encoding="utf-8")

    build_episode_manifest(
        EpisodeManifestConfig(
            dataset_path=str(dataset),
            output_path=str(out_path),
            num_episodes=1,
        )
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == {"sentinel": True}
