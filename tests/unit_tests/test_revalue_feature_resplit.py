import torch

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.data.feature_resplit import (  # noqa: E402
    FeatureResplitConfig,
    resplit_feature_cache,
)


def _cache(episodes: list[int]) -> dict[str, torch.Tensor]:
    episode_index = []
    frame_index = []
    for ep in episodes:
        for frame in range(2):
            episode_index.append(ep)
            frame_index.append(frame)
    rows = len(episode_index)
    return {
        "features": torch.arange(rows * 3, dtype=torch.float32).view(rows, 3),
        "episode_index": torch.tensor(episode_index, dtype=torch.long),
        "frame_index": torch.tensor(frame_index, dtype=torch.long),
        "phase": torch.zeros(rows, dtype=torch.long),
        "phase_progress": torch.zeros(rows, dtype=torch.float32),
        "global_progress": torch.zeros(rows, dtype=torch.float32),
    }


def test_resplit_feature_cache_merges_existing_train_val(tmp_path) -> None:
    src = tmp_path / "old"
    dst = tmp_path / "new"
    src.mkdir()
    torch.save(_cache([0, 1, 2]), src / "train.pt")
    torch.save(_cache([3, 4]), src / "val.pt")

    report = resplit_feature_cache(
        FeatureResplitConfig(
            source_dir=str(src),
            output_dir=str(dst),
            val_episode_ratio=0.2,
            test_episode_ratio=0.2,
            seed=1,
        )
    )

    assert (dst / "train.pt").exists()
    assert (dst / "val.pt").exists()
    assert (dst / "test.pt").exists()
    assert sum(report["split_rows"].values()) == 10

    train = torch.load(dst / "train.pt", map_location="cpu", weights_only=False)
    val = torch.load(dst / "val.pt", map_location="cpu", weights_only=False)
    test = torch.load(dst / "test.pt", map_location="cpu", weights_only=False)
    train_eps = set(train["episode_index"].tolist())
    val_eps = set(val["episode_index"].tolist())
    test_eps = set(test["episode_index"].tolist())

    assert train_eps.isdisjoint(val_eps)
    assert train_eps.isdisjoint(test_eps)
    assert val_eps.isdisjoint(test_eps)
    assert train_eps | val_eps | test_eps == {0, 1, 2, 3, 4}
