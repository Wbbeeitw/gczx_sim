import importlib.util
import json
from pathlib import Path
import sys
import types

import torch


def _load_dataset_module():
    repo_root = Path(__file__).resolve().parents[2]
    probe_dir = repo_root / "examples" / "recap" / "phase_progress_probe"
    module_path = probe_dir / "dataset.py"
    sys.path.insert(0, str(probe_dir))

    openpi_stub = types.ModuleType("openpi")
    openpi_models_stub = types.ModuleType("openpi.models")
    openpi_model_stub = types.ModuleType("openpi.models.model")
    openpi_transforms_stub = types.ModuleType("openpi.transforms")

    class ModelType:
        PI0 = "pi0"
        PI05 = "pi05"
        PI0_FAST = "pi0_fast"

    openpi_model_stub.ModelType = ModelType
    openpi_transforms_stub.RepackTransform = lambda *args, **kwargs: None
    openpi_transforms_stub.InjectDefaultPrompt = lambda *args, **kwargs: None
    openpi_transforms_stub.PadStatesAndActions = lambda *args, **kwargs: None
    openpi_transforms_stub.compose = lambda transforms_list: (lambda sample: sample)

    sys.modules["openpi"] = openpi_stub
    sys.modules["openpi.models"] = openpi_models_stub
    sys.modules["openpi.models.model"] = openpi_model_stub
    sys.modules["openpi.transforms"] = openpi_transforms_stub

    pd_stub = types.ModuleType("pandas")
    pd_stub.read_parquet = lambda *args, **kwargs: None
    sys.modules["pandas"] = pd_stub

    lerobot_stub = types.ModuleType("lerobot")
    lerobot_common_stub = types.ModuleType("lerobot.common")
    lerobot_datasets_stub = types.ModuleType("lerobot.common.datasets")
    lerobot_dataset_stub = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    lerobot_dataset_stub.LeRobotDataset = object
    lerobot_dataset_stub.LeRobotDatasetMetadata = object
    sys.modules["lerobot"] = lerobot_stub
    sys.modules["lerobot.common"] = lerobot_common_stub
    sys.modules["lerobot.common.datasets"] = lerobot_datasets_stub
    sys.modules["lerobot.common.datasets.lerobot_dataset"] = lerobot_dataset_stub

    rlinf_stub = types.ModuleType("rlinf")
    rlinf_data_stub = types.ModuleType("rlinf.data")
    rlinf_data_datasets_stub = types.ModuleType("rlinf.data.datasets")
    rlinf_data_recap_stub = types.ModuleType("rlinf.data.datasets.recap")
    recap_utils_stub = types.ModuleType("rlinf.data.datasets.recap.utils")
    recap_utils_stub.decode_image_struct_batch = lambda batch: batch
    sys.modules["rlinf"] = rlinf_stub
    sys.modules["rlinf.data"] = rlinf_data_stub
    sys.modules["rlinf.data.datasets"] = rlinf_data_datasets_stub
    sys.modules["rlinf.data.datasets.recap"] = rlinf_data_recap_stub
    sys.modules["rlinf.data.datasets.recap.utils"] = recap_utils_stub

    openpi_policies_stub = types.ModuleType("rlinf.models.embodiment.openpi.policies")
    libero_policy_stub = types.ModuleType("libero_policy")
    libero_policy_stub.LiberoInputs = lambda *args, **kwargs: None
    openpi_policies_stub.libero_policy = libero_policy_stub
    sys.modules["rlinf.models.embodiment.openpi.policies"] = openpi_policies_stub

    examples_stub = types.ModuleType("examples")
    recap_stub = types.ModuleType("examples.recap")
    process_stub = types.ModuleType("examples.recap.process")
    episode_subset_utils_stub = types.ModuleType("examples.recap.process.episode_subset_utils")

    def load_episode_subset_file(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def resolve_episode_subset_for_dataset(raw_spec, dataset_path):
        dataset_name = Path(dataset_path).name
        payload = raw_spec.get(dataset_name) or raw_spec.get(str(dataset_path))
        if payload is None:
            return None
        return types.SimpleNamespace(episodes=payload["selected_episodes"])

    episode_subset_utils_stub.load_episode_subset_file = load_episode_subset_file
    episode_subset_utils_stub.resolve_episode_subset_for_dataset = (
        resolve_episode_subset_for_dataset
    )
    sys.modules["examples"] = examples_stub
    sys.modules["examples.recap"] = recap_stub
    sys.modules["examples.recap.process"] = process_stub
    sys.modules["examples.recap.process.episode_subset_utils"] = (
        episode_subset_utils_stub
    )

    spec = importlib.util.spec_from_file_location("phase_progress_probe_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_datasets_respects_max_episodes_before_split(monkeypatch):
    module = _load_dataset_module()

    class FakeMetadata:
        def __init__(self, name, root):
            self.total_episodes = 10

    class FakeHF:
        def set_transform(self, transform):
            self.transform = transform

    class FakeDataset:
        def __init__(self, name, root, episodes, download_videos):
            self.hf_dataset = FakeHF()
            self.episode_data_index = {
                "from": torch.arange(0, 20, 2),
                "to": torch.arange(2, 22, 2),
            }

        def __getitem__(self, idx):
            raise AssertionError("not needed")

    monkeypatch.setattr(module, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(module, "_load_task_descriptions", lambda path: {})
    monkeypatch.setattr(module, "_load_phase_labels", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module.PhaseProbeDataset,
        "_build_transform",
        staticmethod(lambda **kwargs: (lambda sample: sample)),
    )

    train_ds, val_ds = module.build_datasets(
        dataset_path="/tmp/fake",
        max_episodes=5,
        val_episode_ratio=0.2,
        seed=0,
    )

    assert len(train_ds._indices) == 8
    assert len(val_ds._indices) == 2


def test_build_datasets_respects_explicit_episode_subset(monkeypatch, tmp_path):
    module = _load_dataset_module()

    class FakeMetadata:
        def __init__(self, name, root):
            self.total_episodes = 10

    class FakeHF:
        def set_transform(self, transform):
            self.transform = transform

    class FakeDataset:
        def __init__(self, name, root, episodes, download_videos):
            self.hf_dataset = FakeHF()
            self.episode_data_index = {
                "from": torch.arange(0, 20, 2),
                "to": torch.arange(2, 22, 2),
            }

        def __getitem__(self, idx):
            raise AssertionError("not needed")

    subset_path = tmp_path / "subset.json"
    subset_path.write_text(
        json.dumps(
            {
                "fake": {
                    "selected_episodes": [1, 3, 5, 7],
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(module, "_load_task_descriptions", lambda path: {})
    monkeypatch.setattr(module, "_load_phase_labels", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module.PhaseProbeDataset,
        "_build_transform",
        staticmethod(lambda **kwargs: (lambda sample: sample)),
    )

    train_ds, val_ds = module.build_datasets(
        dataset_path="/tmp/fake",
        episode_subset_path=str(subset_path),
        val_episode_ratio=0.25,
        seed=0,
    )

    assert len(train_ds._indices) == 6
    assert len(val_ds._indices) == 2
