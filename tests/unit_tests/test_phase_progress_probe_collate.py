import importlib.util
from pathlib import Path
import sys
import types

import torch


def _load_extract_features_module():
    repo_root = Path(__file__).resolve().parents[2]
    probe_dir = repo_root / "examples" / "recap" / "phase_progress_probe"
    module_path = (
        probe_dir / "extract_features.py"
    )
    sys.path.insert(0, str(probe_dir))

    model_stub = types.ModuleType("model")
    model_stub.VLMBackboneFeatureExtractor = object
    sys.modules["model"] = model_stub

    dataset_stub = types.ModuleType("dataset")
    dataset_stub.build_datasets = lambda *args, **kwargs: None
    sys.modules["dataset"] = dataset_stub

    rlinf_stub = types.ModuleType("rlinf")
    rlinf_models_stub = types.ModuleType("rlinf.models")
    rlinf_embodiment_stub = types.ModuleType("rlinf.models.embodiment")
    rlinf_value_model_stub = types.ModuleType("rlinf.models.embodiment.value_model")
    rlinf_value_model_stub.ValueCriticModel = object

    sys.modules["rlinf"] = rlinf_stub
    sys.modules["rlinf.models"] = rlinf_models_stub
    sys.modules["rlinf.models.embodiment"] = rlinf_embodiment_stub
    sys.modules["rlinf.models.embodiment.value_model"] = rlinf_value_model_stub

    spec = importlib.util.spec_from_file_location(
        "phase_progress_probe_extract_features", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probe_collate_keeps_transformed_samples_as_list():
    module = _load_extract_features_module()
    collate_fn = module._collate_fn

    sample_a = {
        "episode_index": 1,
        "frame_index": 10,
        "phase": 2,
        "phase_progress": 0.25,
        "global_progress": 0.4,
        "transformed": {
            "prompt": "pick the block",
            "image": {
                "base_0_rgb": torch.zeros(4, 4, 3, dtype=torch.uint8),
                "left_wrist_0_rgb": torch.ones(4, 4, 3, dtype=torch.uint8),
                "right_wrist_0_rgb": torch.zeros(4, 4, 3, dtype=torch.uint8),
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            },
            "state": torch.zeros(8),
            "actions": torch.zeros(32),
        },
    }
    sample_b = {
        "episode_index": 1,
        "frame_index": 11,
        "phase": 2,
        "phase_progress": 0.5,
        "global_progress": 0.5,
        "transformed": {
            "prompt": "pick the block",
            "image": {
                "base_0_rgb": torch.full((4, 4, 3), 2, dtype=torch.uint8),
                "left_wrist_0_rgb": torch.full((4, 4, 3), 3, dtype=torch.uint8),
                "right_wrist_0_rgb": torch.zeros(4, 4, 3, dtype=torch.uint8),
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            },
            "state": torch.ones(8),
            "actions": torch.ones(32),
        },
    }

    batch = collate_fn([sample_a, sample_b])

    assert isinstance(batch["transformed"], list)
    assert len(batch["transformed"]) == 2
    assert batch["transformed"][0]["prompt"] == "pick the block"
    assert batch["transformed"][1]["image"]["base_0_rgb"].shape == (4, 4, 3)
    assert torch.equal(
        batch["labels"]["frame_index"], torch.tensor([10, 11], dtype=torch.long)
    )
