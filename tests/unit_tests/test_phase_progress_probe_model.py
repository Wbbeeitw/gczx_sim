import importlib.util
from pathlib import Path
import sys
import types

import torch


def _load_probe_model_module():
    repo_root = Path(__file__).resolve().parents[2]
    probe_dir = repo_root / "examples" / "recap" / "phase_progress_probe"
    module_path = probe_dir / "model.py"
    sys.path.insert(0, str(probe_dir))

    transformers_stub = types.ModuleType("transformers")
    cache_utils_stub = types.ModuleType("transformers.cache_utils")

    class DynamicCache:
        pass

    cache_utils_stub.DynamicCache = DynamicCache
    transformers_stub.cache_utils = cache_utils_stub
    sys.modules["transformers"] = transformers_stub
    sys.modules["transformers.cache_utils"] = cache_utils_stub

    rlinf_stub = types.ModuleType("rlinf")
    rlinf_models_stub = types.ModuleType("rlinf.models")
    rlinf_embodiment_stub = types.ModuleType("rlinf.models.embodiment")
    rlinf_value_model_stub = types.ModuleType("rlinf.models.embodiment.value_model")
    rlinf_value_model_stub.ValueCriticModel = object

    sys.modules["rlinf"] = rlinf_stub
    sys.modules["rlinf.models"] = rlinf_models_stub
    sys.modules["rlinf.models.embodiment"] = rlinf_embodiment_stub
    sys.modules["rlinf.models.embodiment.value_model"] = rlinf_value_model_stub

    spec = importlib.util.spec_from_file_location("phase_progress_probe_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_prefix_features_uses_preprocessed_image_list():
    module = _load_probe_model_module()
    extractor_cls = module.VLMBackboneFeatureExtractor

    class FakeGemma3Config:
        hidden_size = 8

    class FakeGemma3:
        config = FakeGemma3Config()

    class FakeValueExpert:
        gemma3 = FakeGemma3()

        def forward(
            self,
            *,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
        ):
            prefix_embs = inputs_embeds[0]
            return (prefix_embs, None), None

    class FakeValueModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.value_expert = FakeValueExpert()
            self.preprocess_called = False
            self.embed_prefix_arg = None

        def eval(self):
            return self

        def _preprocess_observation(self, observation):
            self.preprocess_called = True
            return (
                [torch.ones(2, 3, 4, 4)],
                [torch.ones(2, dtype=torch.bool)],
                torch.ones(2, 5, dtype=torch.long),
                torch.ones(2, 5, dtype=torch.bool),
                None,
                None,
            )

        def embed_prefix(self, images, image_masks, lang_tokens, lang_masks):
            self.embed_prefix_arg = images
            prefix_embs = torch.ones(2, 6, 8, dtype=torch.float32)
            prefix_pad_masks = torch.ones(2, 6, dtype=torch.bool)
            return prefix_embs, prefix_pad_masks

        def _get_model_dtype(self):
            return torch.float32

    fake_model = FakeValueModel()
    extractor = extractor_cls(fake_model)

    features = extractor.extract_prefix_features(
        {
            "images": {"base_0_rgb": torch.zeros(2, 3, 4, 4)},
            "image_masks": {"base_0_rgb": torch.ones(2, dtype=torch.bool)},
            "tokenized_prompt": torch.ones(2, 5, dtype=torch.long),
            "tokenized_prompt_mask": torch.ones(2, 5, dtype=torch.bool),
        }
    )

    assert fake_model.preprocess_called is True
    assert isinstance(fake_model.embed_prefix_arg, list)
    assert len(fake_model.embed_prefix_arg) == 1
    assert features.shape == (2, 8)
