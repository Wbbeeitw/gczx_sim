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

"""Data helpers for Revalue."""

from rlinf.revalue.data.feature_cache import (
    HEAD_TYPE_SHARED_MLP,
    HEAD_TYPE_TEMPORAL_LOCAL_STAGE_GATED,
    HEAD_TYPE_TEMPORAL_STAGE_PRIOR,
    HEAD_TYPE_TEMPORAL_STAGE_EXPERTS,
    HEAD_TYPE_TEMPORAL_Z_MLP_P,
    FeatureCache,
    TEMPORAL_HEAD_TYPES,
    TemporalWindowDataset,
    build_feature_loaders,
    collate_feature_batch,
)
from rlinf.revalue.data.feature_resplit import (
    FeatureResplitConfig,
    resplit_feature_cache,
)
from rlinf.revalue.data.advantage_table import (
    FeatureAdvantageDataset,
    build_fusion_loaders,
    read_advantages,
    resolve_advantage_path,
    validate_fusion_advantages,
)
from rlinf.revalue.data.phase_dataset import RevaluePhaseDataset, build_phase_datasets
from rlinf.revalue.data.episode_manifest import (
    EpisodeManifestConfig,
    EpisodeSplitSpec,
    EpisodeSubsetSpec,
    build_episode_manifest,
    load_episode_manifest,
    resolve_episode_split_for_dataset,
    resolve_episode_subset_for_dataset,
)

__all__ = [
    "EpisodeManifestConfig",
    "EpisodeSplitSpec",
    "EpisodeSubsetSpec",
    "FeatureAdvantageDataset",
    "FeatureCache",
    "FeatureResplitConfig",
    "HEAD_TYPE_SHARED_MLP",
    "HEAD_TYPE_TEMPORAL_LOCAL_STAGE_GATED",
    "HEAD_TYPE_TEMPORAL_STAGE_PRIOR",
    "HEAD_TYPE_TEMPORAL_STAGE_EXPERTS",
    "HEAD_TYPE_TEMPORAL_Z_MLP_P",
    "TEMPORAL_HEAD_TYPES",
    "TemporalWindowDataset",
    "build_episode_manifest",
    "build_feature_loaders",
    "build_fusion_loaders",
    "collate_feature_batch",
    "load_episode_manifest",
    "read_advantages",
    "RevaluePhaseDataset",
    "resolve_episode_split_for_dataset",
    "resolve_episode_subset_for_dataset",
    "resolve_advantage_path",
    "build_phase_datasets",
    "resplit_feature_cache",
    "validate_fusion_advantages",
]
