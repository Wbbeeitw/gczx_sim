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

"""Constants used by the Revalue pipeline."""

DEFAULT_NUM_BINS = 201
DEFAULT_NUM_PHASES = 5
DEFAULT_VALUE_MIN = -1.0
DEFAULT_VALUE_MAX = 0.0

METHOD_BASE = "base"
METHOD_SHARED_MLP_FUSION = "shared_mlp_fusion"

STAGE_ALL = "all"
STAGE_EXTRACT_FEATURES = "extract_features"
STAGE_TRAIN_ZP = "train_zp"
STAGE_TRAIN_FUSION = "train_fusion"
STAGE_PREDICT = "predict"
STAGE_EXPORT = "export"

SUPPORTED_METHODS = (METHOD_BASE, METHOD_SHARED_MLP_FUSION)
SUPPORTED_STAGES = (
    STAGE_ALL,
    STAGE_EXTRACT_FEATURES,
    STAGE_TRAIN_ZP,
    STAGE_TRAIN_FUSION,
    STAGE_PREDICT,
    STAGE_EXPORT,
)
