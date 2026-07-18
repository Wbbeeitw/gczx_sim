#!/usr/bin/env python
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

"""Print per-task replay success counts from semantic trace audit files."""

from pathlib import Path

import pandas as pd

BASE = Path("/data/libero_long/libero_task58_replayed/meta")

for task_id in [5, 8]:
    audit = pd.read_csv(BASE / f"semantic_trace_task{task_id}_audit.csv")
    episodes = len(audit)
    successes = int(audit["is_success"].astype(bool).sum())
    print(f"task{task_id}: episodes={episodes}, replayed_success={successes}")
