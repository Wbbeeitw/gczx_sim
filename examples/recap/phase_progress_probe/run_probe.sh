#!/usr/bin/env bash
# One-click runner for the phase/progress probe experiment.

set -euo pipefail

# Activate the OpenPI virtual environment.
source /opt/venv/openpi/bin/activate

cd /workspace/RLinf

python examples/recap/phase_progress_probe/run_probe.py "$@"
