#!/bin/bash
# One-shot launcher for task5/8 BC boost SFT on the single-GPU 6000 box.
# Exists only because the terminal mangles pasted long commands.
set -e
cd /workspace/RLinf
source switch_env openpi
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export REPO_PATH=/workspace/RLinf
export PYTHONPATH=/workspace/RLinf
unset RAY_ADDRESS
exec python examples/sft/train_vla_sft.py --config-name task58_bc_openpi_pi05
