#!/bin/bash
# Launch the multitask container on the Pro 6000 host (enine).
# Usage: bash docker/run_mt.sh
set -euo pipefail

docker rm -f rlinf_mt 2>/dev/null || true

docker run -it \
  --name rlinf_mt \
  --gpus all \
  --ipc=host \
  --network=host \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HOME=/workspace/hf_home \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -v /home/enine/rlinf_workspace/RLinf:/workspace/RLinf \
  -v /home/enine/rlinf_workspace/models:/workspace/models \
  -v /home/enine/rlinf_workspace/hf_home:/workspace/hf_home \
  -v /data/libero_long:/data/libero_long \
  -v /data/results:/workspace/results \
  rlinf-openpi-migration-raw:20260713 \
  bash
