# Slim RLinf OpenPI runtime.
#
# The full migration image (rlinf-openpi-migration-raw) ships 8 model venvs
# plus the uv cache (~129 GB uncompressed). The ReCap pipeline only uses
# /opt/venv/openpi (~9.9 GB), so this image repacks just that venv, the
# uv-managed CPython and /opt/assets on top of a small CUDA runtime base.
#
# Build context layout (produced by export_slim.sh):
#   Dockerfile
#   slim_export/
#     openpi/     (copy of /opt/venv/openpi)
#     .python/    (copy of /opt/venv/.python)
#     assets/     (copy of /opt/assets)
#
# Build:  docker build -t rlinf-openpi-slim:20260717 <context dir>

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git openssh-server libgl1 libegl1 libgles2 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY slim_export/.python /opt/venv/.python
COPY slim_export/openpi /opt/venv/openpi
COPY slim_export/assets /opt/assets

RUN printf '%s\n' '#!/bin/bash' \
    'if [ -z "$1" ]; then echo "Usage: switch_env <env_name>"; exit 1; fi' \
    'if [ ! -d "/opt/venv/$1" ]; then echo "Environment $1 does not exist."; exit 1; fi' \
    'source /opt/venv/$1/bin/activate' \
    > /usr/local/bin/switch_env && chmod +x /usr/local/bin/switch_env

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    TOKENIZERS_PARALLELISM=false \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/workspace/hf_home \
    NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace

CMD ["/bin/bash"]
