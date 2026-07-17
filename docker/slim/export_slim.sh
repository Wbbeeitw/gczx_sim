#!/bin/bash
# Extract the openpi venv, uv-managed CPython and /opt/assets from the full
# migration image into ./slim_export (build context for the slim Dockerfile).
#
# Usage (on the GPU host):
#   bash docker/slim/export_slim.sh [image-name]
#
# Default image: rlinf-openpi-migration-raw:20260713
set -euo pipefail

IMAGE="${1:-rlinf-openpi-migration-raw:20260713}"
OUT="$(cd "$(dirname "$0")" && pwd)/slim_export"

mkdir -p "$OUT"
docker run --rm -v "$OUT":/export "$IMAGE" bash -c '
    set -e
    cp -a /opt/venv/openpi /export/openpi
    cp -a /opt/venv/.python /export/.python
    cp -a /opt/assets /export/assets
'

du -sh "$OUT"/*
echo "slim_export ready at $OUT"
