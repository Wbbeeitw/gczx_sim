#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_PATH}"

source switch_env openpi 2>/dev/null || true
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

DATASET="${DATASET:-/workspace/datasets/recap_libero10_task0/libero10_task0_train}"
SPLIT="${SPLIT:-/workspace/results/revalue/first200_split160_40.json}"
REVALUE_ROOT="${REVALUE_ROOT:-/workspace/results/revalue/first200_vm5k_rollout}"
MODEL="${MODEL:-/workspace/models/RLinf-Pi05-LIBERO-SFT}"

# Keep the user's documented BASE_TAG/FUSED_TAG untouched. These explicit tags
# point to the fixed-scale base/fused advantages used for the actual CFG compare.
BASE_ADV_TAG="${BASE_ADV_TAG:-base_rollout_first200_vm5k_q30_fixscale}"
FUSED_ADV_TAG="${FUSED_ADV_TAG:-fused_rollout_first200_vm5k_q30_fixscale}"

EPISODE_SPLIT_NAME="${EPISODE_SPLIT_NAME:-train}"
EXPORT_SPLIT="${EXPORT_SPLIT:-${EPISODE_SPLIT_NAME}}"

CFG_LOG_ROOT="${CFG_LOG_ROOT:-${REVALUE_ROOT}/downstream_train}"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-cfg_base_first200_fixscale}"
FUSED_EXPERIMENT_NAME="${FUSED_EXPERIMENT_NAME:-cfg_fused_first200_fixscale}"

MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${MAX_STEPS}}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:--1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${MAX_STEPS}}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"

FORCE_EXPORT_FUSED="${FORCE_EXPORT_FUSED:-false}"

BASE_ADV_PATH="${DATASET}/meta/advantages_${BASE_ADV_TAG}.parquet"
FUSED_ADV_PATH="${DATASET}/meta/advantages_${FUSED_ADV_TAG}.parquet"

EXTRA_OVERRIDES=("$@")

echo "Repo: ${REPO_PATH}"
echo "Dataset: ${DATASET}"
echo "Split manifest: ${SPLIT}"
echo "Revalue root: ${REVALUE_ROOT}"
echo "Model: ${MODEL}"
echo "Base advantage tag: ${BASE_ADV_TAG}"
echo "Fused advantage tag: ${FUSED_ADV_TAG}"
echo "Episode split: ${EPISODE_SPLIT_NAME}"
echo "Max steps: ${MAX_STEPS}"
echo

if [[ ! -f "${BASE_ADV_PATH}" ]]; then
    echo "Missing base advantage parquet: ${BASE_ADV_PATH}" >&2
    exit 1
fi

if [[ "${FORCE_EXPORT_FUSED}" == "true" || ! -f "${FUSED_ADV_PATH}" ]]; then
    echo "[1/3] Exporting fused advantage tag: ${FUSED_ADV_TAG}"
    python examples/recap/revalue/revalue.py \
        --config-name revalue_shared_mlp_fusion \
        stage=export \
        data.dataset_path="${DATASET}" \
        output.root="${REVALUE_ROOT}" \
        base.tag="${BASE_ADV_TAG}" \
        recap.output_tag="${FUSED_ADV_TAG}" \
        recap.export_split="${EXPORT_SPLIT}"
else
    echo "[1/3] Reusing existing fused advantage parquet: ${FUSED_ADV_PATH}"
fi

run_cfg_train() {
    local advantage_tag="$1"
    local experiment_name="$2"
    local log_dir="$3"

    echo
    echo "Running CFG training:"
    echo "  advantage_tag=${advantage_tag}"
    echo "  experiment_name=${experiment_name}"
    echo "  log_dir=${log_dir}"

    python examples/recap/revalue/revalue.py \
        stage=train_cfg \
        cfg_train.enabled=true \
        cfg_train.dataset_path="${DATASET}" \
        cfg_train.base_model_path="${MODEL}" \
        cfg_train.advantage_tag="${advantage_tag}" \
        cfg_train.episode_split_path="${SPLIT}" \
        cfg_train.episode_split_name="${EPISODE_SPLIT_NAME}" \
        cfg_train.experiment_name="${experiment_name}" \
        cfg_train.log_dir="${log_dir}" \
        cfg_train.model_type=cfg_model \
        cfg_train.openpi_config_name=pi05_libero \
        cfg_train.guidance_type=positive \
        cfg_train.positive_only_conditional=true \
        cfg_train.max_epochs=-1 \
        cfg_train.max_steps="${MAX_STEPS}" \
        cfg_train.save_interval="${SAVE_INTERVAL}" \
        cfg_train.val_check_interval="${VAL_CHECK_INTERVAL}" \
        cfg_train.total_training_steps="${TOTAL_TRAINING_STEPS}" \
        cfg_train.lr_warmup_steps="${LR_WARMUP_STEPS}" \
        cfg_train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
        cfg_train.micro_batch_size="${MICRO_BATCH_SIZE}" \
        "${EXTRA_OVERRIDES[@]}"
}

echo "[2/3] Training base CFG"
run_cfg_train \
    "${BASE_ADV_TAG}" \
    "${BASE_EXPERIMENT_NAME}" \
    "${CFG_LOG_ROOT}/base"

echo
echo "[3/3] Training fused CFG"
run_cfg_train \
    "${FUSED_ADV_TAG}" \
    "${FUSED_EXPERIMENT_NAME}" \
    "${CFG_LOG_ROOT}/fused"

echo
echo "Finished."
echo "Base summary: ${CFG_LOG_ROOT}/base/train_cfg_summary.json"
echo "Fused summary: ${CFG_LOG_ROOT}/fused/train_cfg_summary.json"
