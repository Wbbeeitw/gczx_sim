#!/bin/bash
# Pure policy evaluation loop over LIBERO task ids (no video, no dataset).
#
# Usage:
#   bash examples/recap/rounds/eval_policy_loop.sh [MODEL_PATH] [TASK_IDS...]
#
# Examples:
#   bash examples/recap/rounds/eval_policy_loop.sh
#   bash examples/recap/rounds/eval_policy_loop.sh /workspace/models/RLinf-Pi05-LIBERO-SFT 0 1 2
#
# Results land in <log_root>/task<T>/eval_policy_summary.json
set -u

MODEL_PATH="${1:-/workspace/models/RLinf-Pi05-LIBERO-SFT}"
if [ $# -ge 1 ]; then shift; fi
TASK_IDS=("$@")
if [ ${#TASK_IDS[@]} -eq 0 ]; then
  TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
fi

cd "$(dirname "$0")/../../.." || exit 1
source switch_env openpi
export REPO_PATH="$PWD" PYTHONPATH="$PWD"
unset RAY_ADDRESS || true

LOG_ROOT="${LOG_ROOT:-/workspace/results/sft_base_probe}"
NUM_ENVS="${NUM_ENVS:-4}"
EPOCHS="${EPOCHS:-2}"
mkdir -p "$LOG_ROOT"

for T in "${TASK_IDS[@]}"; do
  python examples/recap/revalue/revalue.py \
    --config-name revalue_shared_mlp_fusion \
    stage=eval_policy \
    policy_eval.enabled=true \
    policy_eval.model_path="$MODEL_PATH" \
    policy_eval.model_type=openpi \
    policy_eval.experiment_name="sft_base_probe_task$T" \
    policy_eval.log_dir="$LOG_ROOT/task$T" \
    policy_eval.config_name=libero_10_pi05_sft_eval \
    policy_eval.openpi_config_name=pi05_libero \
    policy_eval.eval_rollout_epoch="$EPOCHS" \
    policy_eval.total_num_envs="$NUM_ENVS" \
    policy_eval.task_suite_name=libero_10 \
    policy_eval.task_id_filter="[$T]" \
    policy_eval.save_video=false \
    policy_eval.warmup_before_env=true \
    output.root="$LOG_ROOT"
done

echo "== success rates =="
for T in "${TASK_IDS[@]}"; do
  printf "task%s: " "$T"
  grep -o '"eval/success_rate": [0-9.]*' "$LOG_ROOT/task$T/eval_policy_summary.json" 2>/dev/null || echo "missing"
done
