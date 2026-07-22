#!/usr/bin/env bash
# Independent FACD policy repeats using shared labels and Value/Fusion artifacts.
set -euo pipefail

cd /workspace/RLinf

STAGE="${1:-all}"
SEEDS="${FACD_SEEDS:-20260723 20260724}"
SHARED_EXP_ROOT="${FACD_SHARED_EXP_ROOT:-/data/libero_long/round1_v3_facd_exp}"
EXP_ROOT_PREFIX="${FACD_EXP_ROOT_PREFIX:-/data/libero_long/round1_v3_facd_repeat}"
RESULTS_ROOT_PREFIX="${FACD_RESULTS_ROOT_PREFIX:-/workspace/results/round1_v3_facd_repeat}"

case "${STAGE}" in
  train|eval|all)
    ;;
  *)
    echo "usage: $0 [train|eval|all]"
    exit 2
    ;;
esac

test -d "${SHARED_EXP_ROOT}/policy_data" || {
  echo "missing shared policy data: ${SHARED_EXP_ROOT}/policy_data"
  exit 1
}
test -d "${SHARED_EXP_ROOT}/revalue" || {
  echo "missing shared Value/Fusion artifacts: ${SHARED_EXP_ROOT}/revalue"
  exit 1
}

for seed in ${SEEDS}; do
  exp_root="${EXP_ROOT_PREFIX}_seed${seed}_exp"
  results_root="${RESULTS_ROOT_PREFIX}_seed${seed}"
  checkpoint="${results_root}/policy/policy1_facd/checkpoints/global_step_500/actor/model_state_dict/full_weights.pt"
  result="${results_root}/eval/facd_step_500.json"

  mkdir -p "${exp_root}" "${results_root}/recovery"
  if [ ! -e "${exp_root}/policy_data" ]; then
    ln -s "${SHARED_EXP_ROOT}/policy_data" "${exp_root}/policy_data"
  fi
  if [ ! -e "${exp_root}/revalue" ]; then
    ln -s "${SHARED_EXP_ROOT}/revalue" "${exp_root}/revalue"
  fi

  echo "============================================================"
  echo "FACD repeat seed=${seed} stage=${STAGE}"
  echo "exp_root=${exp_root}"
  echo "results_root=${results_root}"
  echo "============================================================"

  if [ "${STAGE}" = train ] || [ "${STAGE}" = all ]; then
    if [ -f "${checkpoint}" ]; then
      echo "reusing completed checkpoint: ${checkpoint}"
    else
      FACD_EXP_ROOT="${exp_root}" \
      FACD_RESULTS_ROOT="${results_root}" \
      POLICY_SEED="${seed}" \
      POLICY_MAX_STEPS=500 \
      EVAL_ROLLOUT_EPOCH=2 \
        bash examples/recap/rounds/run_v3_facd.sh train_policy \
        2>&1 | tee "${results_root}/recovery/train_policy.log"
    fi
  fi

  if [ "${STAGE}" = eval ] || [ "${STAGE}" = all ]; then
    test -f "${checkpoint}" || {
      echo "missing checkpoint: ${checkpoint}"
      exit 1
    }
    if [ -f "${result}" ]; then
      echo "reusing completed evaluation: ${result}"
    else
      FACD_EXP_ROOT="${exp_root}" \
      FACD_RESULTS_ROOT="${results_root}" \
      POLICY_SEED="${seed}" \
      POLICY_MAX_STEPS=500 \
      EVAL_ROLLOUT_EPOCH=2 \
        bash examples/recap/rounds/run_v3_facd.sh eval_checkpoints \
        2>&1 | tee "${results_root}/recovery/eval_checkpoints.log"
    fi
  fi
done
