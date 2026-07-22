#!/usr/bin/env bash
# Fixed-data, fixed-Value ReCap raw-advantage baseline for round1 v3.
set -euo pipefail

cd /workspace/RLinf
source switch_env openpi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export REPO_PATH=/workspace/RLinf
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export HF_LEROBOT_HOME=/data/libero_long
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset RAY_ADDRESS

STAGE="${1:-all}"
TASKS=(task0 task1 task2 task3 task4 task5 task6 task7 task8 task9)

BASE_MODEL="${BASE_MODEL:-/workspace/models/RLinf-Pi05-PPO-LIBERO-130}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/data/libero_long/round1_v3_ppo_rollout_20ep_retry1}"
EXPERT_ROOT="${EXPERT_ROOT:-/data/libero_long/round1_v3_ppo_expert_success10}"
TASK_POOL_ROOT="${TASK_POOL_ROOT:-/data/libero_long/round1_v3_facd_30ep}"
MERGED_POOL="${MERGED_POOL:-/data/libero_long/round1_v3_facd_multitask_300ep}"
VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-/workspace/results/round1_v3_facd/value_sft/value_round1_v3_facd_multitask_300ep/checkpoints/global_step_1200}"
SOURCE_FEATURE_MANIFEST="${SOURCE_FEATURE_MANIFEST:-/data/libero_long/round1_v3_facd_exp/revalue/features/manifest.json}"

SOURCE_RETURNS_TAG="${SOURCE_RETURNS_TAG:-round1_v3_facd_returns}"
SOURCE_RAW_ADVANTAGE_TAG="${SOURCE_RAW_ADVANTAGE_TAG:-round1_v3_facd_base_adv}"
BASELINE_ADVANTAGE_TAG="${BASELINE_ADVANTAGE_TAG:-round1_v3_recap_raw_top30}"

EXP_ROOT="${EXP_ROOT:-/data/libero_long/round1_v3_recap_baseline_exp}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results/round1_v3_recap_baseline}"
REPORT_ROOT="${REPORT_ROOT:-${RESULTS_ROOT}/audit}"

POLICY_MAX_STEPS="${POLICY_MAX_STEPS:-1000}"
POLICY_SAVE_INTERVAL="${POLICY_SAVE_INTERVAL:-500}"
POLICY_MICRO_BATCH_SIZE="${POLICY_MICRO_BATCH_SIZE:-16}"
POLICY_GLOBAL_BATCH_SIZE="${POLICY_GLOBAL_BATCH_SIZE:-64}"
EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-2}"
POSITIVE_QUANTILE="${POSITIVE_QUANTILE:-0.3}"

SOURCE_RAW_ADVANTAGES="${MERGED_POOL}/meta/advantages_${SOURCE_RAW_ADVANTAGE_TAG}.parquet"
POLICY_DATA_REPORT="${EXP_ROOT}/policy_data/multitask_policy_data_report.json"
POLICY_MANIFEST="${EXP_ROOT}/policy_data/multitask_episode_manifest.json"
RAW_AUDIT_REPORT="${REPORT_ROOT}/recap_raw_policy_data_audit.json"
TASK_POOL_AUDIT="${REPORT_ROOT}/task_pool_quality_audit.json"
MERGED_POOL_AUDIT="${REPORT_ROOT}/merged_pool_quality_audit.json"
RAW_COMPARISON="${EXP_ROOT}/revalue/return_compare.json"

task_dataset_args() {
  for task in "${TASKS[@]}"; do
    printf '%s\n' --task-dataset "${task}=${TASK_POOL_ROOT}/${task}_30ep"
  done
}

round_common_overrides() {
  printf '%s\n' \
    --set 'tasks=["task0","task1","task2","task3","task4","task5","task6","task7","task8","task9"]' \
    --set method=recap_baseline \
    --set "paths.base_model=${BASE_MODEL}" \
    --set "paths.parent_dataset=${MERGED_POOL}" \
    --set "paths.child_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
    --set "paths.policy_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
    --set "paths.merged_dataset=${MERGED_POOL}" \
    --set "paths.value_checkpoint=${VALUE_CHECKPOINT}" \
    --set "paths.exp_root=${EXP_ROOT}" \
    --set "paths.results_root=${RESULTS_ROOT}" \
    --set "gpu.cuda_visible_devices=${CUDA_VISIBLE_DEVICES}" \
    --set gpu.gpu_id=0 \
    --set datasets.parent_episodes=0 \
    --set datasets.prebuilt_merged=true \
    --set collect.num_episodes=30 \
    --set parent_policy.checkpoint= \
    --set parent_policy.label=PPO-LIBERO-130 \
    --set "tags.returns=${SOURCE_RETURNS_TAG}" \
    --set "tags.merged_base=${SOURCE_RAW_ADVANTAGE_TAG}" \
    --set "tags.child_fused=${BASELINE_ADVANTAGE_TAG}" \
    --set returns.failure_reward=-300.0 \
    --set returns.gamma=1.0 \
    --set returns.global_min=-900.0 \
    --set returns.global_max=0.0 \
    --set "revalue.positive_quantile=${POSITIVE_QUANTILE}" \
    --set revalue.success_gate=false \
    --set revalue.demo_backstop=false \
    --set policy.advantage_source=raw \
    --set policy.strategy=binary \
    --set policy.guidance_type=positive \
    --set policy.positive_only_conditional=true \
    --set policy.negative_guidance_scale=0.0 \
    --set policy.unconditional_prob=0.1 \
    --set "policy.positive_quantile=${POSITIVE_QUANTILE}"
}

run_round_stage() {
  local pipeline="$1"
  shift
  mapfile -t common < <(round_common_overrides)
  python examples/recap/rounds/run_round.py \
    --config examples/recap/rounds/config/multitask_round1_prcfg.yaml \
    --stage all \
    --set "pipeline=${pipeline}" \
    "${common[@]}" \
    "$@"
}

check_inputs() {
  test -f "${BASE_MODEL}/model.safetensors" || {
    echo "missing PPO base model weights: ${BASE_MODEL}/model.safetensors"
    exit 1
  }
  test -f "${VALUE_CHECKPOINT}/actor/model_state_dict/full_weights.pt" || {
    echo "missing complete Value checkpoint:"
    echo "${VALUE_CHECKPOINT}/actor/model_state_dict/full_weights.pt"
    echo "Set VALUE_CHECKPOINT to the shared round1 v3 Value checkpoint."
    exit 1
  }
  test -f "${MERGED_POOL}/meta/info.json" || {
    echo "missing merged 300ep pool: ${MERGED_POOL}"
    exit 1
  }

  for task in "${TASKS[@]}"; do
    task_pool="${TASK_POOL_ROOT}/${task}_30ep"
    test -f "${task_pool}/meta/info.json" || {
      echo "missing task pool: ${task_pool}"
      exit 1
    }
    test -f "${task_pool}/meta/episode_provenance.jsonl" || {
      echo "missing provenance: ${task_pool}/meta/episode_provenance.jsonl"
      exit 1
    }
    test -f "${task_pool}/meta/full_positive_episodes.json" || {
      echo "missing expert episode metadata: ${task_pool}/meta/full_positive_episodes.json"
      exit 1
    }
    episodes="$(wc -l < "${task_pool}/meta/episodes.jsonl")"
    if [ "${episodes}" -ne 30 ]; then
      echo "${task}: expected 30 episodes, found ${episodes}"
      exit 1
    fi
  done
}

audit_shared_datasets() {
  mkdir -p "${REPORT_ROOT}"
  mapfile -t dataset_args < <(task_dataset_args)

  python examples/recap/process/audit_v3_facd_datasets.py task-pools \
    "${dataset_args[@]}" \
    --rollout-root "${ROLLOUT_ROOT}" \
    --expert-root "${EXPERT_ROOT}" \
    --output-path "${TASK_POOL_AUDIT}"

  python examples/recap/process/audit_v3_facd_datasets.py merged \
    "${dataset_args[@]}" \
    --merged-dataset "${MERGED_POOL}" \
    --output-path "${MERGED_POOL_AUDIT}"
}

score_raw_if_missing() {
  check_inputs
  if [ -f "${SOURCE_RAW_ADVANTAGES}" ]; then
    echo "reusing shared raw Value advantages: ${SOURCE_RAW_ADVANTAGES}"
    return
  fi

  echo "shared raw advantages are missing; scoring the same 300ep pool"
  echo "with the fixed Value checkpoint: ${VALUE_CHECKPOINT}"
  run_round_stage '["score_critic_multitask"]'
  test -f "${SOURCE_RAW_ADVANTAGES}" || {
    echo "raw Value scoring did not create: ${SOURCE_RAW_ADVANTAGES}"
    exit 1
  }
}

export_raw_labels() {
  check_inputs
  test -f "${SOURCE_RAW_ADVANTAGES}" || {
    echo "missing shared raw advantages: ${SOURCE_RAW_ADVANTAGES}"
    echo "Run: bash examples/recap/rounds/run_v3_recap_baseline.sh score_raw"
    exit 1
  }

  run_round_stage '["export_multitask_raw"]'
}

audit_raw_labels() {
  test -f "${POLICY_DATA_REPORT}" || {
    echo "missing policy-data report: ${POLICY_DATA_REPORT}"
    echo "Run the export_raw stage first."
    exit 1
  }
  test -f "${POLICY_MANIFEST}" || {
    echo "missing combined policy manifest: ${POLICY_MANIFEST}"
    exit 1
  }

  feature_manifest="${SOURCE_FEATURE_MANIFEST}"
  if [ ! -f "${feature_manifest}" ]; then
    baseline_feature_manifest="${EXP_ROOT}/revalue/features/manifest.json"
    if [ -f "${baseline_feature_manifest}" ]; then
      feature_manifest="${baseline_feature_manifest}"
    else
      echo "missing raw-Value feature provenance manifest: ${SOURCE_FEATURE_MANIFEST}"
      exit 1
    fi
  fi

  args=(
    --summary-path "${POLICY_DATA_REPORT}"
    --source-advantages-path "${SOURCE_RAW_ADVANTAGES}"
    --source-feature-manifest "${feature_manifest}"
    --value-checkpoint "${VALUE_CHECKPOINT}"
    --advantage-tag "${BASELINE_ADVANTAGE_TAG}"
    --expected-tasks 10
    --expected-episodes-per-task 30
    --expected-rollout-episodes 20
    --expected-expert-episodes 10
    --positive-quantile "${POSITIVE_QUANTILE}"
    --positive-ratio-tolerance 0.02
    --output-path "${RAW_AUDIT_REPORT}"
  )
  for task in "${TASKS[@]}"; do
    args+=(--raw-report "${task}=${EXP_ROOT}/policy_data/${task}/raw_export_report.json")
  done

  python examples/recap/process/audit_recap_baseline_policy_data.py "${args[@]}"
}

train_policy() {
  check_inputs
  test -f "${TASK_POOL_AUDIT}" || {
    echo "missing passed task-pool audit: ${TASK_POOL_AUDIT}"
    exit 1
  }
  test -f "${MERGED_POOL_AUDIT}" || {
    echo "missing passed merged-pool audit: ${MERGED_POOL_AUDIT}"
    exit 1
  }
  test -f "${RAW_AUDIT_REPORT}" || {
    echo "missing passed raw-label audit: ${RAW_AUDIT_REPORT}"
    echo "Run the audit_raw stage before policy training."
    exit 1
  }

  run_round_stage '["train_policy"]' \
    --set "policy.max_steps=${POLICY_MAX_STEPS}" \
    --set "policy.save_interval=${POLICY_SAVE_INTERVAL}" \
    --set policy.lr_warmup_steps=100 \
    --set "policy.global_batch_size=${POLICY_GLOBAL_BATCH_SIZE}" \
    --set "policy.micro_batch_size=${POLICY_MICRO_BATCH_SIZE}" \
    --set 'policy.extra_overrides=["actor.model.openpi.train_expert_only=true","actor.fsdp_config.strategy=fsdp2","actor.fsdp_config.sharding_strategy=no_shard","actor.fsdp_config.use_orig_params=false","+actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[\"NonExistentModule\"]","+actor.fsdp_config.ignored_module_classes=[\"OpenPi0ForCFGActionPrediction\"]","actor.optim.lr=1e-5"]'
}

eval_checkpoints() {
  test -f "${RAW_AUDIT_REPORT}" || {
    echo "missing raw-label audit: ${RAW_AUDIT_REPORT}"
    exit 1
  }
  test -f "${RAW_COMPARISON}" || {
    echo "missing raw Value comparison: ${RAW_COMPARISON}"
    exit 1
  }

  for step in 500 1000; do
    if [ "${step}" -gt "${POLICY_MAX_STEPS}" ]; then
      continue
    fi
    checkpoint="${RESULTS_ROOT}/policy/policy1_recap_baseline/checkpoints/global_step_${step}"
    test -d "${checkpoint}" || {
      echo "missing ReCap baseline checkpoint: ${checkpoint}"
      exit 1
    }

    run_round_stage '["eval_policy"]' \
      --set "policy.max_steps=${step}" \
      --set "eval.output_root=${EXP_ROOT}/eval_step_${step}" \
      --set eval.guidance_type=positive \
      --set eval.positive_only_conditional=true \
      --set eval.guidance_scale=1.0 \
      --set eval.negative_guidance_scale=0.0 \
      --set "eval.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}" \
      --set eval.total_num_envs=10 \
      --set eval.warmup_before_env=true \
      --set eval.save_video=false \
      --set results.enabled=true \
      --set results.round_index=1 \
      --set "results.policy_label=ReCap-raw-step${step}" \
      --set results.critic_label=FixedValue-Raw \
      --set 'results.baseline_success_rates={"task0":0.85,"task1":0.90,"task2":0.90,"task3":0.90,"task4":1.00,"task5":1.00,"task6":0.80,"task7":1.00,"task8":0.20,"task9":0.60}' \
      --set "results.comparison=${RAW_COMPARISON}" \
      --set "results.output_dir=${RESULTS_ROOT}/eval" \
      --set "results.output_name=recap_raw_step_${step}"
  done
}

case "${STAGE}" in
  check)
    check_inputs
    ;;
  audit_data)
    audit_shared_datasets
    ;;
  score_raw)
    score_raw_if_missing
    ;;
  export_raw)
    export_raw_labels
    ;;
  audit_raw)
    audit_raw_labels
    ;;
  train_policy)
    train_policy
    ;;
  eval_checkpoints)
    eval_checkpoints
    ;;
  all)
    audit_shared_datasets
    score_raw_if_missing
    export_raw_labels
    audit_raw_labels
    train_policy
    eval_checkpoints
    ;;
  *)
    echo "unknown stage: ${STAGE}"
    echo "available stages: check, audit_data, score_raw, export_raw, audit_raw, train_policy, eval_checkpoints, all"
    exit 2
    ;;
esac
