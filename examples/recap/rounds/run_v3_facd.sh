#!/usr/bin/env bash
# Round1 v3 FACD pipeline. Collection is performed separately.
set -euo pipefail

cd /workspace/RLinf
source switch_env openpi

export CUDA_VISIBLE_DEVICES=0
export REPO_PATH=/workspace/RLinf
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export HF_LEROBOT_HOME=/data/libero_long
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset RAY_ADDRESS

STAGE="${1:-data_pool}"
TASKS=(task0 task1 task2 task3 task4 task5 task6 task7 task8 task9)

ROLLOUT_ROOT=/data/libero_long/round1_v3_ppo_rollout_20ep_retry1
EXPERT_ROOT=/data/libero_long/round1_v3_ppo_expert_success10
TASK_POOL_ROOT=/data/libero_long/round1_v3_facd_30ep
MERGED_POOL=/data/libero_long/round1_v3_facd_multitask_300ep
REPORT_ROOT=/workspace/results/round1_v3_facd/data_pool
EXP_ROOT=/data/libero_long/round1_v3_facd_exp
RESULTS_ROOT=/workspace/results/round1_v3_facd
BASE_MODEL=/workspace/models/RLinf-Pi05-PPO-LIBERO-130
RETURNS_TAG=round1_v3_facd_returns
BASE_ADVANTAGE_TAG=round1_v3_facd_base_adv
FACD_ADVANTAGE_TAG=round1_v3_facd_fused_top30
POLICY_MAX_STEPS="${POLICY_MAX_STEPS:-1000}"
POLICY_MICRO_BATCH_SIZE="${POLICY_MICRO_BATCH_SIZE:-16}"
EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-2}"
TASK_POOL_AUDIT="${REPORT_ROOT}/task_pool_quality_audit.json"
MERGED_POOL_AUDIT="${REPORT_ROOT}/merged_pool_quality_audit.json"
FACD_LABEL_AUDIT="${REPORT_ROOT}/facd_label_quality_audit.json"

task_dataset_args() {
  for task in "${TASKS[@]}"; do
    printf '%s\n' --task-dataset "${task}=${TASK_POOL_ROOT}/${task}_30ep"
  done
}

audit_task_pools() {
  mapfile -t dataset_args < <(task_dataset_args)
  python examples/recap/process/audit_v3_facd_datasets.py task-pools \
    "${dataset_args[@]}" \
    --rollout-root "${ROLLOUT_ROOT}" \
    --expert-root "${EXPERT_ROOT}" \
    --output-path "${TASK_POOL_AUDIT}"
}

audit_merged_pool() {
  mapfile -t dataset_args < <(task_dataset_args)
  python examples/recap/process/audit_v3_facd_datasets.py merged \
    "${dataset_args[@]}" \
    --merged-dataset "${MERGED_POOL}" \
    --output-path "${MERGED_POOL_AUDIT}"
}

audit_facd_labels() {
  mapfile -t dataset_args < <(task_dataset_args)
  report_args=()
  for task in "${TASKS[@]}"; do
    report_args+=(
      --export-report
      "${task}=${EXP_ROOT}/policy_data/${task}/export_report.json"
    )
  done
  python examples/recap/process/audit_v3_facd_datasets.py facd \
    "${dataset_args[@]}" \
    "${report_args[@]}" \
    --advantage-tag "${FACD_ADVANTAGE_TAG}" \
    --positive-quantile 0.3 \
    --failure-positive-cap 0.2 \
    --output-path "${FACD_LABEL_AUDIT}"
}

build_task_pools() {
  for task in "${TASKS[@]}"; do
    rollout_dataset="${ROLLOUT_ROOT}/${task}_20ep"
    expert_dataset="${EXPERT_ROOT}/${task}_10ep"
    test -f "${rollout_dataset}/meta/episodes.jsonl" || {
      echo "missing rollout dataset: ${rollout_dataset}"
      exit 1
    }
    test -f "${expert_dataset}/meta/episodes.jsonl" || {
      echo "missing expert dataset: ${expert_dataset}"
      exit 1
    }
    rollout_episodes="$(wc -l < "${rollout_dataset}/meta/episodes.jsonl")"
    if [ "${rollout_episodes}" -ne 20 ]; then
      echo "${task}: expected 20 rollout episodes, found ${rollout_episodes}"
      exit 1
    fi
  done

  if [ -e "${TASK_POOL_ROOT}" ]; then
    echo "task pool root already exists: ${TASK_POOL_ROOT}"
    echo "Do not overwrite it; inspect it or choose a new path."
    exit 1
  fi
  mkdir -p "${REPORT_ROOT}"

  python examples/recap/process/build_fixed_multitask_train_pools.py \
    --tasks "${TASKS[@]}" \
    --rollout-pattern "${ROLLOUT_ROOT}/{task}_20ep" \
    --expert-pattern "${EXPERT_ROOT}/{task}_10ep" \
    --experts-per-task 10 \
    --episodes-per-task 30 \
    --output-pattern "${TASK_POOL_ROOT}/{task}_30ep" \
    --report-path "${REPORT_ROOT}/task_pool_report.json"
}

merge_pool() {
  for task in "${TASKS[@]}"; do
    task_pool="${TASK_POOL_ROOT}/${task}_30ep"
    test -f "${task_pool}/meta/episodes.jsonl" || {
      echo "missing task pool: ${task_pool}"
      exit 1
    }
    test -f "${task_pool}/meta/episode_provenance.jsonl" || {
      echo "missing provenance: ${task_pool}/meta/episode_provenance.jsonl"
      exit 1
    }
    test -f "${task_pool}/meta/full_positive_episodes.json" || {
      echo "missing expert backstop: ${task_pool}/meta/full_positive_episodes.json"
      exit 1
    }
    task_episodes="$(wc -l < "${task_pool}/meta/episodes.jsonl")"
    if [ "${task_episodes}" -ne 30 ]; then
      echo "${task}: expected 30 task-pool episodes, found ${task_episodes}"
      exit 1
    fi
  done

  if [ -e "${MERGED_POOL}" ]; then
    echo "merged pool already exists: ${MERGED_POOL}"
    echo "Do not overwrite it; inspect it or choose a new path."
    exit 1
  fi

  mkdir -p "${REPORT_ROOT}"

  python examples/recap/process/merge_lerobot_multitask_datasets.py \
    --datasets \
      "${TASK_POOL_ROOT}/task0_30ep" \
      "${TASK_POOL_ROOT}/task1_30ep" \
      "${TASK_POOL_ROOT}/task2_30ep" \
      "${TASK_POOL_ROOT}/task3_30ep" \
      "${TASK_POOL_ROOT}/task4_30ep" \
      "${TASK_POOL_ROOT}/task5_30ep" \
      "${TASK_POOL_ROOT}/task6_30ep" \
      "${TASK_POOL_ROOT}/task7_30ep" \
      "${TASK_POOL_ROOT}/task8_30ep" \
      "${TASK_POOL_ROOT}/task9_30ep" \
    --output_dataset "${MERGED_POOL}" \
    | tee "${REPORT_ROOT}/merged_pool_report.log"
}

repair_rollout_traces() {
  python examples/recap/process/repair_semantic_trace_artifacts.py \
    --dataset "task0=${ROLLOUT_ROOT}/task0_20ep" \
    --dataset "task1=${ROLLOUT_ROOT}/task1_20ep" \
    --dataset "task2=${ROLLOUT_ROOT}/task2_20ep" \
    --dataset "task3=${ROLLOUT_ROOT}/task3_20ep" \
    --dataset "task4=${ROLLOUT_ROOT}/task4_20ep" \
    --dataset "task5=${ROLLOUT_ROOT}/task5_20ep" \
    --dataset "task6=${ROLLOUT_ROOT}/task6_20ep" \
    --dataset "task7=${ROLLOUT_ROOT}/task7_20ep" \
    --dataset "task8=${ROLLOUT_ROOT}/task8_20ep" \
    --dataset "task9=${ROLLOUT_ROOT}/task9_20ep" \
    --fix-summary \
    --min-trainable 10
}

critic_fusion() {
  test -f "${MERGED_POOL_AUDIT}" || {
    echo "missing passed merged-pool audit: ${MERGED_POOL_AUDIT}"
    echo "Run prepare_data before critic_fusion."
    exit 1
  }
  test -f "${MERGED_POOL}/meta/info.json" || {
    echo "missing merged 300ep pool: ${MERGED_POOL}"
    echo "Run merge_pool before critic_fusion."
    exit 1
  }

  python examples/recap/rounds/run_round.py \
    --config examples/recap/rounds/config/multitask_round1_prcfg.yaml \
    --stage all \
    --set 'pipeline=["fit_critic_multitask","task_heads","predict_multitask"]' \
    --set 'tasks=["task0","task1","task2","task3","task4","task5","task6","task7","task8","task9"]' \
    --set "paths.base_model=${BASE_MODEL}" \
    --set "paths.parent_dataset=${MERGED_POOL}" \
    --set "paths.child_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
    --set "paths.policy_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
    --set "paths.merged_dataset=${MERGED_POOL}" \
    --set "paths.exp_root=${EXP_ROOT}" \
    --set "paths.results_root=${RESULTS_ROOT}" \
    --set datasets.parent_episodes=0 \
    --set datasets.prebuilt_merged=true \
    --set collect.num_episodes=30 \
    --set tags.returns="${RETURNS_TAG}" \
    --set tags.merged_base="${BASE_ADVANTAGE_TAG}" \
    --set tags.child_fused="${FACD_ADVANTAGE_TAG}" \
    --set value.freeze_vlm=true \
    --set value.steps=1200 \
    --set value.save_interval=1200 \
    --set value.micro_batch_size=16 \
    --set value.global_batch_size=64 \
    --set value.lr=1e-5 \
    --set value.value_lr=1e-4 \
    --set value.lr_warmup_steps=100 \
    --set revalue.task_val_episode_ratio=0.2 \
    --set revalue.positive_quantile=0.3 \
    --set revalue.zp_max_epochs=100 \
    --set revalue.zp_early_stop_patience=10 \
    --set revalue.fusion_max_epochs=100 \
    --set revalue.fusion_early_stop_patience=10 \
    --set returns.failure_reward=-300.0
}

export_facd() {
  test -f "${MERGED_POOL_AUDIT}" || {
    echo "missing passed merged-pool audit: ${MERGED_POOL_AUDIT}"
    exit 1
  }
  base_advantages="${MERGED_POOL}/meta/advantages_${BASE_ADVANTAGE_TAG}.parquet"
  predictions="${EXP_ROOT}/revalue/predictions.parquet"
  test -f "${base_advantages}" || {
    echo "missing merged base advantages: ${base_advantages}"
    echo "Run the Value/fusion stages before export_facd."
    exit 1
  }
  test -f "${predictions}" || {
    echo "missing fused predictions: ${predictions}"
    echo "Run the Value/fusion stages before export_facd."
    exit 1
  }

  python examples/recap/rounds/run_round.py \
    --config examples/recap/rounds/config/multitask_round1_prcfg.yaml \
    --stage all \
    --force \
    --set 'pipeline=["export_multitask"]' \
    --set 'tasks=["task0","task1","task2","task3","task4","task5","task6","task7","task8","task9"]' \
    --set "paths.base_model=${BASE_MODEL}" \
    --set "paths.parent_dataset=${MERGED_POOL}" \
    --set "paths.child_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
    --set "paths.policy_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
    --set "paths.merged_dataset=${MERGED_POOL}" \
    --set "paths.exp_root=${EXP_ROOT}" \
    --set "paths.results_root=${RESULTS_ROOT}" \
    --set datasets.parent_episodes=0 \
    --set datasets.prebuilt_merged=true \
    --set collect.num_episodes=30 \
    --set tags.returns="${RETURNS_TAG}" \
    --set tags.merged_base="${BASE_ADVANTAGE_TAG}" \
    --set tags.child_fused="${FACD_ADVANTAGE_TAG}" \
    --set revalue.positive_quantile=0.3 \
    --set revalue.success_gate=true \
    --set revalue.failure_positive_cap=0.2 \
    --set revalue.demo_backstop=true \
    --set returns.failure_reward=-300.0
}

train_policy() {
  manifest="${EXP_ROOT}/policy_data/multitask_episode_manifest.json"
  test -f "${BASE_MODEL}/model.safetensors" || {
    echo "missing PPO base model: ${BASE_MODEL}/model.safetensors"
    exit 1
  }
  test -f "${manifest}" || {
    echo "missing policy manifest: ${manifest}"
    echo "Run export_facd before train_policy."
    exit 1
  }
  test -f "${FACD_LABEL_AUDIT}" || {
    echo "missing passed FACD-label audit: ${FACD_LABEL_AUDIT}"
    echo "Run audit_facd before train_policy."
    exit 1
  }
  for task in "${TASKS[@]}"; do
    task_pool="${TASK_POOL_ROOT}/${task}_30ep"
    advantage_path="${task_pool}/meta/advantages_${FACD_ADVANTAGE_TAG}.parquet"
    export_report="${EXP_ROOT}/policy_data/${task}/export_report.json"
    test -f "${advantage_path}" || {
      echo "missing FACD labels: ${advantage_path}"
      exit 1
    }
    test -f "${export_report}" || {
      echo "missing FACD export report: ${export_report}"
      exit 1
    }
  done

  python examples/recap/rounds/run_round.py \
    --config examples/recap/rounds/config/multitask_round1_prcfg.yaml \
    --stage all \
    --set 'pipeline=["train_policy"]' \
    --set method=facd \
    --set 'tasks=["task0","task1","task2","task3","task4","task5","task6","task7","task8","task9"]' \
    --set "paths.base_model=${BASE_MODEL}" \
    --set "paths.parent_dataset=${MERGED_POOL}" \
    --set "paths.child_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
    --set "paths.policy_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
    --set "paths.merged_dataset=${MERGED_POOL}" \
    --set "paths.exp_root=${EXP_ROOT}" \
    --set "paths.results_root=${RESULTS_ROOT}" \
    --set datasets.parent_episodes=0 \
    --set datasets.prebuilt_merged=true \
    --set collect.num_episodes=30 \
    --set parent_policy.checkpoint='' \
    --set parent_policy.label=PPO-LIBERO-130 \
    --set tags.returns="${RETURNS_TAG}" \
    --set tags.merged_base="${BASE_ADVANTAGE_TAG}" \
    --set tags.child_fused="${FACD_ADVANTAGE_TAG}" \
    --set policy.advantage_source=fused \
    --set policy.strategy=binary \
    --set policy.guidance_type=positive \
    --set policy.positive_only_conditional=true \
    --set policy.negative_guidance_scale=0.0 \
    --set policy.unconditional_prob=0.1 \
    --set policy.positive_quantile=0.3 \
    --set policy.max_steps="${POLICY_MAX_STEPS}" \
    --set policy.save_interval=500 \
    --set policy.lr_warmup_steps=100 \
    --set policy.global_batch_size=64 \
    --set policy.micro_batch_size="${POLICY_MICRO_BATCH_SIZE}" \
    --set 'policy.extra_overrides=["actor.model.openpi.train_expert_only=true","actor.fsdp_config.strategy=fsdp2","actor.fsdp_config.sharding_strategy=no_shard","actor.fsdp_config.use_orig_params=false","+actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[\"NonExistentModule\"]","+actor.fsdp_config.ignored_module_classes=[\"OpenPi0ForCFGActionPrediction\"]","actor.optim.lr=1e-5"]'
}

eval_checkpoints() {
  comparison="${EXP_ROOT}/revalue/return_compare.json"
  policy_data_report="${EXP_ROOT}/policy_data/multitask_policy_data_report.json"
  test -f "${comparison}" || {
    echo "missing Value/fusion comparison: ${comparison}"
    exit 1
  }
  test -f "${policy_data_report}" || {
    echo "missing FACD policy-data report: ${policy_data_report}"
    exit 1
  }

  for step in 500 1000; do
    if [ "${step}" -gt "${POLICY_MAX_STEPS}" ]; then
      continue
    fi
    checkpoint="${RESULTS_ROOT}/policy/policy1_facd/checkpoints/global_step_${step}"
    test -d "${checkpoint}" || {
      echo "missing policy checkpoint: ${checkpoint}"
      exit 1
    }

    python examples/recap/rounds/run_round.py \
      --config examples/recap/rounds/config/multitask_round1_prcfg.yaml \
      --stage all \
      --set 'pipeline=["eval_policy"]' \
      --set method=facd \
      --set 'tasks=["task0","task1","task2","task3","task4","task5","task6","task7","task8","task9"]' \
      --set "paths.base_model=${BASE_MODEL}" \
      --set "paths.parent_dataset=${MERGED_POOL}" \
      --set "paths.child_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
      --set "paths.policy_pattern=${TASK_POOL_ROOT}/{task}_30ep" \
      --set "paths.merged_dataset=${MERGED_POOL}" \
      --set "paths.exp_root=${EXP_ROOT}" \
      --set "paths.results_root=${RESULTS_ROOT}" \
      --set datasets.parent_episodes=0 \
      --set datasets.prebuilt_merged=true \
      --set collect.num_episodes=30 \
      --set parent_policy.checkpoint='' \
      --set tags.returns="${RETURNS_TAG}" \
      --set tags.merged_base="${BASE_ADVANTAGE_TAG}" \
      --set tags.child_fused="${FACD_ADVANTAGE_TAG}" \
      --set policy.advantage_source=fused \
      --set policy.strategy=binary \
      --set policy.guidance_type=positive \
      --set policy.positive_only_conditional=true \
      --set policy.negative_guidance_scale=0.0 \
      --set policy.max_steps="${step}" \
      --set eval.output_root="${EXP_ROOT}/eval_step_${step}" \
      --set eval.guidance_type=positive \
      --set eval.positive_only_conditional=true \
      --set eval.guidance_scale=1.0 \
      --set eval.negative_guidance_scale=0.0 \
      --set eval.eval_rollout_epoch="${EVAL_ROLLOUT_EPOCH}" \
      --set eval.total_num_envs=10 \
      --set eval.warmup_before_env=true \
      --set eval.save_video=false \
      --set results.enabled=true \
      --set results.round_index=1 \
      --set results.policy_label="FACD-step${step}" \
      --set results.critic_label=Value-FACD-frozen \
      --set 'results.baseline_success_rates={"task0":0.85,"task1":0.90,"task2":0.90,"task3":0.90,"task4":1.00,"task5":1.00,"task6":0.80,"task7":1.00,"task8":0.20,"task9":0.60}' \
      --set results.comparison="${comparison}" \
      --set results.output_dir="${RESULTS_ROOT}/eval" \
      --set results.output_name="facd_step_${step}"
  done
}

prepare_data() {
  audit_task_pools
  if [ -e "${MERGED_POOL}" ]; then
    echo "reusing existing merged pool: ${MERGED_POOL}"
  else
    merge_pool
  fi
  audit_merged_pool
}

run_all() {
  prepare_data
  critic_fusion
  export_facd
  audit_facd_labels
  train_policy
  eval_checkpoints
}

case "${STAGE}" in
  task_pools)
    build_task_pools
    ;;
  merge_pool)
    merge_pool
    ;;
  repair_rollout_traces)
    repair_rollout_traces
    ;;
  audit_task_pools)
    audit_task_pools
    ;;
  audit_merged)
    audit_merged_pool
    ;;
  prepare_data)
    prepare_data
    ;;
  data_pool)
    build_task_pools
    merge_pool
    ;;
  critic_fusion)
    critic_fusion
    ;;
  export_facd)
    export_facd
    ;;
  audit_facd)
    audit_facd_labels
    ;;
  train_policy)
    train_policy
    ;;
  eval_checkpoints)
    eval_checkpoints
    ;;
  all)
    run_all
    ;;
  *)
    echo "unknown stage: ${STAGE}"
    echo "available stages: task_pools, audit_task_pools, merge_pool, audit_merged, prepare_data, data_pool, critic_fusion, export_facd, audit_facd, train_policy, eval_checkpoints, all"
    exit 2
    ;;
esac
