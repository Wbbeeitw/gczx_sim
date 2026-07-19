#!/usr/bin/env bash
# One-shot C-chain smoke (fit_critic_multitask + task_heads + predict_multitask)
# on the original-SFT 1ep/task D0 + task5/task8 demo slices.
# Usage:
#   bash examples/recap/rounds/run_c_smoke_original_sft.sh            # real run
#   bash examples/recap/rounds/run_c_smoke_original_sft.sh --dry-run  # print subcommands only
#   GPU_ID=1 bash examples/recap/rounds/run_c_smoke_original_sft.sh   # run on GPU1
set -euo pipefail

GPU_ID="${GPU_ID:-0}"

cd /workspace/RLinf
source switch_env openpi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export REPO_PATH=/workspace/RLinf
export PYTHONPATH=/workspace/RLinf
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export HF_LEROBOT_HOME=/data/libero_long
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset RAY_ADDRESS

python examples/recap/rounds/run_round.py \
  --config examples/recap/rounds/config/multitask_smoke_demos_prcfg.yaml \
  --stage all \
  --set round_id=0 \
  --set task=multitask \
  --set task_id=0 \
  --set task_suite_name=libero_10 \
  --set method=prcfg \
  --set bootstrap=false \
  --set repo_root=/workspace/RLinf \
  --set 'pipeline=["fit_critic_multitask","task_heads","predict_multitask"]' \
  --set 'tasks=["task0","task1","task2","task3","task4","task5","task6","task7","task8","task9"]' \
  --set "gpu.cuda_visible_devices=\"${GPU_ID}\"" \
  --set "gpu.gpu_id=${GPU_ID}" \
  --set paths.base_model=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  --set paths.siglip=/workspace/models/siglip2-so400m-patch14-224 \
  --set paths.gemma3=/workspace/models/gemma-3-270m \
  --set paths.tokenizer=/workspace/models/gemma-3-270m \
  --set paths.parent_dataset=/data/libero_long/smk_test/c_original_sft/unused_parent \
  --set 'paths.child_pattern=/data/libero_long/smk_test/d0_original_sft_all10_1ep/{task}_1ep' \
  --set paths.merged_dataset=/data/libero_long/smk_test/c_original_sft/merged_d0_demo24 \
  --set paths.exp_root=/data/libero_long/smk_test/c_original_sft/exp \
  --set paths.results_root=/workspace/RLinf/persistent_results/smk_test/c_original_sft \
  --set paths.value_checkpoint=null \
  --set datasets.parent_episodes=0 \
  --set collect.num_episodes=1 \
  --set 'demo_datasets={"task5":{"path":"/data/libero_long/libero_task58_replayed_merged","start":0,"end":15},"task8":{"path":"/data/libero_long/libero_task58_replayed_merged","start":41,"end":50}}' \
  --set parent_policy.checkpoint='' \
  --set parent_policy.label=original_sft \
  --set tags.returns=smk_test_original_sft_returns \
  --set tags.merged_base=smk_test_original_sft_base \
  --set tags.child_fused=smk_test_original_sft_fused \
  --set returns.failure_reward=-300.0 \
  --set returns.gamma=1.0 \
  --set returns.global_min=-900.0 \
  --set returns.global_max=0.0 \
  --set returns.num_workers=2 \
  --set value.freeze_vlm=false \
  --set value.steps=2 \
  --set value.save_interval=2 \
  --set value.micro_batch_size=1 \
  --set value.global_batch_size=8 \
  --set value.lr=1.0e-5 \
  --set value.value_lr=1.0e-4 \
  --set value.lr_warmup_steps=1 \
  --set value.action_dim=7 \
  --set value.action_horizon=10 \
  --set value.critic_expert_variant=gemma_1m \
  --set revalue.label_name=phase_progress_multitask \
  --set revalue.num_phases=4 \
  --set revalue.success_phase=3 \
  --set revalue.seed=42 \
  --set revalue.lookahead_step=10 \
  --set revalue.positive_quantile=0.3 \
  --set revalue.zp_head_type=temporal_local_stage_gated \
  --set revalue.zp_use_class_weights=false \
  --set revalue.zp_max_epochs=1 \
  --set revalue.zp_early_stop_patience=1 \
  --set revalue.fusion_max_epochs=1 \
  --set revalue.fusion_early_stop_patience=1 \
  --set revalue.fusion_alpha=1.0 \
  --set revalue.extract_batch_size=1 \
  --set revalue.train_batch_size=32 \
  --set revalue.task_val_episode_ratio=0.2 \
  --set revalue.allow_single_episode_overlap=true \
  "$@"
