#!/usr/bin/env bash
# Round1 v3: expert-only policy training (train_policy + eval_policy only).
# Reuses round1_v2 fused advantages and Value2 return comparison.
# Run inside the openpi container: bash examples/recap/rounds/run_v3_expert.sh
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

python examples/recap/rounds/run_round.py \
  --config examples/recap/rounds/config/multitask_round1_prcfg.yaml \
  --stage all \
  --set 'pipeline=["train_policy","eval_policy"]' \
  --set 'tasks=["task0","task1","task2","task3","task4","task5","task6","task7","task8","task9"]' \
  --set 'paths.child_pattern=/data/libero_long/round1_v2_fixed/{task}' \
  --set paths.merged_dataset=/data/libero_long/round1_v2_merged_220ep \
  --set paths.exp_root=/workspace/results/round1_v3_expert/exp \
  --set paths.results_root=/workspace/results/round1_v3_expert/checkpoints \
  --set parent_policy.checkpoint='' \
  --set parent_policy.label=SFT-Base \
  --set tags.returns=round1_v2_returns \
  --set tags.merged_base=round1_v2_base_adv \
  --set tags.child_fused=round1_v2_fused_top30 \
  --set policy.strategy=csa_residual \
  --set policy.guidance_type=positive \
  --set policy.positive_only_conditional=false \
  --set policy.negative_guidance_scale=0.0 \
  --set policy.positive_residual_alpha=0.5 \
  --set policy.positive_quantile=0.3 \
  --set policy.unconditional_prob=0.1 \
  --set policy.csa_bottom_quantile=0.15 \
  --set policy.csa_bottom_negative_prob=0.5 \
  --set policy.csa_positive_prompt_prob=0.85 \
  --set policy.csa_weight_lambda=0.2 \
  --set policy.max_steps=1500 \
  --set policy.save_interval=500 \
  --set policy.lr_warmup_steps=100 \
  --set policy.global_batch_size=64 \
  --set policy.micro_batch_size=8 \
  --set 'policy.extra_overrides=["actor.model.openpi.train_expert_only=true","actor.fsdp_config.strategy=fsdp2","actor.fsdp_config.sharding_strategy=no_shard","actor.fsdp_config.use_orig_params=false","+actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[\"NonExistentModule\"]","+actor.fsdp_config.ignored_module_classes=[\"OpenPi0ForCFGActionPrediction\"]","actor.optim.lr=1e-5"]' \
  --set eval.guidance_type=positive \
  --set eval.positive_only_conditional=false \
  --set eval.guidance_scale=1.0 \
  --set eval.negative_guidance_scale=0.0 \
  --set eval.eval_rollout_epoch=2 \
  --set eval.total_num_envs=10 \
  --set eval.warmup_before_env=true \
  --set eval.save_video=false \
  --set results.enabled=true \
  --set results.round_index=1 \
  --set results.policy_label=Policy1-v3-expert-lr1e5 \
  --set results.critic_label=Value2-frozen \
  --set results.comparison=/workspace/RLinf/persistent_results/round1_v2/exp/revalue/return_compare.json \
  --set results.output_dir=/workspace/results/round1_v3_expert/results \
  --set results.output_name=round1_v3_expert_policy1_value2
