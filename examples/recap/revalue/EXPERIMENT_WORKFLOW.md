# Revalue 端到端工作流

这是 Revalue 论文实验的维护端到端工作流。
请在 Docker 容器内运行：

```bash
docker exec -it rlinf_pi06star bash
cd /workspace/RLinf
git pull fork gczx_sim
source switch_env openpi 2>/dev/null || true
export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
```

面向用户的工作流不再是一连串分散的脚本。唯一入口点为：

```bash
python examples/recap/revalue/revalue.py
```

在 Revalue 导出标准 `meta/advantages_<tag>.parquet` 后，仍使用原有 ReCap 训练
infra。

## 方法

维护的比较仅包含两种方法：

1. `base`: raw pi0.5/value critic logits 与 base ReCap advantage。
2. `shared_mlp_fusion`: frozen VLM features -> z/p head -> frozen z/p head +
   raw logits -> fusion MLP -> fused value。

Fused 方法有意设计为两阶段：

```text
train_zp:
  frozen VLM features -> SharedMLPPhaseProgressHead

train_fusion:
  frozen z/p head + raw value logits -> LogitFusionMLP
```

## 公共变量

```bash
cd /workspace/RLinf

export DATASET=/workspace/datasets/recap_libero10_task0/libero10_task0_train
export VALUE_CKPT=/workspace/results/value_sft/checkpoints/global_step_5000
export SIGLIP=/workspace/models/siglip2-so400m-patch14-224
export GEMMA=/workspace/models/gemma-3-270m

export RUN_NAME=base30ep_random200
export BASE_TAG=base30ep_random200_logits_phase_dist
export FUSED_TAG=${RUN_NAME}_shared_mlp_fusion
export REVALUE_ROOT=/workspace/results/revalue/${RUN_NAME}_shared_mlp_fusion
```

所有非 source 输出均放在 `${REVALUE_ROOT}` 下。Dataset sidecar 写入
`${DATASET}/meta` 下。

## 一条命令：完整 Fused Pipeline

该命令运行：

```text
prepare_data -> build_base -> extract_features -> train_zp -> train_fusion
-> predict -> export -> compare_returns
```

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  data.dataset_path=${DATASET} \
  data.robot_type=libero \
  data.env_type=libero \
  data.model_type=pi05 \
  data.label_name=phase_progress_semantic \
  manifest.num_episodes=200 \
  manifest.success_ratio=0.5 \
  manifest.val_episode_ratio=0.2 \
  manifest.test_episode_ratio=0.0 \
  value.checkpoint=${VALUE_CKPT} \
  value.siglip_path=${SIGLIP} \
  value.gemma3_path=${GEMMA} \
  value.tokenizer_path=${GEMMA} \
  value.critic_expert_variant=gemma_1m \
  returns.global_min=-700.0 \
  returns.global_max=0.0 \
  returns.dataset_type=rollout \
  returns.failure_reward=-300.0 \
  base.tag=${BASE_TAG} \
  base.batch_size=256 \
  output.root=${REVALUE_ROOT} \
  recap.output_tag=${FUSED_TAG} \
  recap.lookahead_step=10 \
  recap.gamma=1.0 \
  recap.positive_quantile=0.3 \
  train.device=cuda \
  train.batch_size=256 \
  train.extract_batch_size=16
```

预期的 Revalue 输出：

```text
${REVALUE_ROOT}/episode_manifest.json
${REVALUE_ROOT}/features/train.pt
${REVALUE_ROOT}/features/val.pt
${REVALUE_ROOT}/zp_head/zp_head.pt
${REVALUE_ROOT}/fusion/fusion.pt
${REVALUE_ROOT}/predictions.parquet
${REVALUE_ROOT}/return_compare.json
```

预期的 dataset sidecar：

```text
${DATASET}/meta/returns_${BASE_TAG}.parquet
${DATASET}/meta/advantages_${BASE_TAG}.parquet
${DATASET}/meta/advantages_${FUSED_TAG}.parquet
```

## 分 Stage 执行

当调试或某个 stage 已完成时使用。

### 1. Prepare Data

采样 balanced episode subset，写入一个包含 selected/train/val/test episode id
的 manifest。

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=prepare_data \
  data.dataset_path=${DATASET} \
  data.label_name=phase_progress_semantic \
  manifest.num_episodes=200 \
  manifest.success_ratio=0.5 \
  manifest.val_episode_ratio=0.2 \
  manifest.test_episode_ratio=0.0 \
  output.root=${REVALUE_ROOT}
```

### 2. Build Base

计算 returns 和 raw critic base advantage，并保存 `value_logits_current`。
用户不再需要直接调用旧的分散数据处理入口。

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=build_base \
  data.dataset_path=${DATASET} \
  data.robot_type=libero \
  data.model_type=pi05 \
  value.checkpoint=${VALUE_CKPT} \
  value.siglip_path=${SIGLIP} \
  value.gemma3_path=${GEMMA} \
  value.tokenizer_path=${GEMMA} \
  value.critic_expert_variant=gemma_1m \
  returns.global_min=-700.0 \
  returns.global_max=0.0 \
  returns.dataset_type=rollout \
  returns.failure_reward=-300.0 \
  base.tag=${BASE_TAG} \
  output.root=${REVALUE_ROOT}
```

### 3. Extract Features

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=extract_features \
  data.dataset_path=${DATASET} \
  data.robot_type=libero \
  data.env_type=libero \
  data.model_type=pi05 \
  value.checkpoint=${VALUE_CKPT} \
  value.siglip_path=${SIGLIP} \
  value.gemma3_path=${GEMMA} \
  value.tokenizer_path=${GEMMA} \
  value.critic_expert_variant=gemma_1m \
  output.root=${REVALUE_ROOT} \
  train.device=cuda \
  train.extract_batch_size=16
```

### 4. Train Z/P Head

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=train_zp \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  train.device=cuda \
  train.batch_size=256 \
  zp.max_epochs=100 \
  zp.early_stop_patience=10
```

### 5. Train Fusion

此 stage 冻结 z/p head，仅训练 fusion MLP。

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=train_fusion \
  data.dataset_path=${DATASET} \
  base.tag=${BASE_TAG} \
  output.root=${REVALUE_ROOT} \
  returns.global_min=-700.0 \
  returns.global_max=0.0 \
  train.device=cuda \
  train.batch_size=256 \
  fusion.max_epochs=100 \
  fusion.early_stop_patience=10 \
  fusion.alpha=1.0
```

### 6. Predict

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=predict \
  data.dataset_path=${DATASET} \
  base.tag=${BASE_TAG} \
  output.root=${REVALUE_ROOT} \
  train.device=cuda \
  train.batch_size=512
```

### 7. Export Fused Advantage

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=export \
  data.dataset_path=${DATASET} \
  base.tag=${BASE_TAG} \
  output.root=${REVALUE_ROOT} \
  recap.output_tag=${FUSED_TAG} \
  recap.lookahead_step=10 \
  recap.gamma=1.0 \
  recap.positive_quantile=0.3 \
  recap.discount_next_value=true \
  recap.export_split=train
```

### 8. Compare Return Prediction

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=compare_returns \
  data.dataset_path=${DATASET} \
  base.tag=${BASE_TAG} \
  output.root=${REVALUE_ROOT} \
  returns.global_min=-700.0 \
  returns.global_max=0.0
```

报告位于：

```text
${REVALUE_ROOT}/return_compare.json
```

包含 frame-level 指标和 episode-level 配对摘要。

## 复现有 Random200 产物

如果你已有 feature cache 和 base advantage parquet：

```bash
export FEATURES_DIR=/workspace/results/phase_progress_probe/features_random200_base30ep_balanced
export ADV_PATH=${DATASET}/meta/advantages_${BASE_TAG}.parquet
export REVALUE_ROOT=/workspace/results/revalue/${RUN_NAME}_shared_mlp_fusion_return_compare
```

仅运行使用它们的维护 Revalue stage：

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=train_zp \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  train.device=cuda

python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=train_fusion \
  data.dataset_path=${DATASET} \
  recap.source_advantages_path=${ADV_PATH} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  returns.global_min=-700.0 \
  returns.global_max=0.0 \
  train.device=cuda

python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=predict \
  data.dataset_path=${DATASET} \
  recap.source_advantages_path=${ADV_PATH} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  train.device=cuda

python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=compare_returns \
  data.dataset_path=${DATASET} \
  recap.source_advantages_path=${ADV_PATH} \
  output.root=${REVALUE_ROOT} \
  returns.global_min=-700.0 \
  returns.global_max=0.0
```

## ReCap/CFG 训练

Revalue 导出标准的 ReCap advantage 文件，因此原有 CFG 训练 infra 无需改变。

Base:

```bash
bash examples/recap/cfg/run_cfg_sft.sh libero_cfg_openpi \
  runner.logger.experiment_name=cfg_${BASE_TAG} \
  data.train_data_paths="[{'dataset_path':'${DATASET}','type':'rollout','weight':1.0}]" \
  data.advantage_tag=${BASE_TAG} \
  data.episode_split_path=${REVALUE_ROOT}/episode_manifest.json \
  data.episode_split_name=train \
  actor.model.model_path=/workspace/models/pi05_base_pytorch
```

Fused:

```bash
bash examples/recap/cfg/run_cfg_sft.sh libero_cfg_openpi \
  runner.logger.experiment_name=cfg_${FUSED_TAG} \
  data.train_data_paths="[{'dataset_path':'${DATASET}','type':'rollout','weight':1.0}]" \
  data.advantage_tag=${FUSED_TAG} \
  data.episode_split_path=${REVALUE_ROOT}/episode_manifest.json \
  data.episode_split_name=train \
  actor.model.model_path=/workspace/models/pi05_base_pytorch
```

python examples/recap/revalue/revalue.py stage=train_cfg \
  cfg_train.enabled=true \
  cfg_train.base_model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  cfg_train.advantage_tag=your_exported_tag
python examples/recap/revalue/revalue.py stage=eval_policy \
  policy_eval.enabled=true \
  policy_eval.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  policy_eval.model_type=cfg_model \
  policy_eval.checkpoint_path=/workspace/RLinf/logs/.../full_weights.pt
python examples/recap/revalue/revalue.py stage=collect_rollouts \
  rollout_collect.enabled=true \
  rollout_collect.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  rollout_collect.model_type=cfg_model \
  rollout_collect.checkpoint_path=/workspace/RLinf/logs/.../full_weights.pt \
  rollout_collect.output_dir=/workspace/datasets/libero_task0_cfg_rollouts