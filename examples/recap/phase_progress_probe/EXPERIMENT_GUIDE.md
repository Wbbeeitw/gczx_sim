# Phase-Progress Probe 实验指南

本文档总结了 `examples/recap/phase_progress_probe/` 下的当前 phase/progress probe 项目，
以及它与 `examples/recap/process/` 下的 ReCap 后处理 pipeline 的关联。

本文目的不是描述 RLinf 的全部内容，而是记录你一直在迭代的具体实验路线：

1. 从已训练的 value critic 出发
2. 提取 frozen VLM features
3. 训练 `z/p` predictor
4. 评估预测的 `z/p` 能否改善 value 估计
5. 可选地训练一个 learned logit-space fusion MLP
6. 将修正后的 advantage 导回标准 ReCap tag


## 1. 项目目标

该项目研究 frozen `ValueCriticModel` backbone 是否已经包含足够的任务结构信息来预测：

- semantic phase `z_t`
- within-phase progress `p_t`
- 可选地，基于 raw critic logits 和预测的 `z/p` 构建的 fused value correction

更广泛的动机是：

- raw visual critic value 可能因 occlusion、ambiguous views 或 visually similar states 而产生 bias
- stage 和 progress 作为关于 remaining return 的额外证据
- 该证据可以用于：
  - 简单的 predicted linear bias correction
  - learned logit-space fusion model


## 2. 目录地图

### 2.1 Probe 目录

主目录：

- `examples/recap/phase_progress_probe/`

重要脚本：

- `extract_features.py`
  - 加载 value critic checkpoint
  - 冻结 VLM
  - 将每帧 prefix features 缓存到 `train.pt` 和 `val.pt`
- `train_head.py`
  - 训练 baseline 单帧 `Shared MLP Trunk + Two Heads`
- `predict_and_analyze.py`
  - 预测 `z/p`
  - 计算快速分析指标
  - 也可以做旧版非严格的 predicted/oracle correction 分析
- `run_strict_baseline_suite.py`
  - 编排 strict 4-method baseline 对比
  - 训练每种 `z/p` 方法
  - 在 train/val 上预测
  - 仅在 train 上拟合 linear correction
  - 仅在 val 上报告 raw vs predicted
- `train_logit_fusion_from_predictions.py`
  - 从 frozen `z/p` predictions 训练 logit-space fusion MLP
- `evaluate_raw_vs_fusion_strict.py`
  - 仅在 val 上评估 raw vs fused value MSE
- `run_strict_fusion_suite.py`
  - 编排 strict fusion 协议：
    - 训练 `z/p`
    - 导出 frozen predictions
    - 每种方法训练一个 fusion MLP
    - 在 val 上评估 raw vs fused
- `model.py`
  - baseline 单帧 head
- `model_temporal_phase_prior.py`
  - temporal phase-prior branch
- `model_temporal_z_mlp_p.py`
  - temporal `z -> MLP -> p` branch

当前非主线 strict 实验的探索分支：

- `model_phase_specific.py`
- `model_phase_prior_mlp.py`
- `model_temporal_phase_specific.py`
- `model_temporal_transformer_phase_specific.py`
- `model_temporal_tsm_z_mlp_p.py`


### 2.2 ReCap process 目录

相关目录：

- `examples/recap/process/`

重要脚本：

- `compute_returns.py`
  - 写入 `meta/returns_<tag>.parquet`
- `compute_advantages.py`
  - 写入 `meta/advantages_<tag>.parquet`
  - 可选地保存完整 201-bin value distribution
- `sample_episode_subset.py`
  - 采样一个固定的 episode subset，可选地 balance success/failure
- `build_episode_split_manifest.py`
  - 将固定的 episode subset 转换为确定性的 train/val episode split
- `episode_subset_utils.py`
  - 解析 subset JSON 和 train/val split JSON 的共享逻辑
- `export_joint_fused_advantages.py`
  - 将 fused value predictions 导回 ReCap advantage tag，供下游 CFG/ReCap 训练使用


## 3. 当前主线方法

以下是当前 strict suite 中使用的方法。

### 3.1 Raw Critic

无 learned `z/p` head。

- 直接使用 advantage parquet 中的 `value_current`
- 作为基准 MSE


### 3.2 Shared MLP Trunk + Two Heads

定义在 `model.py` 中。

架构：

- 输入 feature `[B, D]`
- shared MLP trunk
- phase classification head
- progress regression head with sigmoid
- global progress 由 predicted phase distribution 和 phase progress 推导得出

这是当前 `random200` 运行中最强的 strict baseline。


### 3.3 Temporal Phase Prior

定义在 `model_temporal_phase_prior.py` 中。

架构：

- 输入是 local frozen feature window `[B, T, D]`
- stage encoder 预测 phase distribution
- progress branch 显式地以 stage prior 为条件
- global progress 使用 empirical phase span priors 而非 equal-length phases 计算


### 3.4 Temporal z -> MLP -> p

定义在 `model_temporal_z_mlp_p.py` 中。

架构：

- local frozen feature window -> temporal encoder
- center hidden state `h_t` -> phase head
- 拼接 `[h_t, z_embedding]`
- 小型 MLP 预测 phase progress
- global progress 使用 empirical phase span priors

这是最接近预期因子化结构的 temporal branch：

- 先推断 phase `z`
- 再以 `z` 为条件推断 progress `p`


### 3.5 Logit-Space Fusion MLP

定义在 `logit_fusion_model.py` 中，由 `train_logit_fusion_from_predictions.py` 训练。

输入：

- raw 201-bin critic logits `value_logits_current`
- predicted phase probabilities `phase_probs_pred`
- predicted phase progress `phase_progress_pred`
- predicted global progress `global_progress_pred`

输出：

- `delta_logits`

Fusion 规则：

- `fused_logits = raw_logits + alpha * delta_logits`
- scalar fused value 是 fused categorical return distribution 的期望值


## 4. 数据和产物类型

### 4.1 Dataset 侧产物

这些位于 dataset 的 `meta/` 目录下。

- `phase_progress_semantic.parquet`
  - semantic phase / progress 标签
- `returns_<tag>.parquet`
  - return 和 reward sidecar
- `advantages_<tag>.parquet`
  - ReCap advantage sidecar

重要提示：

- `returns_<tag>.parquet` 是针对完整 dataset 的，而不仅仅是选定的 subset
- subset selection 稍后在 `compute_advantages.py` 和 feature extraction 中应用


### 4.2 Probe 侧产物

这些通常位于：

- `/workspace/results/phase_progress_probe/...`

常见文件：

- `train.pt`、`val.pt`
  - 缓存的 features 和标签
- `head.pt`
  - 训练好的 `z/p` head checkpoint
- `phase_predictions.parquet`
  - frozen frame-level `z/p` predictions
- `logit_fusion.pt`
  - 训练好的 fusion model checkpoint
- `report.json`
  - 脚本的 per-step report
- `suite.log`
  - orchestrator log
- `baseline_summary.json`、`baseline_summary.md`
  - strict baseline suite 摘要
- `fusion_summary.json`、`fusion_summary.md`
  - strict fusion suite 摘要


## 5. 推荐的端到端 Pipeline

本节记录了你当前正在使用的实操 pipeline。

假设的 dataset：

- `/workspace/datasets/recap_libero10_task0/libero10_task0_train`

假设的 value checkpoint：

- `/workspace/models/value_libero_sft_30ep_5k/global_step_5000`

假设的环境设置：

```bash
cd /workspace/RLinf
source switch_env openpi
export REPO_PATH=/workspace/RLinf
export TOKENIZERS_PARALLELISM=false
```


### Step 0. 采样固定的 balanced subset

目的：

- 选择一个确定性的固定 episode 池
- 避免在不同实验之间改变评估池

命令：

```bash
python examples/recap/process/sample_episode_subset.py \
  --dataset_path /workspace/datasets/recap_libero10_task0/libero10_task0_train \
  --output /workspace/results/phase_progress_probe/subsets/libero10_task0_random200_seed42_balanced.json \
  --num_episodes 200 \
  --seed 42 \
  --success_ratio 0.5
```

输出：

- 一个包含选定 episode id 的 JSON manifest


### Step 1. 构建确定性的 train/val episode split

目的：

- 按 episode 划分 fixed subset
- 防止 train 和 val 之间的 frame-level leakage

命令：

```bash
python examples/recap/process/build_episode_split_manifest.py \
  --dataset_path /workspace/datasets/recap_libero10_task0/libero10_task0_train \
  --subset_path /workspace/results/phase_progress_probe/subsets/libero10_task0_random200_seed42_balanced.json \
  --output /workspace/results/phase_progress_probe/subsets/libero10_task0_random200_seed42_balanced_split.json \
  --val_episode_ratio 0.2 \
  --seed 42
```

输出：

- 一个 JSON manifest，包含：
  - `selected_episodes`
  - `train_episodes`
  - `val_episodes`


### Step 2. 计算全 dataset return sidecar

目的：

- 创建一个干净的 return tag，供 `random200` 实验引用
- 这是全 dataset 元数据，而非仅 subset 元数据

命令：

```bash
python examples/recap/process/compute_returns.py \
  --config-name compute_returns \
  data.train_data_paths=[] \
  data.dataset_type=sft \
  data.gamma=1.0 \
  data.failure_reward=-300.0 \
  data.tag=base30ep_random200_logits \
  +data.dataset_path=/workspace/datasets/recap_libero10_task0/libero10_task0_train
```

做了什么：

- 写入 `meta/returns_base30ep_random200_logits.parquet`

重要提示：

- 此处必须使用 `data.train_data_paths=[]`，以便脚本使用 single-dataset branch
- `+data.dataset_path=...` 使用 Hydra append 语法，因为 `dataset_path` 未在基础 YAML 中声明


### Step 3. 计算带 201-bin value logits 的 subset advantage

目的：

- 计算 strict fusion suite 所使用的实际 advantage parquet
- 包含 fusion 训练所需的 `value_logits_current`

命令：

```bash
python examples/recap/process/compute_advantages.py \
  --config-name compute_advantages_base_train_5k \
  advantage.value_checkpoint=/workspace/models/value_libero_sft_30ep_5k/global_step_5000 \
  advantage.tag=base30ep_random200_logits_phase_dist \
  advantage.returns_tag=base30ep_random200_logits \
  advantage.model.critic_expert_variant=gemma_1m \
  advantage.model.tokenizer_path=/workspace/models/gemma-3-270m \
  advantage.model.siglip_path=/workspace/models/siglip2-so400m-patch14-224 \
  advantage.model.gemma3_path=/workspace/models/gemma-3-270m \
  +advantage.save_value_distribution=true \
  +advantage.episode_subset_path=/workspace/results/phase_progress_probe/subsets/libero10_task0_random200_seed42_balanced.json \
  2>&1 | tee /workspace/results/phase_progress_probe/logs/random200_base30ep_balanced/04_compute_advantages_with_dist.log
```

做了什么：

- 使用 `advantage.returns_tag` 加载完整的 return sidecar
- 将样本限制到选定的 200 个 episode
- 写入：
  - `meta/advantages_base30ep_random200_logits_phase_dist.parquet`

重要提示：

- `+advantage.save_value_distribution=true` 是必须的
- 没有它，advantage parquet 将不包含 `value_logits_current`
- 然后 strict fusion suite 将失败


### Step 4. 提取 frozen VLM features

目的：

- 冻结 value critic VLM backbone
- 缓存 train/val episode 的每帧 features

命令：

```bash
python examples/recap/phase_progress_probe/extract_features.py \
  --dataset_path /workspace/datasets/recap_libero10_task0/libero10_task0_train \
  --value_checkpoint /workspace/models/value_libero_sft_30ep_5k/global_step_5000 \
  --siglip_path /workspace/models/siglip2-so400m-patch14-224 \
  --gemma3_path /workspace/models/gemma-3-270m \
  --tokenizer_path /workspace/models/gemma-3-270m \
  --critic_expert_variant gemma_1m \
  --return_min -700 \
  --return_max 0 \
  --episode_split_path /workspace/results/phase_progress_probe/subsets/libero10_task0_random200_seed42_balanced_split.json \
  --output_dir /workspace/results/phase_progress_probe/features_random200_base30ep_balanced \
  --batch_size 16 \
  --seed 42
```

做了什么：

- 加载 value critic checkpoint
- 运行 frozen prefix encoder
- 写入：
  - `train.pt`
  - `val.pt`
  - `extract_args.json`


### Step 5A. 运行 strict baseline suite

目的：

- 比较 raw critic vs predicted linear correction
- 使用 strict train-fit / val-eval 协议

命令：

```bash
python examples/recap/phase_progress_probe/run_strict_baseline_suite.py \
  --features_dir /workspace/results/phase_progress_probe/features_random200_base30ep_balanced \
  --advantages_path /workspace/datasets/recap_libero10_task0/libero10_task0_train/meta/advantages_base30ep_random200_logits_phase_dist.parquet \
  --output_root /workspace/results/phase_progress_probe/strict_baseline_suite_random200_base30ep \
  --return_min -700 \
  --return_max 0 \
  --train_batch_size 256 \
  --analyze_batch_size 1024 \
  --max_epochs 50 \
  --seed 42 \
  --shared_mlp_hidden_dim 640 \
  --shared_mlp_trunk_depth 6 \
  --force
```

做了什么：

- 在 train split 上训练每种 `z/p` 方法
- 在 train 和 val 上预测
- 仅在 train 上拟合 per-phase linear correction
- 仅在 val 上评估 raw vs predicted

输出：

- 每种方法：
  - `train/head.pt`
  - `analysis/phase_predictions.parquet`
  - `strict_eval/report.json`
- suite 摘要：
  - `baseline_summary.json`
  - `baseline_summary.md`
  - `suite.log`


### Step 5B. 运行 strict fusion suite

目的：

- 比较 raw critic vs learned logit-space fusion
- 每种方法有自己的 `z/p` head 和自己的 fusion MLP

命令：

```bash
python examples/recap/phase_progress_probe/run_strict_fusion_suite.py \
  --features_dir /workspace/results/phase_progress_probe/features_random200_base30ep_balanced \
  --advantages_path /workspace/datasets/recap_libero10_task0/libero10_task0_train/meta/advantages_base30ep_random200_logits_phase_dist.parquet \
  --output_root /workspace/results/phase_progress_probe/strict_fusion_suite_random200_base30ep \
  --return_min -700 \
  --return_max 0 \
  --train_batch_size 256 \
  --analyze_batch_size 1024 \
  --fusion_batch_size 256 \
  --fusion_hidden_dim 256 \
  --fusion_depth 2 \
  --fusion_dropout 0.1 \
  --fusion_alpha 1.0 \
  --max_epochs 50 \
  --seed 42 \
  --shared_mlp_hidden_dim 640 \
  --shared_mlp_trunk_depth 6 \
  --force
```

做了什么：

- 对每种方法：
  - 训练 `z/p`
  - 写入 frozen `phase_predictions.parquet`
  - 训练单独的 `logit_fusion.pt`
  - 在 val 上评估 raw vs fused

输出：

- 每种方法：
  - `train/head.pt`
  - `analysis/phase_predictions.parquet`
  - `fusion/logit_fusion.pt`
  - `strict_eval/report.json`
- suite 摘要：
  - `fusion_summary.json`
  - `fusion_summary.md`
  - `suite.log`

重要提示：

- suite 现在会执行 preflight check，确保 `advantages_path` 包含 `value_logits_current`
- 如果缺少该列，suite 会立即失败并显示明确的消息


### Step 6. 将 fused value 导回 ReCap tag

目的：

- 将 probe/fusion 实验桥接回下游 CFG/ReCap 训练

典型使用模式：

- 取一个包含 `value_fused` 的 fused prediction parquet
- 将其与源 advantage parquet 合并
- 重新计算 `value_current`、`value_next` 和 continuous advantage
- 保存新的 `advantages_<tag>.parquet`

主脚本：

- `examples/recap/process/export_joint_fused_advantages.py`

这是与下游训练的集成点。


## 6. 每条命令的实际含义

### `compute_returns.py`

角色：

- 仅生成 return 和 reward sidecar

它不会：

- 训练任何模型
- 使用任何选定的 subset
- 产生 advantage


### `compute_advantages.py`

角色：

- 加载 value critic predictions
- 加载 return sidecar
- 计算 ReCap-style advantage
- 可选地写入完整 value distribution

如果设置了 `+advantage.save_value_distribution=true`，parquet 包含：

- `value_logits_current`
- `value_probs_current`
- `value_logits_next`
- `value_probs_next`

这些列是 learned fusion 所必需的。


### `extract_features.py`

角色：

- 将 frozen value critic VLM 转换为可重用的 frame-level features

它不会：

- 训练 VLM
- 修改 critic checkpoint


### `train_head*.py`

角色：

- 仅在缓存的 features 上训练轻量级 `z/p` head

当前所有 probe head 都保持 backbone frozen。


### `predict_and_analyze*.py`

角色：

- 生成 frame-level predictions
- 计算快速诊断指标

重要警告：

- 旧版 `predict_and_analyze*.py` 脚本适用于快速迭代
- 但其内置的 correction 分析不是最终的 strict 协议
- 对于最终比较，请使用：
  - `evaluate_raw_vs_predicted_strict.py`
  - `run_strict_baseline_suite.py`
  - `evaluate_raw_vs_fusion_strict.py`
  - `run_strict_fusion_suite.py`


### `run_strict_baseline_suite.py`

角色：

- strict raw vs predicted-linear-correction 比较

协议：

- 在 train 上拟合 correction
- 在 val 上评估 frozen correction


### `run_strict_fusion_suite.py`

角色：

- strict raw vs learned-fusion 比较

协议：

- 在 train 上训练 `z/p`
- 冻结 predictions
- 在 train 上训练 fusion MLP
- 在 val 上评估 fused value


## 7. Strictness 与 Leakage 注意事项

本节至关重要。

### 7.1 Train/val split 是按 episode 而非按 frame

这是 probe 的正确协议。

原因：

- 同一 episode 中的相邻 frame 高度相关
- frame-level random split 会将 trajectory identity 泄漏到 validation


### 7.2 Return 是全 dataset 元数据，subset selection 稍后进行

`compute_returns.py` 写入全 dataset 的 sidecar。

Subset restriction 发生在：

- `compute_advantages.py`
- `extract_features.py`

因此以下情况是正确的：

- `returns_base30ep_random200_logits.parquet` 仍然包含所有 dataset 行


### 7.3 旧版分析脚本可能过于乐观

旧版 `predict_and_analyze*.py` 的 correction 逻辑很方便，但对于论文级别的结果，你应该优先使用 strict evaluator 和 suite。

原因：

- 旧版分析代码在一个合并表中拟合并评估 correction
- 这可能产生过于乐观的 validation 数字

Strict 脚本通过以下方式解决此问题：

- 仅在 train 上拟合
- 仅在 val 上评估


### 7.4 Fusion 需要 raw 201-bin logits

Strict fusion 不能仅使用 scalar `value_current` 工作。

它需要：

- `value_logits_current`

因此 `compute_advantages.py` 必须使用以下参数运行：

- `+advantage.save_value_distribution=true`


### 7.5 Val MSE 是 validation split 上的 frame-level MSE

当报告显示 `Val MSE` 时，它的含义是：

- 为每个 val frame 计算 normalized scalar target return
- 将 predicted scalar value 与 normalized target 进行比较
- 在所有 val frame 上取平均

它不是：

- per-episode 平均 MSE
- final-task success rate


## 8. 当前状态快照

截至在 `base30ep` 上运行的 `random200 balanced` strict baseline：

- Raw Critic: `0.019564`
- Shared MLP Trunk + Two Heads: `0.017387`
- Temporal Phase Prior: `0.018237`
- Temporal z->MLP->p: `0.018128`

解读：

- 当前最佳 strict `z/p` baseline 是 `Shared MLP Trunk + Two Heads`
- Temporal 方法相比 raw 仍有改善，但在该轮运行中未能超越更强的 single-frame MLP


## 9. 常见故障模式

### Hydra override 错误

问题：

- `Could not override ...`

原因：

- key 未在 structured config 中预先声明

修复：

- 对于 appended key 使用 `+key=value`

示例：

- `+advantage.save_value_distribution=true`
- `+advantage.episode_subset_path=...`
- `+data.dataset_path=...`


### `compute_returns.py` 崩溃并报 `entry = dict(entry)`

原因：

- `data.train_data_paths` 被作为字符串列表传入
- 脚本期望一个 dict-like entry 的列表

修复：

- 传入一个合适的 dict 列表
- 或者清空列表并使用 single-dataset 模式：

```bash
data.train_data_paths=[] +data.dataset_path=/path/to/dataset
```


### Fusion suite 提示 `value_logits_current` 缺失

原因：

- advantage parquet 在创建时没有启用 `save_value_distribution`

修复：

- 使用以下参数重新计算 advantage：

```bash
+advantage.save_value_distribution=true
```


### 加载 checkpoint 时提示 tokenizer 缺失

原因：

- value checkpoint 目录本身不包含 tokenizer 文件

修复：

- 显式传入：
  - `tokenizer_path`
  - `siglip_path`
  - `gemma3_path`


## 10. 建议的工作方式

快速迭代：

1. 采样 fixed subset（一次）
2. 构建确定性 split（一次）
3. 每个 tag 计算 return（一次）
4. 每个 value checkpoint / tag 计算 advantage（一次）
5. 每个 value checkpoint / split 提取 features（一次）
6. 使用缓存的 features 迭代 head 和 suite

论文级比较：

1. 仅使用 strict suite
2. 不要在最终主表中报告 oracle
3. 仅在 val 上报告 raw vs predicted 或 raw vs fused
4. 所有方法保持 train/val split 固定不变


## 11. 推荐的下一步

一旦 fusion suite 结果令人满意，下一步工程步骤是：

1. 选择一个 fused prediction 来源
2. 将 fused value 导回 `advantages_<new_tag>.parquet`
3. 将下游 CFG/ReCap 训练指向 `data.advantage_tag=<new_tag>`
4. 评估 success rate 是否改善，而不仅仅是 value MSE

这就是 probe 从仅仅是一个分析项目，变成真正 ReCap 训练循环一部分的时刻。
