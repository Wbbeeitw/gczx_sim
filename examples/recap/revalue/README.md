# Revalue

Revalue 是 ReCap 的 phase/progress-aware value 重估计路径，目前处于维护状态。
它刻意将论文实验面保持得很小：

1. `base`: 使用 raw `ValueCriticModel`/pi0.5 critic advantage tag。
2. `shared_mlp_fusion`: 训练一个 shared MLP z/p head，冻结它，然后训练一个
   logit-space fusion MLP。

Fused 路径严格分为两阶段：

```text
train_zp:
  frozen ValueCriticModel features -> SharedMLPPhaseProgressHead

train_fusion:
  frozen z/p head + raw value logits -> LogitFusionMLP -> fused value
```

输出是标准的 ReCap advantage parquet：

```text
<dataset>/meta/advantages_<output_tag>.parquet
```

下游 CFG/ReCap 训练通过以下方式选择结果：

```yaml
data:
  advantage_tag: <output_tag>
```

## 用法

运行完整的 fused pipeline：

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  data.dataset_path=/home/enine/rlinf_workspace/datasets/... \
  value.checkpoint=/home/enine/rlinf_workspace/results/.../global_step_5000 \
  recap.source_tag=base \
  recap.output_tag=zp_fused
```

分 stage 运行：

```bash
python examples/recap/revalue/revalue.py stage=extract_features
python examples/recap/revalue/revalue.py stage=train_zp
python examples/recap/revalue/revalue.py stage=train_fusion
python examples/recap/revalue/revalue.py stage=predict
python examples/recap/revalue/revalue.py stage=export
```

检查 base tag 是否存在：

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_base \
  recap.source_tag=base
```

## 所需的 Source Tag

Fused 方法要求源 advantage parquet 包含 `value_logits_current`。使用以下方式生成 base tag：

```yaml
advantage:
  tag: base
  save_value_distribution: true
```

非 source 输出（如 features、checkpoints、predictions）应放在 source tree 之外，
例如：

```text
/home/enine/rlinf_workspace/results/revalue/
```
