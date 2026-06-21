# Revalue

Revalue is the maintained phase/progress-aware value re-estimation path for
ReCap. It keeps the thesis experiment surface intentionally small:

1. `base`: use the raw `ValueCriticModel`/pi0.5 critic advantage tag.
2. `shared_mlp_fusion`: train a shared MLP z/p head, freeze it, then train a
   logit-space fusion MLP.

The fused path is strictly two-stage:

```text
train_zp:
  frozen ValueCriticModel features -> SharedMLPPhaseProgressHead

train_fusion:
  frozen z/p head + raw value logits -> LogitFusionMLP -> fused value
```

The output is a standard ReCap advantage parquet:

```text
<dataset>/meta/advantages_<output_tag>.parquet
```

Downstream CFG/ReCap training then selects the result with:

```yaml
data:
  advantage_tag: <output_tag>
```

## Usage

Run the full fused pipeline:

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  data.dataset_path=/home/enine/rlinf_workspace/datasets/... \
  value.checkpoint=/home/enine/rlinf_workspace/results/.../global_step_5000 \
  recap.source_tag=base \
  recap.output_tag=zp_fused
```

Run stages separately:

```bash
python examples/recap/revalue/revalue.py stage=extract_features
python examples/recap/revalue/revalue.py stage=train_zp
python examples/recap/revalue/revalue.py stage=train_fusion
python examples/recap/revalue/revalue.py stage=predict
python examples/recap/revalue/revalue.py stage=export
```

Check that a base tag exists:

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_base \
  recap.source_tag=base
```

## Required Source Tag

The fused method requires the source advantage parquet to contain
`value_logits_current`. Generate the base tag with:

```yaml
advantage:
  tag: base
  save_value_distribution: true
```

Non-source outputs such as features, checkpoints, and predictions should live
outside the source tree, for example under:

```text
/home/enine/rlinf_workspace/results/revalue/
```
