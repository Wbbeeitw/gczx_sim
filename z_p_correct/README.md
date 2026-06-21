# z_p_correct

Standalone package for phase/progress-aware value correction in RLinf.

## Idea

Use a lightweight head to predict semantic phase `z_t` and progress `p_t` from
frozen VLM features, then train a small fusion MLP that corrects the raw critic
value distribution in logit space:

```text
fused_logits = raw_logits + alpha * delta_logits(z_t, p_t)
```

Training is strictly two-stage:

1. **Train z/p head** on cached VLM features.
2. **Freeze the head**, then **train the fusion MLP only**.

## Structure

```text
z_p_correct/
├── config/                  # Example YAML configs
├── z_p_correct/
│   ├── models/heads/        # 4 registered head architectures
│   ├── models/fusion/       # LogitFusionMLP
│   ├── data/                # Dataset, feature cache, fusion dataset
│   ├── training/            # HeadTrainer + FusionTrainer
│   ├── analysis/            # Predictor + Evaluator
│   ├── integration/         # OnlineFusionCritic + export_fused_advantages
│   ├── registry/            # head registry
│   └── utils/               # metrics + path helpers
├── scripts/
│   ├── extract_features.py
│   ├── train_z_p_correct.py
│   └── run_comparison.py
└── tests/
```

## Supported heads

| Name | File | Description |
|---|---|---|
| `base_shared_mlp` | `models/heads/base_shared_mlp.py` | Shared trunk + phase/progress heads |
| `our_phase_progress` | `models/heads/phase_progress_head.py` | Our shared-trunk head |
| `temporal_phase_prior` | `models/heads/temporal_phase_prior.py` | Temporal encoder + phase-prior progress |
| `temporal_z_mlp_p` | `models/heads/temporal_z_mlp_p.py` | Temporal encoder → z → z-conditioned MLP → p |

## Usage

### 1. Extract features

```bash
python z_p_correct/scripts/extract_features.py \
  --dataset_path /workspace/datasets/recap_libero10_task0/libero10_task0_train \
  --value_checkpoint /workspace/models/value_libero_sft_30ep_5k/global_step_5000 \
  --siglip_path /path/to/siglip2 \
  --gemma3_path /path/to/gemma3-270m \
  --output_dir /workspace/results/z_p_correct/features_random200 \
  --max_episodes 200
```

Outputs `train.pt` and `val.pt` with features, labels, raw_logits, and atoms.

### 2. Train one method

```bash
python z_p_correct/scripts/train_z_p_correct.py \
  --head_type our_phase_progress \
  --features_dir /workspace/results/z_p_correct/features_random200 \
  --advantages_path /workspace/datasets/.../meta/advantages_base30ep_random200_logits_phase_dist.parquet \
  --output_dir /workspace/results/z_p_correct/runs
```

This runs both training stages and saves `head.pt`, `fusion.pt`, predictions,
and an `eval_report.json`.

### 3. Compare multiple methods

```bash
python z_p_correct/scripts/run_comparison.py \
  --config z_p_correct/config/compare.yaml
```

Or directly:

```bash
python z_p_correct/scripts/run_comparison.py \
  --features_dir /workspace/results/z_p_correct/features_random200 \
  --advantages_path /workspace/datasets/.../meta/advantages_....parquet \
  --output_dir /workspace/results/z_p_correct/comparison
```

### 4. Export fused advantages

```python
from z_p_correct.integration import export_fused_advantages

export_fused_advantages(
    dataset_path="/workspace/datasets/...",
    predictions_path="/workspace/results/z_p_correct/runs/our_phase_progress/predictions.parquet",
    source_tag="base30ep_random200_logits_phase_dist",
    output_tag="our_phase_progress_fused",
    lookahead_step=16,
    gamma=1.0,
)
```

## Design constraints

- **No modifications to existing RLinf files.** All code lives under
  `RLinf/z_p_correct/`.
- The fusion trainer explicitly freezes the z/p head before optimizing the
  fusion MLP.
- Episode-level train/val splits are preserved via the feature cache.
