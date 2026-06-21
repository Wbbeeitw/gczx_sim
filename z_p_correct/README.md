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
│   ├── models/heads/        # Registered head architectures
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

## Supported methods

| Name | File | Description |
|---|---|---|
| `base` | N/A (raw critic) | Raw ValueCriticModel baseline |
| `shared_mlp` | `models/heads/phase_progress_head.py` | Shared trunk + phase/progress heads + fusion |
| `temporal_phase_prior` | `models/heads/temporal_phase_prior.py` | Temporal encoder + phase-prior progress + fusion |
| `temporal_z_mlp_p` | `models/heads/temporal_z_mlp_p.py` | Temporal encoder → z → z-conditioned MLP → p + fusion |

**Note:** `OnlineFusionCritic` currently supports single-frame heads only
(`base`, `shared_mlp`). Temporal online integration is planned for later.

## Important: return range

This project uses a return normalization range of **`-700.0` to `0.0`** for the
LIBERO value critic. Make sure `--return_min` and `--return_max` match your
advantages parquet and value checkpoint.

## Usage

### 1. Extract features

```bash
python z_p_correct/scripts/extract_features.py \
  --dataset_path /workspace/datasets/recap_libero10_task0/libero10_task0_train \
  --value_checkpoint /workspace/models/value_libero_sft_30ep_5k/global_step_5000 \
  --siglip_path /workspace/models/siglip2-so400m-patch14-224 \
  --gemma3_path /workspace/models/gemma-3-270m \
  --tokenizer_path /workspace/models/gemma-3-270m \
  --critic_expert_variant gemma_1m \
  --output_dir /workspace/results/z_p_correct/features_random200 \
  --max_episodes 200
```

Outputs `train.pt` and `val.pt` with features, labels, raw_logits, and atoms.

### 2. Train one method

```bash
python z_p_correct/scripts/train_z_p_correct.py \
  --head_type shared_mlp \
  --features_dir /workspace/results/phase_progress_probe/features_random200_base30ep_balanced \
  --advantages_path /workspace/datasets/recap_libero10_task0/libero10_task0_train/meta/advantages_base30ep_random200_logits_phase_dist.parquet \
  --output_dir /workspace/results/z_p_correct/runs \
  --return_min -700.0 \
  --return_max 0.0
```

This runs both training stages and saves `head.pt`, `fusion.pt`,
`predictions.parquet` (with `value_fused` column), and `eval_report.json`.

### 3. Compare all four methods

```bash
python z_p_correct/scripts/run_comparison.py \
  --config z_p_correct/config/compare.yaml \
  --features_dir /workspace/results/phase_progress_probe/features_random200_base30ep_balanced \
  --advantages_path /workspace/datasets/recap_libero10_task0/libero10_task0_train/meta/advantages_base30ep_random200_logits_phase_dist.parquet \
  --output_dir /workspace/results/z_p_correct/comparison_random200
```

### 4. Export fused advantages

```bash
python -c "
from z_p_correct.integration import export_fused_advantages
export_fused_advantages(
    dataset_path='/workspace/datasets/recap_libero10_task0/libero10_task0_train',
    predictions_path='/workspace/results/z_p_correct/runs/shared_mlp/predictions.parquet',
    source_tag='base30ep_random200_logits_phase_dist',
    output_tag='shared_mlp_fused',
    lookahead_step=16,
    gamma=1.0,
)
"
```

## Design constraints

- **No modifications to existing RLinf files.** All code lives under
  `RLinf/z_p_correct/`.
- The fusion trainer explicitly freezes the z/p head before optimizing the
  fusion MLP.
- Episode-level train/val splits are preserved via the feature cache.
