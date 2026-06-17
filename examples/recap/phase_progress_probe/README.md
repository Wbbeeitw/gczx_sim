# Phase / Progress Probe on ValueCriticModel VLM Backbone

Quick probe that checks whether the frozen VLM backbone of `ValueCriticModel`
already contains enough visual information to predict the semantic phase
`z_t` and phase progress `p_t` defined in `gen_semantic_phase.py`.

All scripts live under `examples/recap/phase_progress_probe/` and do **not**
modify existing code in `examples/recap/process/` or `rlinf/`.

## Pipeline

```
extract_features.py  ->  train_head.py  ->  predict_and_analyze.py
    VLM features           light head          metrics + value-MSE check
```

1. **Extract features**: load the trained value critic, run only the Gemma3
   prefix forward (images + prompt), and save mean-pooled prefix features for
   train/val episodes.
2. **Train head**: a tiny MLP with two heads — phase classifier and progress
   regressor. The VLM is frozen.
3. **Predict & analyze**: predict `phase_pred` / `phase_progress_pred` for every
   frame, and optionally compare the value-MSE reduction obtained with
   **predicted** z/p vs. **oracle** z/p.

## Files

| File | Purpose |
|------|---------|
| `dataset.py` | `PhaseProbeDataset` with episode-level train/val split |
| `model.py` | `VLMBackboneFeatureExtractor` + `PhaseProgressHead` |
| `extract_features.py` | Cache VLM prefix features |
| `train_head.py` | Train the light head on cached features |
| `predict_and_analyze.py` | Predict z/p and evaluate value correction |
| `run_probe.py` / `run_probe.sh` | One-click end-to-end runner |
| `configs/libero_task0.yaml` | Default paths and hyperparameters |

## Usage

### One-click run

```bash
source /opt/venv/openpi/bin/activate
cd /workspace/RLinf
bash examples/recap/phase_progress_probe/run_probe.sh
```

Override any config path:

```bash
bash examples/recap/phase_progress_probe/run_probe.sh \
  --dataset_path /workspace/datasets/recap_libero10_task0/libero10_task0_train \
  --value_checkpoint /workspace/results/value_sft_maniskill/checkpoints/global_step_5000 \
  --siglip_path /workspace/models/siglip2-so400m-patch14-224 \
  --gemma3_path /workspace/models/gemma-3-270m \
  --advantages_path /path/to/advantages.parquet
```

### Step by step

```bash
# 1. extract VLM features
python examples/recap/phase_progress_probe/extract_features.py \
  --dataset_path /workspace/datasets/recap_libero10_task0/libero10_task0_train \
  --value_checkpoint /workspace/results/value_sft_maniskill/checkpoints/global_step_5000 \
  --siglip_path /workspace/models/siglip2-so400m-patch14-224 \
  --gemma3_path /workspace/models/gemma-3-270m \
  --output_dir /workspace/results/phase_progress_probe/features

# 2. train the head
python examples/recap/phase_progress_probe/train_head.py \
  --features_dir /workspace/results/phase_progress_probe/features \
  --output_dir /workspace/results/phase_progress_probe/head

# 3. predict and analyze
python examples/recap/phase_progress_probe/predict_and_analyze.py \
  --features_dir /workspace/results/phase_progress_probe/features \
  --head_checkpoint /workspace/results/phase_progress_probe/head/head.pt \
  --output_dir /workspace/results/phase_progress_probe/analysis \
  --advantages_path /path/to/advantages.parquet \
  --return_min -700.0 --return_max 0.0
```

## Outputs

Under `output_root` (default `/workspace/results/phase_progress_probe`):

```
features/
  train.pt
  val.pt
  extract_args.json
head/
  head.pt
  metrics.json
  train_args.json
analysis/
  phase_predictions.parquet
  advantages_predicted_corrected.parquet  (if --advantages_path given)
  report.json
  analyze_args.json
```

## Interpreting the report

`analysis/report.json` contains:

- `prediction_metrics`
  - `phase_acc`: top-1 phase classification accuracy
  - `phase_progress_mae` / `phase_progress_mse`
  - `global_progress_mae` / `global_progress_mse`
  - `phase_{i}_acc`: per-phase accuracy
- `correction` (only if `--advantages_path` is provided)
  - `mse_raw`: value MSE without correction
  - `mse_oracle`: value MSE using **true** z/p
  - `mse_predicted`: value MSE using **predicted** z/p
  - `oracle_improvement_pct` / `predicted_improvement_pct`
  - `per_phase`: breakdown by true phase

The key question is whether `predicted_improvement_pct` is close to
`oracle_improvement_pct`. If it is, the VLM backbone already encodes enough
phase/progress information to fix the value bias.

## Expected rough thresholds

| Metric | Rough target |
|--------|--------------|
| `phase_acc` | > 70% |
| `phase_progress_mae` | < 0.15 |
| `predicted_improvement_pct` | within 10% of oracle |

If predicted correction is far below oracle, the frozen backbone lacks explicit
phase cues and we should consider adding z/p as an auxiliary task during value
SFT.

## Notes

- Train/val split is done **by episode** to prevent leakage.
- The VLM backbone (SigLIP2 + Gemma3) is fully frozen; only the small head is
  trained.
- Feature extraction is the slowest step; once cached, head training and
  analysis are fast.
