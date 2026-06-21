# Revalue Experiment Workflow

This document records the maintained command path for the thesis Revalue
experiment on LIBERO task0. Run these commands inside the Docker container,
not on the host:

```bash
docker exec -it rlinf_pi06star bash
cd /workspace/RLinf
git pull fork gczx_sim
```

The maintained comparison has only two methods:

1. `base`: use the raw pi0.5/value critic advantage tag.
2. `shared_mlp_fusion`: train a shared MLP z/p head, freeze it, then train a
   logit-space fusion MLP.

The fused method is strictly two-stage:

```text
train_zp:
  frozen VLM features -> SharedMLPPhaseProgressHead

train_fusion:
  frozen z/p head + raw value logits -> LogitFusionMLP -> fused value
```

## Common Variables

```bash
cd /workspace/RLinf

export DATASET=/workspace/datasets/recap_libero10_task0/libero10_task0_train
export VALUE_CKPT=/workspace/results/value_sft/checkpoints/global_step_5000
export SIGLIP=/workspace/models/siglip2-so400m-patch14-224
export GEMMA=/workspace/models/gemma-3-270m

export RUN_NAME=base30ep_random200
export SUBSET=/workspace/results/revalue/manifests/${RUN_NAME}_balanced.json
export SPLIT=/workspace/results/revalue/manifests/${RUN_NAME}_balanced_split.json

export BASE_TAG=base30ep_random200_logits_phase_dist
export ADV_PATH=${DATASET}/meta/advantages_${BASE_TAG}.parquet

export FEATURES_DIR=/workspace/results/phase_progress_probe/features_random200_base30ep_balanced
export REVALUE_ROOT=/workspace/results/revalue/${RUN_NAME}_shared_mlp_fusion
export FUSED_TAG=${RUN_NAME}_shared_mlp_fusion
```

All outputs are under `/workspace/results` or the dataset `meta/` directory.
Do not save features, checkpoints, or predictions under the source tree.

## 1. Sample An Episode Subset

Skip this section if `FEATURES_DIR` and `ADV_PATH` already exist and you want
to reuse the current random200/base30ep data.

Create a balanced 200-episode subset from semantic phase labels:

```bash
python examples/recap/process/sample_episode_subset.py \
  --dataset_path ${DATASET} \
  --phase_path ${DATASET}/meta/phase_progress_semantic.parquet \
  --output ${SUBSET} \
  --num_episodes 200 \
  --seed 42 \
  --success_ratio 0.5
```

Create a deterministic train/val episode split for z/p training and CFG
training:

```bash
python examples/recap/process/build_episode_split_manifest.py \
  --dataset_path ${DATASET} \
  --subset_path ${SUBSET} \
  --output ${SPLIT} \
  --val_episode_ratio 0.2 \
  --seed 42
```

## 2. Compute Returns

Skip this section if the dataset already has the needed returns sidecar.
The command below writes:

```text
${DATASET}/meta/returns_${BASE_TAG}.parquet
```

```bash
cd /workspace/RLinf/examples/recap/process

python compute_returns.py --config-name compute_returns \
  data.train_data_paths="[{'dataset_path':'${DATASET}','type':'rollout'}]" \
  data.dataset_type=rollout \
  data.gamma=1.0 \
  data.failure_reward=-300.0 \
  data.tag=${BASE_TAG} \
  data.num_workers=64
```

## 3. Compute Base Advantages With Value Logits

This is the required raw critic baseline for both `base` and
`shared_mlp_fusion`. The output is:

```text
${DATASET}/meta/advantages_${BASE_TAG}.parquet
```

The important flag is:

```text
advantage.save_value_distribution=true
```

Without it, the parquet will not contain `value_logits_current`, and fusion
training cannot run.

```bash
cd /workspace/RLinf/examples/recap/process

torchrun --nproc_per_node=1 compute_advantages.py --config-name compute_advantages \
  advantage.value_checkpoint=${VALUE_CKPT} \
  advantage.tag=${BASE_TAG} \
  advantage.returns_tag=${BASE_TAG} \
  advantage.episode_subset_path=${SUBSET} \
  advantage.save_value_distribution=true \
  advantage.batch_size=256 \
  advantage.model.critic_expert_variant=gemma_1m \
  advantage.model.siglip_path=${SIGLIP} \
  advantage.model.gemma3_path=${GEMMA} \
  advantage.model.tokenizer_path=${GEMMA} \
  data.train_data_paths="[{'dataset_path':'${DATASET}','robot_type':'libero','type':'rollout','weight':1.0}]" \
  data.robot_type=libero \
  data.model_type=pi05 \
  data.advantage_lookahead_step=10 \
  data.gamma=1.0 \
  data.return_min=-700.0 \
  data.return_max=0.0
```

Validate the base advantage file:

```bash
cd /workspace/RLinf

python - <<'PY'
import os
import pandas as pd

p = os.environ["ADV_PATH"]
df = pd.read_parquet(p)
print(p)
print("rows:", len(df))
print("episodes:", df["episode_index"].nunique())
print("has value_logits_current:", "value_logits_current" in df.columns)
print("columns:", list(df.columns))
PY
```

## 4. Extract Frozen VLM Features

Skip this section if using the existing feature cache:

```bash
export FEATURES_DIR=/workspace/results/phase_progress_probe/features_random200_base30ep_balanced
```

To regenerate feature cache with the maintained Revalue path:

```bash
cd /workspace/RLinf

python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=extract_features \
  data.dataset_path=${DATASET} \
  data.episode_split_path=${SPLIT} \
  data.label_name=phase_progress_semantic \
  value.checkpoint=${VALUE_CKPT} \
  value.siglip_path=${SIGLIP} \
  value.gemma3_path=${GEMMA} \
  value.tokenizer_path=${GEMMA} \
  value.critic_expert_variant=gemma_1m \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  train.extract_batch_size=16 \
  train.device=cuda
```

Validate the feature cache:

```bash
python - <<'PY'
import os
import torch

d = os.environ["FEATURES_DIR"]
for split in ["train", "val"]:
    x = torch.load(f"{d}/{split}.pt", map_location="cpu", weights_only=False)
    print(split, x["features"].shape, x["phase"].shape)
    print(sorted(x.keys()))
PY
```

## 5. Check The Base Method

The base method does not train anything. It only verifies that the raw critic
advantage parquet exists and can be selected by downstream ReCap/CFG training.

```bash
cd /workspace/RLinf

python examples/recap/revalue/revalue.py \
  --config-name revalue_base \
  data.dataset_path=${DATASET} \
  recap.source_tag=${BASE_TAG} \
  recap.source_advantages_path=${ADV_PATH}
```

Downstream CFG/ReCap selects the base method with:

```yaml
data:
  advantage_tag: ${BASE_TAG}
```

## 6. Train The Z/P Head

This stage trains only `SharedMLPPhaseProgressHead` on frozen VLM features.
It does not update the value model and does not train the fusion MLP.

```bash
cd /workspace/RLinf

python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=train_zp \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  train.device=cuda \
  train.batch_size=256 \
  zp.max_epochs=100 \
  zp.early_stop_patience=10
```

Outputs:

```text
${REVALUE_ROOT}/zp_head/zp_head.pt
${REVALUE_ROOT}/zp_head/metrics.json
```

## 7. Freeze Z/P And Train Fusion

This stage loads `${REVALUE_ROOT}/zp_head/zp_head.pt`, freezes it, and trains
only `LogitFusionMLP`.

```bash
cd /workspace/RLinf

python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=train_fusion \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  recap.source_advantages_path=${ADV_PATH} \
  returns.global_min=-700.0 \
  returns.global_max=0.0 \
  value.v_min=-1.0 \
  value.v_max=0.0 \
  value.num_bins=201 \
  train.device=cuda \
  train.batch_size=256 \
  fusion.max_epochs=100 \
  fusion.early_stop_patience=10 \
  fusion.alpha=1.0
```

Outputs:

```text
${REVALUE_ROOT}/fusion/fusion.pt
${REVALUE_ROOT}/fusion/metrics.json
```

## 8. Predict Fused Values

```bash
cd /workspace/RLinf

python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=predict \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  output.predictions_path=${REVALUE_ROOT}/predictions.parquet \
  recap.source_advantages_path=${ADV_PATH} \
  train.device=cuda \
  train.batch_size=512
```

Output:

```text
${REVALUE_ROOT}/predictions.parquet
```

## 9. Export Fused Advantages For ReCap

This stage recomputes the ReCap advantage values using fused values and writes
a standard advantage tag.

```bash
cd /workspace/RLinf

python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  stage=export \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.predictions_path=${REVALUE_ROOT}/predictions.parquet \
  recap.source_advantages_path=${ADV_PATH} \
  recap.output_tag=${FUSED_TAG} \
  recap.lookahead_step=10 \
  recap.gamma=1.0 \
  recap.positive_quantile=0.3 \
  recap.discount_next_value=true \
  recap.export_split=train
```

Outputs:

```text
${DATASET}/meta/advantages_${FUSED_TAG}.parquet
${DATASET}/meta/${FUSED_TAG}_revalue_report.json
```

Downstream CFG/ReCap selects the fused method with:

```yaml
data:
  advantage_tag: ${FUSED_TAG}
```

## 10. Compare Base And Fused Value Prediction

This comparison is done on the frames covered by `predictions.parquet`. The
target is normalized raw return:

```text
return in [-700, 0] -> target value in [-1, 0]
```

```bash
cd /workspace/RLinf

python - <<'PY'
import os
import numpy as np
import pandas as pd

adv_path = os.environ["ADV_PATH"]
pred_path = os.environ["REVALUE_ROOT"] + "/predictions.parquet"
ret_min, ret_max = -700.0, 0.0

adv = pd.read_parquet(adv_path)
pred = pd.read_parquet(pred_path)

merged = pred.merge(
    adv[["episode_index", "frame_index", "return", "value_current"]],
    on=["episode_index", "frame_index"],
    how="inner",
)
merged["target_value"] = (
    (merged["return"] - ret_min) / (ret_max - ret_min) - 1.0
)

def metrics(part, col):
    err = part[col] - part["target_value"]
    return {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
    }

for split_name, part in [
    ("all", merged),
    ("train", merged[merged["split"] == "train"]),
    ("val", merged[merged["split"] == "val"]),
]:
    if len(part) == 0:
        continue
    base = metrics(part, "value_current")
    fused = metrics(part, "value_fused")
    improvement = (
        100.0 * (1.0 - fused["mse"] / base["mse"])
        if base["mse"] > 0.0
        else 0.0
    )
    print(f"\n[{split_name}] rows={len(part)} episodes={part['episode_index'].nunique()}")
    print(f"base  mse={base['mse']:.6f} mae={base['mae']:.6f}")
    print(
        f"fused mse={fused['mse']:.6f} mae={fused['mae']:.6f} "
        f"improvement={improvement:.2f}%"
    )
PY
```

## 11. Train ReCap/CFG With Base Advantages

This run uses:

```text
${DATASET}/meta/advantages_${BASE_TAG}.parquet
```

```bash
cd /workspace/RLinf

bash examples/recap/cfg/run_cfg_sft.sh libero_cfg_openpi \
  runner.logger.experiment_name=cfg_${BASE_TAG} \
  runner.max_epochs=30000 \
  runner.save_interval=3000 \
  data.train_data_paths="[{'dataset_path':'${DATASET}','type':'rollout','weight':1.0}]" \
  data.advantage_tag=${BASE_TAG} \
  data.episode_split_path=${SPLIT} \
  data.episode_split_name=train \
  actor.model.model_path=/workspace/models/pi05_base_pytorch \
  actor.model.openpi.config_name=pi05_libero \
  actor.model.openpi.guidance_type=positive \
  actor.model.openpi.positive_only_conditional=true \
  actor.optim.total_training_steps=30000
```

## 12. Train ReCap/CFG With Fused Advantages

This run uses:

```text
${DATASET}/meta/advantages_${FUSED_TAG}.parquet
```

The command is intentionally identical to the base run except for
`runner.logger.experiment_name` and `data.advantage_tag`.

```bash
cd /workspace/RLinf

bash examples/recap/cfg/run_cfg_sft.sh libero_cfg_openpi \
  runner.logger.experiment_name=cfg_${FUSED_TAG} \
  runner.max_epochs=30000 \
  runner.save_interval=3000 \
  data.train_data_paths="[{'dataset_path':'${DATASET}','type':'rollout','weight':1.0}]" \
  data.advantage_tag=${FUSED_TAG} \
  data.episode_split_path=${SPLIT} \
  data.episode_split_name=train \
  actor.model.model_path=/workspace/models/pi05_base_pytorch \
  actor.model.openpi.config_name=pi05_libero \
  actor.model.openpi.guidance_type=positive \
  actor.model.openpi.positive_only_conditional=true \
  actor.optim.total_training_steps=30000
```

## Short Path For Existing Random200 Artifacts

Use this when these two files already exist:

```text
/workspace/results/phase_progress_probe/features_random200_base30ep_balanced
/workspace/datasets/recap_libero10_task0/libero10_task0_train/meta/advantages_base30ep_random200_logits_phase_dist.parquet
```

```bash
cd /workspace/RLinf

export DATASET=/workspace/datasets/recap_libero10_task0/libero10_task0_train
export FEATURES_DIR=/workspace/results/phase_progress_probe/features_random200_base30ep_balanced
export ADV_PATH=/workspace/datasets/recap_libero10_task0/libero10_task0_train/meta/advantages_base30ep_random200_logits_phase_dist.parquet
export BASE_TAG=base30ep_random200_logits_phase_dist
export RUN_NAME=base30ep_random200
export REVALUE_ROOT=/workspace/results/revalue/${RUN_NAME}_shared_mlp_fusion
export FUSED_TAG=${RUN_NAME}_shared_mlp_fusion

python examples/recap/revalue/revalue.py --config-name revalue_base \
  data.dataset_path=${DATASET} \
  recap.source_tag=${BASE_TAG} \
  recap.source_advantages_path=${ADV_PATH}

python examples/recap/revalue/revalue.py --config-name revalue_shared_mlp_fusion stage=train_zp \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  train.device=cuda

python examples/recap/revalue/revalue.py --config-name revalue_shared_mlp_fusion stage=train_fusion \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  recap.source_advantages_path=${ADV_PATH} \
  returns.global_min=-700.0 \
  returns.global_max=0.0 \
  train.device=cuda

python examples/recap/revalue/revalue.py --config-name revalue_shared_mlp_fusion stage=predict \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.features_dir=${FEATURES_DIR} \
  output.predictions_path=${REVALUE_ROOT}/predictions.parquet \
  recap.source_advantages_path=${ADV_PATH} \
  train.device=cuda

python examples/recap/revalue/revalue.py --config-name revalue_shared_mlp_fusion stage=export \
  data.dataset_path=${DATASET} \
  output.root=${REVALUE_ROOT} \
  output.predictions_path=${REVALUE_ROOT}/predictions.parquet \
  recap.source_advantages_path=${ADV_PATH} \
  recap.output_tag=${FUSED_TAG}
```

After this short path, run CFG/ReCap twice:

```bash
data.advantage_tag=${BASE_TAG}
data.advantage_tag=${FUSED_TAG}
```
