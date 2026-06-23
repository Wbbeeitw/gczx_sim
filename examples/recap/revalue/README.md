# Revalue

Revalue is the maintained phase/progress-aware value re-estimation workflow for
ReCap. It replaces the old scattered experiment scripts with one entry point:

```bash
python examples/recap/revalue/revalue.py
```

The maintained comparison has only two methods:

1. `base`: raw pi0.5/value critic return logits and ReCap advantages.
2. `shared_mlp_fusion`: train a shared MLP z/p head, freeze it, then train a
   logit-space fusion MLP to correct the raw value distribution.

## Workflow

The full fused workflow is:

```text
prepare_data
  -> sample balanced episode subset
  -> write one train/val/test manifest

build_base
  -> compute returns sidecar
  -> compute raw critic base advantages with value_logits_current

extract_features
  -> cache frozen VLM features for train/val

train_zp
  -> train SharedMLPPhaseProgressHead

train_fusion
  -> freeze z/p head
  -> train LogitFusionMLP

predict
  -> write frame-level fused values

export
  -> write meta/advantages_<output_tag>.parquet for ReCap

compare_returns
  -> write return prediction comparison JSON

train_cfg
  -> train downstream CFG on the exported advantage tag

eval_policy
  -> run LIBERO embodied evaluation for a specified checkpoint

collect_rollouts
  -> collect LIBERO rollouts from a specified policy/checkpoint
```

Run the whole fused path:

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_shared_mlp_fusion \
  data.dataset_path=/workspace/datasets/recap_libero10_task0/libero10_task0_train \
  value.checkpoint=/workspace/results/value_sft/checkpoints/global_step_5000 \
  value.siglip_path=/workspace/models/siglip2-so400m-patch14-224 \
  value.gemma3_path=/workspace/models/gemma-3-270m \
  value.tokenizer_path=/workspace/models/gemma-3-270m \
  output.root=/workspace/results/revalue/libero_task0_shared_mlp \
  base.tag=base30ep_random200_logits_phase_dist \
  recap.output_tag=base30ep_random200_shared_mlp_fusion \
  manifest.num_episodes=200
```

Run one stage at a time:

```bash
python examples/recap/revalue/revalue.py stage=prepare_data
python examples/recap/revalue/revalue.py stage=build_base
python examples/recap/revalue/revalue.py stage=extract_features
python examples/recap/revalue/revalue.py stage=train_zp
python examples/recap/revalue/revalue.py stage=train_fusion
python examples/recap/revalue/revalue.py stage=predict
python examples/recap/revalue/revalue.py stage=export
python examples/recap/revalue/revalue.py stage=compare_returns
python examples/recap/revalue/revalue.py stage=train_cfg
python examples/recap/revalue/revalue.py stage=eval_policy
python examples/recap/revalue/revalue.py stage=collect_rollouts
```

Run only the base data path:

```bash
python examples/recap/revalue/revalue.py \
  --config-name revalue_base \
  data.dataset_path=/workspace/datasets/recap_libero10_task0/libero10_task0_train \
  value.checkpoint=/workspace/results/value_sft/checkpoints/global_step_5000 \
  base.tag=base30ep_random200_logits_phase_dist \
  manifest.num_episodes=200
```

## Outputs

For `output.root=/workspace/results/revalue/run`, Revalue writes:

```text
/workspace/results/revalue/run/episode_manifest.json
/workspace/results/revalue/run/features/{train,val}.pt
/workspace/results/revalue/run/zp_head/zp_head.pt
/workspace/results/revalue/run/fusion/fusion.pt
/workspace/results/revalue/run/predictions.parquet
/workspace/results/revalue/run/return_compare.json
```

The dataset receives standard ReCap sidecars:

```text
<dataset>/meta/returns_<base.tag>.parquet
<dataset>/meta/advantages_<base.tag>.parquet
<dataset>/meta/advantages_<recap.output_tag>.parquet
```

Downstream CFG/ReCap training keeps using the original infra. Select the raw
base tag or fused tag with:

```yaml
data:
  advantage_tag: <base.tag or recap.output_tag>
```

Detailed commands for the current LIBERO task0 experiment are in
`EXPERIMENT_WORKFLOW.md`.

## Downstream Integration

Revalue can now continue past export and drive the downstream smoke workflow in
one place.

Train CFG from the exported fused advantage tag:

```bash
python examples/recap/revalue/revalue.py \
  stage=train_cfg \
  cfg_train.enabled=true \
  cfg_train.base_model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  cfg_train.advantage_tag=base30ep_random200_resplit100_100_shared_mlp_fusion_valonly \
  cfg_train.episode_split_name=val
```

Evaluate a checkpoint with conservative LIBERO eval defaults:

```bash
python examples/recap/revalue/revalue.py \
  stage=eval_policy \
  policy_eval.enabled=true \
  policy_eval.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  policy_eval.checkpoint_path=/workspace/RLinf/logs/.../full_weights.pt \
  policy_eval.model_type=cfg_model
```

Collect LIBERO rollouts from either the original SFT model or a trained CFG
checkpoint:

```bash
python examples/recap/revalue/revalue.py \
  stage=collect_rollouts \
  rollout_collect.enabled=true \
  rollout_collect.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  rollout_collect.checkpoint_path=/workspace/RLinf/logs/.../full_weights.pt \
  rollout_collect.model_type=cfg_model \
  rollout_collect.output_dir=/workspace/datasets/libero_task0_cfg_rollouts
```

The new stage summaries are written under `output.root` by default:

```text
<output.root>/downstream_train/train_cfg_summary.json
<output.root>/policy_eval/eval_policy_summary.json
<output.root>/collected_rollouts/collection_summary.json
```
