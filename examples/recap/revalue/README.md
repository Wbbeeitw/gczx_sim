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

build_base_from_cache
  -> reuse train.pt / val.pt raw logits and raw values
  -> rebuild meta/advantages_<base.tag>.parquet without a second image pass

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
python examples/recap/revalue/revalue.py stage=build_base_from_cache
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

Optional cache-first base path:

```bash
python examples/recap/revalue/revalue.py stage=extract_features
python examples/recap/revalue/revalue.py stage=build_base_from_cache
```

This keeps the legacy `build_base` stage available, but avoids re-running value
inference over raw images when the feature cache already exists.

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

For LIBERO-10 task1 through task7 collection, add
`rollout_collect.semantic_trace=true` and set
`rollout_collect.semantic_trace_task` to the matching task. This keeps
privileged simulator state out of policy observations and writes task-specific
artifacts under the collected dataset's `meta/` directory:

```text
semantic_trace_task1.parquet
phase_progress_semantic_trace_task1.parquet
semantic_trace_task1_audit.csv
semantic_trace_task1_metadata.json
```

For task2, use `semantic_trace_task2` as the output name. Its trace records
the native LIBERO predicates for stove activation and moka pot placement on
the cook region; the frypan is an audit-only distractor. Use
`data.label_name=phase_progress_semantic_trace_task2`,
`manifest.success_phase=3`, and `manifest.num_phases=4` when training ReValue
from task2 labels.

For task3, use `semantic_trace_task3` as the output name. Its trace records
the native LIBERO predicates for black bowl containment in the bottom drawer
and drawer closure; wine bottle and wine rack contacts are audit-only
distractors. Use `data.label_name=phase_progress_semantic_trace_task3`,
`manifest.success_phase=3`, and `manifest.num_phases=4` when training ReValue
from task3 labels.

For task4, use `semantic_trace_task4` as the output name. Its trace records
the native LIBERO predicates for each mug's correct plate placement; a mug on
the opposite plate and red coffee mug contacts are audit-only conditions. Use
`data.label_name=phase_progress_semantic_trace_task4`,
`manifest.success_phase=3`, and `manifest.num_phases=4` when training ReValue
from task4 labels.

For task5, use `semantic_trace_task5` as the output name. Its trace records
native book containment in the caddy's back compartment and an auditable
pre-insertion boundary based on book-to-compartment distance and caddy contact.
Use `data.label_name=phase_progress_semantic_trace_task5`,
`manifest.success_phase=3`, and `manifest.num_phases=4` when training ReValue
from task5 labels.

For task6, use `semantic_trace_task6` as the output name. Its trace records
native white-mug-on-plate and chocolate-pudding-in-right-region predicates.
Incorrect pudding placement and red coffee mug contact are audit-only. Use
`data.label_name=phase_progress_semantic_trace_task6`,
`manifest.success_phase=3`, and `manifest.num_phases=4` when training ReValue
from task6 labels.

The semantic trace supports the four task1 stages, including a B2 transition
when either one object is stably in the basket or both objects are jointly
controlled near the basket. Use `data.label_name=phase_progress_semantic_trace_task1`,
`manifest.success_phase=3`, and `manifest.num_phases=4` when training ReValue
from these labels.

The new stage summaries are written under `output.root` by default:

```text
<output.root>/downstream_train/train_cfg_summary.json
<output.root>/policy_eval/eval_policy_summary.json
<output.root>/collected_rollouts/collection_summary.json
```
