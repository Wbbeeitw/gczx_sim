# Iterated ReCap rounds

`run_round.py` turns one pinned YAML file into a resumable LIBERO round:

```text
collect -> human audit -> fit critic -> export child view -> train -> evaluate
```

Large retrainable checkpoints use `paths.results_root`; datasets, feature
caches, reports, comparisons, and logs use `paths.exp_root` or the named
dataset paths under `/data/libero_long`.

## Run Round 2

Inside the RLinf OpenPI container:

```bash
cd /workspace/RLinf
source switch_env openpi

python examples/recap/rounds/run_round.py \
  --config examples/recap/rounds/config/task1_iter02_prcfg.yaml \
  --stage all
```

The first invocation collects the new rollouts, renders the requested phase
visualizations, and stops. Review `collection_summary.json`, the simulator
semantic trace, its audit CSV, and the visualizations. Then resume with the
same command plus the audit confirmation:

```bash
python examples/recap/rounds/run_round.py \
  --config examples/recap/rounds/config/task1_iter02_prcfg.yaml \
  --stage all \
  --confirm-audit
```

The second invocation skips current steps and runs the remaining critic,
policy-data export, policy training, evaluation, and result-recording steps.
Each step has its own command fingerprint and report, so changing a relevant
YAML value invalidates only the affected stage instead of silently accepting
an existing file.

The final result JSON exposes three primary experiment metrics under
`primary_metrics`: fused critic frame MAE, policy success rate, and mean ACT of
successful episodes. Phase/progress and fusion validation metrics remain in
the same record as diagnostic metrics.

Within `fit_critic`, execution remains strictly ordered: returns and Value
training finish first, then frozen Value/VLM features and base advantages are
built, then the z/p head is trained, and fusion is trained only after the z/p
checkpoint exists. Command compaction does not parallelize these dependencies.

## Recovery and overrides

Repeat the same command after an interruption. A step is skipped only when its
report fingerprint matches and every declared artifact still exists.

Use `--set KEY=VALUE` for an intentional one-off override. The override is part
of the command fingerprint:

```bash
python examples/recap/rounds/run_round.py \
  --config examples/recap/rounds/config/task1_iter02_prcfg.yaml \
  --stage eval_policy \
  --set eval.eval_rollout_epoch=10
```

`--force` reruns compute steps. Replacing a collected or merged dataset also
requires `--overwrite-datasets`; verify the YAML paths before using it.

## Later CFG/FACD rounds

Clone the previous YAML and update all round-specific paths, tags, episode
counts, parent checkpoint, and result labels. Training, collection, and
evaluation expose the same CFG controls:

```yaml
policy:
  strategy: csa_soft
  guidance_type: dual_scale
  positive_only_conditional: false
  guidance_scale: 1.0
  negative_guidance_scale: 0.0

collect:
  guidance_type: dual_scale
  positive_only_conditional: false
  guidance_scale: 1.0
  negative_guidance_scale: 0.0

eval:
  guidance_type: dual_scale
  positive_only_conditional: false
  guidance_scale: 1.0
  negative_guidance_scale: 0.0
```

Set the actual FACD strategy and positive/negative scales from the experiment
specification. The orchestrator does not automatically switch methods by round
number.
