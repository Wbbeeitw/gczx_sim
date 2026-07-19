#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/RLinf-Pi05-LIBERO-SFT}"
CHECKPOINT="${CHECKPOINT-/workspace/RLinf/persistent_results/task58_bc_h200_2gpu_20260719/checkpoints/global_step_450/actor/model_state_dict/full_weights.pt}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-20260719}"
RESUME="${RESUME:-false}"

case "$MODE" in
  smoke)
    TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
    NUM_EPISODES="${NUM_EPISODES:-1}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-/data/libero_long/smk_test/d0_step450_all10_1ep}"
    MIN_TRAINABLE="${MIN_TRAINABLE:-0}"
    ;;
  batch)
    TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
    NUM_EPISODES="${NUM_EPISODES:-40}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-/data/libero_long/d0_step450_40ep}"
    MIN_TRAINABLE="${MIN_TRAINABLE:-0}"
    ;;
  *)
    echo "usage: bash examples/recap/rounds/collect_d0.sh [smoke|batch]" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EMBODIED_PATH="${EMBODIED_PATH:-/workspace/RLinf/examples/embodiment}"
export REPO_PATH="${REPO_PATH:-/workspace/RLinf}"
export PYTHONPATH="${PYTHONPATH:-/workspace/RLinf}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/libero_long}"
unset RAY_ADDRESS

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "base model not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ -n "$CHECKPOINT" && ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_EPISODES must be a positive integer, got: $NUM_EPISODES" >&2
  exit 2
fi
if [[ "$RESUME" != "true" && "$RESUME" != "false" ]]; then
  echo "RESUME must be true or false, got: $RESUME" >&2
  exit 2
fi

normalized_tasks="${TASKS//,/ }"
read -r -a task_ids <<< "$normalized_tasks"
if [[ "${#task_ids[@]}" -eq 0 ]]; then
  echo "TASKS must contain at least one LIBERO-10 task id" >&2
  exit 2
fi

declare -A seen_tasks=()
dataset_paths=()
for task_id in "${task_ids[@]}"; do
  if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( task_id < 0 || task_id > 9 )); then
    echo "invalid LIBERO-10 task id: $task_id" >&2
    exit 2
  fi
  if [[ -n "${seen_tasks[$task_id]:-}" ]]; then
    echo "duplicate task id: $task_id" >&2
    exit 2
  fi
  seen_tasks[$task_id]=1
  dataset_paths+=("$OUTPUT_ROOT/task${task_id}_${NUM_EPISODES}ep")
done

mkdir -p "$OUTPUT_ROOT/logs"

echo "mode=$MODE"
echo "tasks=${task_ids[*]}"
echo "episodes_per_task=$NUM_EPISODES"
echo "checkpoint=${CHECKPOINT:-<base-model-only>}"
echo "output_root=$OUTPUT_ROOT"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES gpu_id=$GPU_ID"

for index in "${!task_ids[@]}"; do
  task_id="${task_ids[$index]}"
  dataset="${dataset_paths[$index]}"
  log_path="$OUTPUT_ROOT/logs/task${task_id}_${NUM_EPISODES}ep.log"

  if [[ -e "$dataset" ]]; then
    if [[ "$RESUME" != "true" ]]; then
      echo "dataset already exists; choose a new OUTPUT_ROOT or set RESUME=true: $dataset" >&2
      exit 1
    fi
    echo "validating existing task${task_id} dataset before resume"
  else
    echo "collecting task${task_id}: $dataset"
    collect_args=(
      --pretrained_path "$MODEL_PATH"
      --output_dir "$dataset"
      --task_suite_name libero_10
      --task_id "$task_id"
      --num_episodes "$NUM_EPISODES"
      --seed "$SEED"
      --gpu_id "$GPU_ID"
      --config_name pi05_libero
      --model_type openpi
      --action_chunk 5
      --num_steps 5
      --num_steps_wait 10
      --warmup_before_env
      --semantic_trace
      --semantic_trace_task "task${task_id}"
      --semantic_trace_output_name "semantic_trace_task${task_id}"
    )
    if [[ -n "$CHECKPOINT" ]]; then
      collect_args+=(--checkpoint_path "$CHECKPOINT")
    fi
    python examples/recap/process/collect_libero_rollouts.py \
      "${collect_args[@]}" \
      2>&1 | tee "$log_path"
  fi

  python examples/recap/process/validate_libero_rollout_collection.py \
    "$dataset" \
    --expected-task-id "$task_id" \
    --expected-episodes "$NUM_EPISODES" \
    --min-trainable "$MIN_TRAINABLE"
done

echo "D0 $MODE collection and validation passed for ${#task_ids[@]} task(s)."
echo "Re-run the read-only aggregate audit with:"
printf 'python examples/recap/process/validate_libero_rollout_collection.py --expected-episodes %q' "$NUM_EPISODES"
printf ' %q' "${dataset_paths[@]}"
printf '\n'
