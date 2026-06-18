import os, sys
sys.path.insert(0, "/workspace/RLinf")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import collections, math, pathlib
import imageio, numpy as np, tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from toolkits.eval_scripts_openpi import setup_logger, setup_policy

# --- Simulate main() start ---
LIBERO_ENV_RESOLUTION = 256

print("=== create logger ===", flush=True)
logger = setup_logger("test", "/workspace/RLinf/logs/eval/test")
print("OK", flush=True)

print("=== create policy ===", flush=True)
import argparse
args = argparse.Namespace(
    config_name="pi05_libero",
    pretrained_path="/workspace/models/RLinf-Pi05-LIBERO-SFT",
    num_steps=5, action_chunk=5, seed=42,
    task_suite_name="libero_10", num_trials_per_task=1,
    num_steps_wait=10, num_save_videos=10, video_temp_subsample=10,
    log_dir="/workspace/RLinf/logs/eval", exp_name="test",
)
policy = setup_policy(args)
print("OK", flush=True)

print("=== create benchmark ===", flush=True)
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()
task = task_suite.get_task(0)
print("task:", task.language, flush=True)

print("=== create env ===", flush=True)
task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(task_bddl_file), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
env.seed(42)
env.reset()
print("env OK", flush=True)

print("=== run inference ===", flush=True)
policy.reset()
obs = env.set_init_state(task_suite.get_task_init_states(0)[0])
img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
state = np.concatenate((obs["robot0_eef_pos"], np.zeros(3), obs["robot0_gripper_qpos"]))
action_chunk = policy.infer({
    "observation/image": img,
    "observation/wrist_image": wrist,
    "observation/state": state,
    "prompt": str(task.language),
})["actions"]
print("inference OK! action shape:", action_chunk.shape, flush=True)
env.close()
print("=== ALL DONE ===", flush=True)
