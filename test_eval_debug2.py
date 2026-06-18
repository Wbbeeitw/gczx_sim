import os, sys
sys.path.insert(0, "/workspace/RLinf")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

print("1. imports...", flush=True)
import torch
from toolkits.eval_scripts_openpi import setup_logger, setup_policy
print("2. imports OK", flush=True)

import argparse
args = argparse.Namespace(
    config_name="pi05_libero",
    pretrained_path="/workspace/models/RLinf-Pi05-LIBERO-SFT",
    num_steps=5,
    action_chunk=5,
)
print("3. calling setup_policy...", flush=True)
policy = setup_policy(args)
print("4. policy setup OK!", flush=True)

# Now try env + policy inference together
print("5. creating env...", flush=True)
from libero.libero import benchmark, get_libero_path
import pathlib
task_suite = benchmark.get_benchmark_dict()["libero_10"]()
task = task_suite.get_task(0)
full = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
from libero.libero.envs import OffScreenRenderEnv
env = OffScreenRenderEnv(bddl_file_name=str(full), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
env.seed(42)
env.reset()
print("6. env reset OK!", flush=True)

# Test inference
import numpy as np
obs = env.set_init_state(task_suite.get_task_init_states(0)[0])
img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
print("7. running policy inference...", flush=True)
action = policy.infer({
    "observation/image": img,
    "observation/wrist_image": wrist,
    "observation/state": np.zeros(7),
    "prompt": task.language,
})
print("8. inference OK! actions:", action["actions"].shape if "actions" in action else action, flush=True)
