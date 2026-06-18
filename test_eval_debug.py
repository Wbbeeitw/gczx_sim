import os, sys
sys.path.insert(0, "/workspace/RLinf")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

print("1. imports start", flush=True)
import torch
print("2. torch OK, cuda:", torch.cuda.is_available(), flush=True)
print("3. libero import...", flush=True)
from libero.libero import benchmark, get_libero_path
print("4. libero OK", flush=True)
print("5. setup_policy import...", flush=True)
from toolkits.eval_scripts_openpi import setup_logger, setup_policy
print("6. setup_policy import OK", flush=True)

import argparse
args = argparse.Namespace(
    config_name="pi05_libero",
    pretrained_path="/workspace/models/RLinf-Pi05-LIBERO-SFT",
)
print("7. calling setup_policy...", flush=True)
policy = setup_policy(args)
print("8. policy setup OK!", flush=True)
