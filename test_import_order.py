import os, sys
sys.path.insert(0, "/workspace/RLinf")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

print("=== step 1: collections, math, pathlib ===", flush=True)
import collections, math, pathlib
print("OK", flush=True)

print("=== step 2: imageio ===", flush=True)
import imageio
print("OK", flush=True)

print("=== step 3: numpy ===", flush=True)
import numpy as np
print("OK", flush=True)

print("=== step 4: tqdm ===", flush=True)
import tqdm
print("OK", flush=True)

print("=== step 5: libero benchmark ===", flush=True)
from libero.libero import benchmark, get_libero_path
print("OK", flush=True)

print("=== step 6: OffScreenRenderEnv ===", flush=True)
from libero.libero.envs import OffScreenRenderEnv
print("OK", flush=True)

print("=== step 7: toolkits ===", flush=True)
from toolkits.eval_scripts_openpi import setup_logger, setup_policy
print("OK", flush=True)

print("=== ALL IMPORTS PASSED - segfault is later ===", flush=True)
