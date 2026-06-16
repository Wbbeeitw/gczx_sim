import argparse
import os
import sys

sys.path.insert(0, "/workspace/RLinf")

import gymnasium as gym
import numpy as np

import examples.recap.maniskill.sequential_pick_cube_env
from mani_skill.utils.wrappers.record import RecordEpisode


def move_to(env, target_pos, gripper_action, max_steps=200, pos_tol=0.002):
    """P-controller to move tcp to target_pos."""
    agent = env.unwrapped.agent
    integral = np.zeros(3)
    for _ in range(max_steps):
        current_pos = agent.tcp_pose.p[0].cpu().numpy()
        err = target_pos - current_pos
        dist = np.linalg.norm(err)
        action = np.zeros(7)
        integral += err * 0.01
        vel = err * 5.0 + integral
        action[:3] = np.clip(vel, -0.1, 0.1)
        action[6] = gripper_action
        env.step(action)
        if dist < pos_tol:
            return True
    return False


def pick_and_place(env, cube_pos, goal_pos, lift_height=0.10):
    """Pick cube at cube_pos and place at goal_pos."""
    agent = env.unwrapped.agent
    above_cube = cube_pos + np.array([0.0, 0.0, lift_height])
    above_goal = goal_pos + np.array([0.0, 0.0, lift_height])

    # 1. approach above cube
    move_to(env, above_cube, 1.0)
    # 2. descend to grasp height
    move_to(env, cube_pos, 1.0)
    # 3. close gripper while pushing down slightly
    for _ in range(30):
        action = np.zeros(7)
        action[2] = -0.01
        action[6] = -1.0
        env.step(action)
    # verify grasp
    if not agent.is_grasping(env.unwrapped.cubes[env.unwrapped.current_target.item()]):
        return False
    # 4. lift
    move_to(env, above_cube, -1.0)
    # 5. move above goal
    move_to(env, above_goal, -1.0)
    # 6. descend to place height
    move_to(env, goal_pos, -1.0)
    # 7. open gripper
    for _ in range(20):
        action = np.zeros(7)
        action[6] = 1.0
        env.step(action)
    # 8. lift
    move_to(env, above_goal, 1.0)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="/workspace/datasets/sequential_pickcube_raw")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make(
        "SequentialPickCube-v1",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
    )
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_trajectory=True,
        trajectory_name="trajectory",
        save_video=True,
        video_fps=10,
        save_on_reset=False,  # manually flush
    )

    env_cfg = env.unwrapped
    cube_center_z = env_cfg.pedestal_height + env_cfg.cube_half_size
    cube_positions = [
        np.array([env_cfg.CUBE_POSITIONS[i][0], env_cfg.CUBE_POSITIONS[i][1], cube_center_z])
        for i in range(3)
    ]
    goal_positions = [
        np.array([env_cfg.GOAL_POSITIONS[i][0], env_cfg.GOAL_POSITIONS[i][1], cube_center_z])
        for i in range(3)
    ]

    success_count = 0
    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed_start + ep)
        episode_ok = True
        for i in range(3):
            ok = pick_and_place(env, cube_positions[i], goal_positions[i])
            if not ok:
                episode_ok = False
                break
        env.flush_trajectory()
        if episode_ok:
            success_count += 1
            print(f"Episode {ep}: SUCCESS")
        else:
            print(f"Episode {ep}: FAIL")

    env.close()
    print(f"\nTotal: {success_count}/{args.num_episodes} success")


if __name__ == "__main__":
    main()
