import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("SequentialPickCube-v1", max_episode_steps=2000)
class SequentialPickCubeEnv(BaseEnv):
    """
    长程任务：依次把红、绿、蓝三个 cube 放到三个目标位置。
    每个 episode 约 900-1200 帧，9 个语义阶段。
    """
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    # 固定位置，方便硬编码专家策略
    CUBE_POSITIONS = [
        [0.15, -0.15],  # red
        [0.15, 0.00],   # green
        [0.15, 0.15],   # blue
    ]
    GOAL_POSITIONS = [
        [-0.10, -0.15],  # goal 1
        [-0.10, 0.00],   # goal 2
        [-0.10, 0.15],   # goal 3
    ]
    COLORS = [
        [1, 0, 0, 1],  # red
        [0, 1, 0, 1],  # green
        [0, 0, 1, 1],  # blue
    ]

    cube_half_size = 0.015
    pedestal_height = 0.045
    pedestal_radius = 0.03
    goal_thresh = 0.020

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # 第三人称相机，224x224，和 LIBERO 一致
        pose = sapien_utils.look_at(eye=[0.5, 0.0, 0.5], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 224, 224, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.0, 0.5], target=[0.0, 0.0, 0.0])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        # 机器人初始位置，和 PickCube 一致
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 创建 3 个 cube 和垫高底座
        self.cubes = []
        self.pedestals = []
        for i in range(3):
            pedestal = actors.build_cylinder(
                self.scene,
                radius=self.pedestal_radius,
                half_length=self.pedestal_height / 2,
                color=[0.8, 0.8, 0.8, 1],
                name=f"pedestal_{i}",
                body_type="static",
                initial_pose=sapien.Pose(p=[0, 0, self.pedestal_height / 2]),
            )
            self.pedestals.append(pedestal)

            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=self.COLORS[i],
                name=f"cube_{i}",
                initial_pose=sapien.Pose(p=[0, 0, self.pedestal_height + self.cube_half_size]),
            )
            self.cubes.append(cube)

        # 创建 3 个 goal site（可视化用）
        self.goal_sites = []
        for i in range(3):
            goal = actors.build_sphere(
                self.scene,
                radius=self.goal_thresh,
                color=self.COLORS[i],
                name=f"goal_site_{i}",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self.goal_sites.append(goal)
            self._hidden_objects.append(goal)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 初始化 3 个 pedestal 和 cube 位置
            for i in range(3):
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, 0] = self.CUBE_POSITIONS[i][0]
                xyz[:, 1] = self.CUBE_POSITIONS[i][1]
                xyz[:, 2] = self.pedestal_height / 2
                self.pedestals[i].set_pose(Pose.create_from_pq(xyz))

                xyz[:, 2] = self.pedestal_height + self.cube_half_size
                self.cubes[i].set_pose(Pose.create_from_pq(xyz))

            # 初始化 3 个 goal 位置
            for i in range(3):
                goal_xyz = torch.zeros((b, 3), device=self.device)
                goal_xyz[:, 0] = self.GOAL_POSITIONS[i][0]
                goal_xyz[:, 1] = self.GOAL_POSITIONS[i][1]
                goal_xyz[:, 2] = self.pedestal_height + self.cube_half_size
                self.goal_sites[i].set_pose(Pose.create_from_pq(goal_xyz))

            # 当前目标索引，从 0 开始
            self.current_target = torch.zeros(b, dtype=torch.long, device=self.device)

    def _get_obs_extra(self, info: dict):
        obs = dict(
            current_target=self.current_target,
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        return obs

    def evaluate(self):
        batch_size = len(self.current_target)
        is_obj_placed = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        is_grasped = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # 只检查当前目标 cube
        for i in range(3):
            mask = self.current_target == i
            if mask.any():
                placed = (
                    torch.linalg.norm(
                        self.goal_sites[i].pose.p - self.cubes[i].pose.p, axis=1
                    )
                    <= self.goal_thresh
                )
                grasped = self.agent.is_grasping(self.cubes[i])
                is_obj_placed = is_obj_placed | (mask & placed)
                is_grasped = is_grasped | (mask & grasped)

        is_robot_static = self.agent.is_static(0.2)

        # 当前目标完成且机器人静止：切换到下一个目标
        target_complete = is_obj_placed & is_robot_static
        self.current_target = torch.where(
            target_complete,
            torch.clamp(self.current_target + 1, max=2),
            self.current_target,
        )

        # 3 个目标都完成才算成功
        success = (self.current_target == 2) & target_complete

        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "current_target": self.current_target,
        }

    def compute_dense_reward(self, obs, action, info):
        batch_size = len(self.current_target)
        reward = torch.zeros(batch_size, device=self.device)

        for i in range(3):
            mask = self.current_target == i
            if not mask.any():
                continue

            tcp_to_obj = torch.linalg.norm(
                self.cubes[i].pose.p - self.agent.tcp_pose.p, axis=1
            )
            reaching = 1 - torch.tanh(5 * tcp_to_obj)

            obj_to_goal = torch.linalg.norm(
                self.goal_sites[i].pose.p - self.cubes[i].pose.p, axis=1
            )
            placing = 1 - torch.tanh(5 * obj_to_goal)

            grasped = self.agent.is_grasping(self.cubes[i]).float()

            r = reaching + grasped + placing * grasped
            reward = reward + mask.float() * r

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5

    def get_language_instruction(self):
        return "pick the red, green, and blue cubes and place them at the three target locations"
