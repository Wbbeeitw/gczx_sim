"""Privileged-state semantic traces for LIBERO rollout collection.

The collector records compact task-relevant simulator state separately from
policy observations.  The trace supports automatic, auditable phase labels
without exposing privileged data to the policy.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NUM_TASK1_PHASES = 4
FAILED_TERMINAL_PHASE_PROGRESS_CAP = 0.15
TASK1_SUCCESS_PHASE = 3
NUM_TASK2_PHASES = 4
TASK2_SUCCESS_PHASE = 3
NUM_TASK3_PHASES = 4
TASK3_SUCCESS_PHASE = 3
NUM_TASK4_PHASES = 4
TASK4_SUCCESS_PHASE = 3
NUM_TASK5_PHASES = 4
TASK5_SUCCESS_PHASE = 3
NUM_TASK6_PHASES = 4
TASK6_SUCCESS_PHASE = 3


@dataclass(frozen=True)
class Task1SemanticTraceConfig:
    """Task1 simulator-state extraction settings."""

    object_a_alias: str = "cream_cheese"
    object_b_alias: str = "butter"
    basket_alias: str = "basket"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    basket_near_threshold: float = 0.18
    stable_frames: int = 5


@dataclass(frozen=True)
class Task1SemanticBodies:
    """Resolved MuJoCo body names used by a task1 trace."""

    object_a: str
    object_b: str
    basket: str


@dataclass(frozen=True)
class Task2SemanticTraceConfig:
    """Task2 simulator-state extraction settings."""

    moka_pot_alias: str = "moka_pot"
    stove_button_alias: str = "flat_stove_1_button"
    frypan_alias: str = "chefmate_8_frypan"
    moka_pot_state_name: str = "moka_pot_1"
    stove_state_name: str = "flat_stove_1"
    cook_region_state_name: str = "flat_stove_1_cook_region"
    stove_button_joint_name: str = "flat_stove_1_button"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    button_motion_threshold: float = 0.001
    stable_frames: int = 3


@dataclass(frozen=True)
class Task2SemanticBodies:
    """Resolved MuJoCo body names used by a task2 trace."""

    moka_pot: str
    stove_button: str
    frypan: str


@dataclass(frozen=True)
class Task3SemanticTraceConfig:
    """Task3 simulator-state extraction settings."""

    black_bowl_alias: str = "akita_black_bowl"
    bottom_drawer_alias: str = "white_cabinet_1_cabinet_bottom"
    wine_bottle_alias: str = "wine_bottle"
    wine_rack_alias: str = "wine_rack"
    black_bowl_state_name: str = "akita_black_bowl_1"
    bottom_drawer_state_name: str = "white_cabinet_1_bottom_region"
    bottom_drawer_joint_name: str = "white_cabinet_1_bottom_level"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    drawer_motion_threshold: float = 0.001
    stable_frames: int = 3


@dataclass(frozen=True)
class Task3SemanticBodies:
    """Resolved MuJoCo body names used by a task3 trace."""

    black_bowl: str
    bottom_drawer: str
    wine_bottle: str
    wine_rack: str


@dataclass(frozen=True)
class Task4SemanticTraceConfig:
    """Task4 simulator-state extraction settings."""

    porcelain_mug_alias: str = "porcelain_mug"
    white_yellow_mug_alias: str = "white_yellow_mug"
    left_plate_alias: str = "plate_1"
    right_plate_alias: str = "plate_2"
    red_coffee_mug_alias: str = "red_coffee_mug"
    porcelain_mug_state_name: str = "porcelain_mug_1"
    white_yellow_mug_state_name: str = "white_yellow_mug_1"
    left_plate_state_name: str = "plate_1"
    right_plate_state_name: str = "plate_2"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    stable_frames: int = 3


@dataclass(frozen=True)
class Task4SemanticBodies:
    """Resolved MuJoCo body names used by a task4 trace."""

    porcelain_mug: str
    white_yellow_mug: str
    left_plate: str
    right_plate: str
    red_coffee_mug: str


@dataclass(frozen=True)
class Task5SemanticTraceConfig:
    """Task5 simulator-state extraction settings."""

    black_book_alias: str = "black_book_1_main"
    caddy_alias: str = "desk_caddy"
    white_yellow_mug_alias: str = "white_yellow_mug"
    black_book_state_name: str = "black_book_1"
    back_compartment_state_name: str = "desk_caddy_1_back_contain_region"
    back_compartment_site_name: str = "desk_caddy_1_back_contain_region"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    final_insertion_distance_threshold: float = 0.10
    stable_frames: int = 3


@dataclass(frozen=True)
class Task5SemanticBodies:
    """Resolved MuJoCo body names used by a task5 trace."""

    black_book: str
    caddy: str
    white_yellow_mug: str


@dataclass(frozen=True)
class Task6SemanticTraceConfig:
    """Task6 simulator-state extraction settings."""

    porcelain_mug_alias: str = "porcelain_mug"
    chocolate_pudding_alias: str = "chocolate_pudding"
    plate_alias: str = "plate_1"
    red_coffee_mug_alias: str = "red_coffee_mug"
    porcelain_mug_state_name: str = "porcelain_mug_1"
    chocolate_pudding_state_name: str = "chocolate_pudding_1"
    plate_state_name: str = "plate_1"
    plate_left_region_state_name: str = "living_room_table_plate_left_region"
    plate_right_region_state_name: str = "living_room_table_plate_right_region"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    stable_frames: int = 3


@dataclass(frozen=True)
class Task6SemanticBodies:
    """Resolved MuJoCo body names used by a task6 trace."""

    porcelain_mug: str
    chocolate_pudding: str
    plate: str
    red_coffee_mug: str


@dataclass(frozen=True)
class Task7SemanticTraceConfig:
    """Task7 simulator-state extraction settings."""

    alphabet_soup_alias: str = "alphabet_soup_1_main"
    cream_cheese_alias: str = "cream_cheese_1_main"
    basket_alias: str = "basket_1_main"
    tomato_sauce_alias: str = "tomato_sauce_1_main"
    ketchup_alias: str = "ketchup_1_main"
    alphabet_soup_state_name: str = "alphabet_soup_1"
    cream_cheese_state_name: str = "cream_cheese_1"
    basket_contain_region_state_name: str = "basket_1_contain_region"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    basket_near_threshold: float = 0.18
    stable_frames: int = 5


@dataclass(frozen=True)
class Task8SemanticTraceConfig:
    """Task8 simulator-state extraction settings."""

    moka_pot_1_alias: str = "moka_pot_1_main"
    moka_pot_2_alias: str = "moka_pot_2_main"
    stove_alias: str = "flat_stove_1_main"
    moka_pot_1_state_name: str = "moka_pot_1"
    moka_pot_2_state_name: str = "moka_pot_2"
    cook_region_state_name: str = "flat_stove_1_cook_region"
    stove_state_name: str = "flat_stove_1"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    stable_frames: int = 3


@dataclass(frozen=True)
class Task9SemanticTraceConfig:
    """Task9 simulator-state extraction settings."""

    white_yellow_mug_alias: str = "white_yellow_mug_1_main"
    microwave_alias: str = "microwave_1_main"
    porcelain_mug_alias: str = "porcelain_mug_1_main"
    white_yellow_mug_state_name: str = "white_yellow_mug_1"
    heating_region_state_name: str = "microwave_1_heating_region"
    microwave_state_name: str = "microwave_1"
    microwave_joint_name: str = "microwave_1_microjoint"
    gripper_width_threshold: float = 0.05
    controlled_motion_threshold: float = 0.001
    microwave_closed_qpos: float = 0.0
    stable_frames: int = 3


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _model_name(model: Any, kind: str, index: int) -> str | None:
    legacy_getter = getattr(model, f"{kind}_id2name", None)
    if callable(legacy_getter):
        result = legacy_getter(index)
        return str(result) if result else None

    accessor = getattr(model, kind, None)
    if callable(accessor):
        result = accessor(index)
        name = getattr(result, "name", None)
        return str(name) if name else None
    return None


def _body_names(model: Any) -> list[str]:
    count = int(getattr(model, "nbody", 0))
    return [
        name
        for index in range(count)
        if (name := _model_name(model, "body", index)) is not None
    ]


def _resolve_body_name(model: Any, alias: str) -> str:
    normalized_alias = _normalize_name(alias)
    matches = [
        name
        for name in _body_names(model)
        if normalized_alias in _normalize_name(name)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Could not resolve LIBERO semantic body alias {alias!r}. "
            f"Available bodies: {_body_names(model)}"
        )
    raise ValueError(
        f"Semantic body alias {alias!r} is ambiguous: {matches}. "
        "Use a more specific alias."
    )


def _body_id(model: Any, name: str) -> int:
    legacy_getter = getattr(model, "body_name2id", None)
    if callable(legacy_getter):
        return int(legacy_getter(name))
    accessor = getattr(model, "body", None)
    if callable(accessor):
        return int(accessor(name).id)
    raise AttributeError("MuJoCo model does not expose a body-name lookup API.")


def _body_pose(sim: Any, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    model = sim.model
    data = sim.data
    body_index = _body_id(model, body_name)
    if hasattr(data, "body_xpos"):
        position = np.asarray(data.body_xpos[body_index], dtype=np.float32)
        quaternion = np.asarray(data.body_xquat[body_index], dtype=np.float32)
        return position, quaternion
    body = data.body(body_name)
    return (
        np.asarray(body.xpos, dtype=np.float32),
        np.asarray(body.xquat, dtype=np.float32),
    )


def _site_position(sim: Any, site_name: str) -> np.ndarray:
    """Read the world position of a named MuJoCo site."""
    model = sim.model
    legacy_getter = getattr(model, "site_name2id", None)
    if callable(legacy_getter):
        site_id = int(legacy_getter(site_name))
    else:
        accessor = getattr(model, "site", None)
        if not callable(accessor):
            raise AttributeError("MuJoCo model does not expose a site-name lookup API.")
        site_id = int(accessor(site_name).id)
    data = sim.data
    if hasattr(data, "site_xpos"):
        return np.asarray(data.site_xpos[site_id], dtype=np.float32)
    return np.asarray(data.site(site_name).xpos, dtype=np.float32)


def _geom_ids_for_body(model: Any, body_name: str, alias: str) -> set[int]:
    body_index = _body_id(model, body_name)
    geom_body_ids = np.asarray(getattr(model, "geom_bodyid", []), dtype=np.int64)
    direct_body_geoms = set(np.flatnonzero(geom_body_ids == body_index).tolist())
    normalized_alias = _normalize_name(alias)
    alias_geoms = {
        index
        for index in range(int(getattr(model, "ngeom", 0)))
        if (name := _model_name(model, "geom", index)) is not None
        and normalized_alias in _normalize_name(name)
    }
    return direct_body_geoms | alias_geoms


def _gripper_geom_ids(model: Any) -> set[int]:
    count = int(getattr(model, "ngeom", 0))
    keywords = ("finger", "gripper", "panda", "robot0")
    return {
        index
        for index in range(count)
        if (name := _model_name(model, "geom", index)) is not None
        and any(keyword in _normalize_name(name) for keyword in keywords)
    }


def _has_contact(sim: Any, left_geom_ids: set[int], right_geom_ids: set[int]) -> bool:
    if not left_geom_ids or not right_geom_ids:
        return False
    data = sim.data
    for contact_index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[contact_index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        if (first in left_geom_ids and second in right_geom_ids) or (
            second in left_geom_ids and first in right_geom_ids
        ):
            return True
    return False


def _joint_qpos(sim: Any, joint_name: str) -> float:
    """Read the scalar qpos of a named MuJoCo joint."""
    model = sim.model
    legacy_getter = getattr(model, "joint_name2id", None)
    if callable(legacy_getter):
        joint_id = int(legacy_getter(joint_name))
    else:
        accessor = getattr(model, "joint", None)
        if not callable(accessor):
            raise AttributeError("MuJoCo model does not expose a joint-name lookup API.")
        joint_id = int(accessor(joint_name).id)
    qpos_address = int(np.asarray(model.jnt_qposadr)[joint_id])
    return float(sim.data.qpos[qpos_address])


def _check_success(env: Any) -> bool:
    checker = getattr(env, "check_success", None)
    if callable(checker):
        return bool(checker())
    checker = getattr(env, "_check_success", None)
    if callable(checker):
        return bool(checker())
    return False


class Task1SemanticTraceRecorder:
    """Extract compact task1 state from a live LIBERO simulator."""

    def __init__(self, env: Any, config: Task1SemanticTraceConfig | None = None):
        self.config = config or Task1SemanticTraceConfig()
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._sim = sim
        model = sim.model
        self.bodies = Task1SemanticBodies(
            object_a=_resolve_body_name(model, self.config.object_a_alias),
            object_b=_resolve_body_name(model, self.config.object_b_alias),
            basket=_resolve_body_name(model, self.config.basket_alias),
        )
        self._object_a_geoms = _geom_ids_for_body(
            model, self.bodies.object_a, self.config.object_a_alias
        )
        self._object_b_geoms = _geom_ids_for_body(
            model, self.bodies.object_b, self.config.object_b_alias
        )
        self._basket_geoms = _geom_ids_for_body(
            model, self.bodies.basket, self.config.basket_alias
        )
        self._gripper_geoms = _gripper_geom_ids(model)
        self._previous_positions: dict[str, np.ndarray] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the resolved trace schema for reproducibility."""
        return {
            "version": "task1_semantic_trace_v1",
            "config": asdict(self.config),
            "bodies": asdict(self.bodies),
        }

    def capture(
        self,
        env: Any,
        episode_index: int,
        frame_index: int,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture one simulator-aligned task1 semantic trace record."""
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        object_a_pos, object_a_quat = _body_pose(sim, self.bodies.object_a)
        object_b_pos, object_b_quat = _body_pose(sim, self.bodies.object_b)
        basket_pos, basket_quat = _body_pose(sim, self.bodies.basket)
        object_a_motion = self._motion("object_a", object_a_pos)
        object_b_motion = self._motion("object_b", object_b_pos)

        gripper_width = self._gripper_width(observation, env)
        gripper_closed = gripper_width <= self.config.gripper_width_threshold
        object_a_gripper_contact = _has_contact(
            sim, self._object_a_geoms, self._gripper_geoms
        )
        object_b_gripper_contact = _has_contact(
            sim, self._object_b_geoms, self._gripper_geoms
        )
        object_a_basket_contact = _has_contact(
            sim, self._object_a_geoms, self._basket_geoms
        )
        object_b_basket_contact = _has_contact(
            sim, self._object_b_geoms, self._basket_geoms
        )
        object_a_distance = float(np.linalg.norm(object_a_pos - basket_pos))
        object_b_distance = float(np.linalg.norm(object_b_pos - basket_pos))
        object_a_near = object_a_distance <= self.config.basket_near_threshold
        object_b_near = object_b_distance <= self.config.basket_near_threshold
        object_a_controlled = (
            gripper_closed
            and object_a_gripper_contact
            and object_a_motion >= self.config.controlled_motion_threshold
        )
        object_b_controlled = (
            gripper_closed
            and object_b_gripper_contact
            and object_b_motion >= self.config.controlled_motion_threshold
        )

        return {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "env_success": _check_success(env),
            "gripper_width": gripper_width,
            "gripper_closed": gripper_closed,
            "object_a_x": float(object_a_pos[0]),
            "object_a_y": float(object_a_pos[1]),
            "object_a_z": float(object_a_pos[2]),
            "object_a_qw": float(object_a_quat[0]),
            "object_a_qx": float(object_a_quat[1]),
            "object_a_qy": float(object_a_quat[2]),
            "object_a_qz": float(object_a_quat[3]),
            "object_a_motion": object_a_motion,
            "object_a_gripper_contact": object_a_gripper_contact,
            "object_a_controlled": object_a_controlled,
            "object_a_basket_contact": object_a_basket_contact,
            "object_a_near_basket": object_a_near,
            "object_b_x": float(object_b_pos[0]),
            "object_b_y": float(object_b_pos[1]),
            "object_b_z": float(object_b_pos[2]),
            "object_b_qw": float(object_b_quat[0]),
            "object_b_qx": float(object_b_quat[1]),
            "object_b_qy": float(object_b_quat[2]),
            "object_b_qz": float(object_b_quat[3]),
            "object_b_motion": object_b_motion,
            "object_b_gripper_contact": object_b_gripper_contact,
            "object_b_controlled": object_b_controlled,
            "object_b_basket_contact": object_b_basket_contact,
            "object_b_near_basket": object_b_near,
            "basket_x": float(basket_pos[0]),
            "basket_y": float(basket_pos[1]),
            "basket_z": float(basket_pos[2]),
            "basket_qw": float(basket_quat[0]),
            "basket_qx": float(basket_quat[1]),
            "basket_qy": float(basket_quat[2]),
            "basket_qz": float(basket_quat[3]),
        }

    def _motion(self, key: str, position: np.ndarray) -> float:
        previous = self._previous_positions.get(key)
        self._previous_positions[key] = position.copy()
        if previous is None:
            return 0.0
        return float(np.linalg.norm(position - previous))

    @staticmethod
    def _gripper_width(observation: dict[str, Any] | None, env: Any) -> float:
        if observation is not None:
            values = observation.get("robot0_gripper_qpos")
            if values is not None:
                return float(np.abs(np.asarray(values, dtype=np.float32)).sum())
        get_observations = getattr(env, "_get_observations", None)
        if callable(get_observations):
            values = get_observations().get("robot0_gripper_qpos")
            if values is not None:
                return float(np.abs(np.asarray(values, dtype=np.float32)).sum())
        return float("inf")


class Task7SemanticTraceRecorder(Task1SemanticTraceRecorder):
    """Extract task7 state while reusing task1's two-object phase contract."""

    def __init__(self, env: Any, config: Task7SemanticTraceConfig | None = None):
        self.task7_config = config or Task7SemanticTraceConfig()
        super().__init__(
            env,
            Task1SemanticTraceConfig(
                object_a_alias=self.task7_config.alphabet_soup_alias,
                object_b_alias=self.task7_config.cream_cheese_alias,
                basket_alias=self.task7_config.basket_alias,
                gripper_width_threshold=self.task7_config.gripper_width_threshold,
                controlled_motion_threshold=self.task7_config.controlled_motion_threshold,
                basket_near_threshold=self.task7_config.basket_near_threshold,
                stable_frames=self.task7_config.stable_frames,
            ),
        )
        model = env.sim.model
        self._tomato_sauce_geoms = _geom_ids_for_body(
            model,
            _resolve_body_name(model, self.task7_config.tomato_sauce_alias),
            self.task7_config.tomato_sauce_alias,
        )
        self._ketchup_geoms = _geom_ids_for_body(
            model,
            _resolve_body_name(model, self.task7_config.ketchup_alias),
            self.task7_config.ketchup_alias,
        )
        self._state_objects: Any | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "version": "task7_semantic_trace_v1",
            "config": asdict(self.task7_config),
            "bodies": asdict(self.bodies),
            "state_names": {
                "alphabet_soup": self.task7_config.alphabet_soup_state_name,
                "cream_cheese": self.task7_config.cream_cheese_state_name,
                "basket_contain_region": self.task7_config.basket_contain_region_state_name,
            },
        }

    def capture(self, env: Any, episode_index: int, frame_index: int, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        record = super().capture(env, episode_index, frame_index, observation)
        state_objects = self._state_objects_for_env(env)
        basket_region = state_objects[self.task7_config.basket_contain_region_state_name]
        alphabet_soup = state_objects[self.task7_config.alphabet_soup_state_name]
        cream_cheese = state_objects[self.task7_config.cream_cheese_state_name]
        record["object_a_basket_contact"] = bool(basket_region.check_contain(alphabet_soup))
        record["object_b_basket_contact"] = bool(basket_region.check_contain(cream_cheese))
        record["tomato_sauce_gripper_contact"] = _has_contact(env.sim, self._tomato_sauce_geoms, self._gripper_geoms)
        record["ketchup_gripper_contact"] = _has_contact(env.sim, self._ketchup_geoms, self._gripper_geoms)
        return record

    def _state_objects_for_env(self, env: Any) -> Any:
        if self._state_objects is None:
            states = getattr(getattr(env, "env", env), "object_states_dict", None)
            required = {self.task7_config.alphabet_soup_state_name, self.task7_config.cream_cheese_state_name, self.task7_config.basket_contain_region_state_name}
            if states is None or required - set(states):
                raise ValueError(f"LIBERO task7 semantic states are unavailable: {sorted(required - set(states or {}))}")
            self._state_objects = states
        return self._state_objects


class Task8SemanticTraceRecorder(Task1SemanticTraceRecorder):
    """Extract two-moka-pot stove placement state for Task8."""

    def __init__(self, env: Any, config: Task8SemanticTraceConfig | None = None):
        self.task8_config = config or Task8SemanticTraceConfig()
        super().__init__(env, Task1SemanticTraceConfig(
            object_a_alias=self.task8_config.moka_pot_1_alias,
            object_b_alias=self.task8_config.moka_pot_2_alias,
            basket_alias=self.task8_config.stove_alias,
            gripper_width_threshold=self.task8_config.gripper_width_threshold,
            controlled_motion_threshold=self.task8_config.controlled_motion_threshold,
            stable_frames=self.task8_config.stable_frames,
        ))
        self._state_objects: Any | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {"version": "task8_semantic_trace_v1", "config": asdict(self.task8_config), "bodies": asdict(self.bodies)}

    def capture(self, env: Any, episode_index: int, frame_index: int, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        record = super().capture(env, episode_index, frame_index, observation)
        states = self._states(env)
        region = states[self.task8_config.cook_region_state_name]
        record["object_a_basket_contact"] = bool(region.check_ontop(states[self.task8_config.moka_pot_1_state_name]))
        record["object_b_basket_contact"] = bool(region.check_ontop(states[self.task8_config.moka_pot_2_state_name]))
        record["stove_turn_on"] = bool(states[self.task8_config.stove_state_name].turn_on())
        return record

    def _states(self, env: Any) -> Any:
        if self._state_objects is None:
            states = getattr(getattr(env, "env", env), "object_states_dict", None)
            required = {self.task8_config.moka_pot_1_state_name, self.task8_config.moka_pot_2_state_name, self.task8_config.cook_region_state_name, self.task8_config.stove_state_name}
            if states is None or required - set(states):
                raise ValueError(f"LIBERO task8 semantic states unavailable: {sorted(required - set(states or {}))}")
            self._state_objects = states
        return self._state_objects


class Task9SemanticTraceRecorder:
    """Extract privileged mug-in-microwave and door-close state for Task9."""

    def __init__(self, env: Any, config: Task9SemanticTraceConfig | None = None):
        self.config = config or Task9SemanticTraceConfig()
        model = env.sim.model
        self._mug_body = _resolve_body_name(model, self.config.white_yellow_mug_alias)
        self._microwave_body = _resolve_body_name(model, self.config.microwave_alias)
        self._mug_geoms = _geom_ids_for_body(model, self._mug_body, self.config.white_yellow_mug_alias)
        self._porcelain_geoms = _geom_ids_for_body(model, _resolve_body_name(model, self.config.porcelain_mug_alias), self.config.porcelain_mug_alias)
        self._microwave_geoms = _geom_ids_for_body(model, self._microwave_body, self.config.microwave_alias)
        self._gripper_geoms = _gripper_geom_ids(model)
        self._states: Any | None = None
        self._previous_position: np.ndarray | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the resolved Task9 trace schema for reproducibility."""
        return {
            "version": "task9_semantic_trace_v1",
            "config": asdict(self.config),
            "bodies": {
                "white_yellow_mug": self._mug_body,
                "microwave": self._microwave_body,
            },
            "state_names": {
                "white_yellow_mug": self.config.white_yellow_mug_state_name,
                "heating_region": self.config.heating_region_state_name,
                "microwave": self.config.microwave_state_name,
            },
        }

    def capture(self, env: Any, episode_index: int, frame_index: int, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        position, quaternion = _body_pose(env.sim, self._mug_body)
        microwave_position, _ = _body_pose(env.sim, self._microwave_body)
        motion = 0.0 if self._previous_position is None else float(np.linalg.norm(position - self._previous_position))
        self._previous_position = position.copy()
        width = Task1SemanticTraceRecorder._gripper_width(observation, env)
        mug_contact = _has_contact(env.sim, self._mug_geoms, self._gripper_geoms)
        states = self._state_objects(env)
        mug = states[self.config.white_yellow_mug_state_name]
        microwave = states[self.config.microwave_state_name]
        heating = states[self.config.heating_region_state_name]
        return {"episode_index": int(episode_index), "frame_index": int(frame_index), "env_success": _check_success(env), "mug_motion": motion, "mug_controlled": bool(width <= self.config.gripper_width_threshold and mug_contact and motion >= self.config.controlled_motion_threshold), "mug_in_heating_region": bool(heating.check_contain(mug)), "microwave_is_open": bool(microwave.is_open()), "microwave_is_close": bool(microwave.is_close()), "microwave_joint_qpos": _joint_qpos(env.sim, self.config.microwave_joint_name), "mug_microwave_distance": float(np.linalg.norm(position - microwave_position)), "mug_microwave_contact": _has_contact(env.sim, self._mug_geoms, self._microwave_geoms), "porcelain_mug_gripper_contact": _has_contact(env.sim, self._porcelain_geoms, self._gripper_geoms), "mug_x": float(position[0]), "mug_y": float(position[1]), "mug_z": float(position[2]), "mug_qw": float(quaternion[0]), "mug_qx": float(quaternion[1]), "mug_qy": float(quaternion[2]), "mug_qz": float(quaternion[3])}

    def _state_objects(self, env: Any) -> Any:
        if self._states is None:
            states = getattr(getattr(env, "env", env), "object_states_dict", None)
            required = {self.config.white_yellow_mug_state_name, self.config.heating_region_state_name, self.config.microwave_state_name}
            if states is None or required - set(states): raise ValueError(f"LIBERO task9 semantic states unavailable: {sorted(required - set(states or {}))}")
            self._states = states
        return self._states


class Task2SemanticTraceRecorder:
    """Extract task2 privileged state from a live LIBERO simulator."""

    def __init__(self, env: Any, config: Task2SemanticTraceConfig | None = None):
        self.config = config or Task2SemanticTraceConfig()
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._sim = sim
        model = sim.model
        self.bodies = Task2SemanticBodies(
            moka_pot=_resolve_body_name(model, self.config.moka_pot_alias),
            stove_button=_resolve_body_name(model, self.config.stove_button_alias),
            frypan=_resolve_body_name(model, self.config.frypan_alias),
        )
        self._moka_pot_geoms = _geom_ids_for_body(
            model, self.bodies.moka_pot, self.config.moka_pot_alias
        )
        self._stove_button_geoms = _geom_ids_for_body(
            model, self.bodies.stove_button, self.config.stove_button_alias
        )
        self._frypan_geoms = _geom_ids_for_body(
            model, self.bodies.frypan, self.config.frypan_alias
        )
        self._gripper_geoms = _gripper_geom_ids(model)
        self._state_objects: Any | None = None
        self._previous_moka_pot_position: np.ndarray | None = None
        self._previous_button_qpos: float | None = None
        self._current_episode_index: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the resolved task2 trace schema for reproducibility."""
        return {
            "version": "task2_semantic_trace_v1",
            "config": asdict(self.config),
            "bodies": asdict(self.bodies),
            "state_names": {
                "moka_pot": self.config.moka_pot_state_name,
                "stove": self.config.stove_state_name,
                "cook_region": self.config.cook_region_state_name,
            },
        }

    def capture(
        self,
        env: Any,
        episode_index: int,
        frame_index: int,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture one simulator-aligned task2 semantic trace record."""
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._start_episode(int(episode_index))
        moka_pot_position, moka_pot_quaternion = _body_pose(
            sim, self.bodies.moka_pot
        )
        moka_pot_motion = self._moka_pot_motion(moka_pot_position)
        button_qpos = _joint_qpos(sim, self.config.stove_button_joint_name)
        button_motion = self._button_motion(button_qpos)
        gripper_width = Task1SemanticTraceRecorder._gripper_width(observation, env)
        gripper_closed = gripper_width <= self.config.gripper_width_threshold
        moka_pot_gripper_contact = _has_contact(
            sim, self._moka_pot_geoms, self._gripper_geoms
        )
        stove_button_gripper_contact = _has_contact(
            sim, self._stove_button_geoms, self._gripper_geoms
        )
        frypan_gripper_contact = _has_contact(
            sim, self._frypan_geoms, self._gripper_geoms
        )
        moka_pot_controlled = (
            gripper_closed
            and moka_pot_gripper_contact
            and moka_pot_motion >= self.config.controlled_motion_threshold
        )
        stove_button_interacted = (
            button_motion >= self.config.button_motion_threshold
        )
        state_objects = self._state_objects_for_env(env)
        moka_pot_state = state_objects[self.config.moka_pot_state_name]
        stove_state = state_objects[self.config.stove_state_name]
        cook_region_state = state_objects[self.config.cook_region_state_name]
        stove_turn_on = bool(stove_state.turn_on())
        moka_pot_on_cook_region = bool(cook_region_state.check_ontop(moka_pot_state))

        return {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "env_success": _check_success(env),
            "gripper_width": gripper_width,
            "gripper_closed": gripper_closed,
            "moka_pot_x": float(moka_pot_position[0]),
            "moka_pot_y": float(moka_pot_position[1]),
            "moka_pot_z": float(moka_pot_position[2]),
            "moka_pot_qw": float(moka_pot_quaternion[0]),
            "moka_pot_qx": float(moka_pot_quaternion[1]),
            "moka_pot_qy": float(moka_pot_quaternion[2]),
            "moka_pot_qz": float(moka_pot_quaternion[3]),
            "moka_pot_motion": moka_pot_motion,
            "moka_pot_gripper_contact": moka_pot_gripper_contact,
            "moka_pot_controlled": moka_pot_controlled,
            "stove_button_qpos": button_qpos,
            "stove_button_motion": button_motion,
            "stove_button_gripper_contact": stove_button_gripper_contact,
            "stove_button_interacted": stove_button_interacted,
            "stove_turn_on": stove_turn_on,
            "moka_pot_on_cook_region": moka_pot_on_cook_region,
            "frypan_gripper_contact": frypan_gripper_contact,
        }

    def _moka_pot_motion(self, position: np.ndarray) -> float:
        previous = self._previous_moka_pot_position
        self._previous_moka_pot_position = position.copy()
        if previous is None:
            return 0.0
        return float(np.linalg.norm(position - previous))

    def _start_episode(self, episode_index: int) -> None:
        if self._current_episode_index == episode_index:
            return
        self._current_episode_index = episode_index
        self._previous_moka_pot_position = None
        self._previous_button_qpos = None

    def _button_motion(self, qpos: float) -> float:
        previous = self._previous_button_qpos
        self._previous_button_qpos = qpos
        if previous is None:
            return 0.0
        return abs(qpos - previous)

    def _state_objects_for_env(self, env: Any) -> Any:
        if self._state_objects is not None:
            return self._state_objects
        base_env = getattr(env, "env", env)
        state_objects = getattr(base_env, "object_states_dict", None)
        if state_objects is None:
            raise AttributeError(
                "LIBERO task2 environment does not expose object_states_dict after reset."
            )
        required_state_names = {
            self.config.moka_pot_state_name,
            self.config.stove_state_name,
            self.config.cook_region_state_name,
        }
        missing_state_names = required_state_names - set(state_objects)
        if missing_state_names:
            raise ValueError(
                "LIBERO task2 semantic states are unavailable: "
                f"{sorted(missing_state_names)}"
            )
        self._state_objects = state_objects
        return state_objects


class Task3SemanticTraceRecorder:
    """Extract task3 privileged state from a live LIBERO simulator."""

    def __init__(self, env: Any, config: Task3SemanticTraceConfig | None = None):
        self.config = config or Task3SemanticTraceConfig()
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._sim = sim
        model = sim.model
        self.bodies = Task3SemanticBodies(
            black_bowl=_resolve_body_name(model, self.config.black_bowl_alias),
            bottom_drawer=_resolve_body_name(model, self.config.bottom_drawer_alias),
            wine_bottle=_resolve_body_name(model, self.config.wine_bottle_alias),
            wine_rack=_resolve_body_name(model, self.config.wine_rack_alias),
        )
        self._black_bowl_geoms = _geom_ids_for_body(
            model, self.bodies.black_bowl, self.config.black_bowl_alias
        )
        self._bottom_drawer_geoms = _geom_ids_for_body(
            model, self.bodies.bottom_drawer, self.config.bottom_drawer_alias
        )
        self._wine_bottle_geoms = _geom_ids_for_body(
            model, self.bodies.wine_bottle, self.config.wine_bottle_alias
        )
        self._wine_rack_geoms = _geom_ids_for_body(
            model, self.bodies.wine_rack, self.config.wine_rack_alias
        )
        self._gripper_geoms = _gripper_geom_ids(model)
        self._state_objects: Any | None = None
        self._previous_black_bowl_position: np.ndarray | None = None
        self._previous_drawer_qpos: float | None = None
        self._current_episode_index: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the resolved task3 trace schema for reproducibility."""
        return {
            "version": "task3_semantic_trace_v1",
            "config": asdict(self.config),
            "bodies": asdict(self.bodies),
            "state_names": {
                "black_bowl": self.config.black_bowl_state_name,
                "bottom_drawer": self.config.bottom_drawer_state_name,
            },
        }

    def capture(
        self,
        env: Any,
        episode_index: int,
        frame_index: int,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture one simulator-aligned task3 semantic trace record."""
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._start_episode(int(episode_index))
        black_bowl_position, black_bowl_quaternion = _body_pose(
            sim, self.bodies.black_bowl
        )
        black_bowl_motion = self._black_bowl_motion(black_bowl_position)
        drawer_qpos = _joint_qpos(sim, self.config.bottom_drawer_joint_name)
        drawer_motion = self._drawer_motion(drawer_qpos)
        gripper_width = Task1SemanticTraceRecorder._gripper_width(observation, env)
        gripper_closed = gripper_width <= self.config.gripper_width_threshold
        black_bowl_gripper_contact = _has_contact(
            sim, self._black_bowl_geoms, self._gripper_geoms
        )
        bottom_drawer_gripper_contact = _has_contact(
            sim, self._bottom_drawer_geoms, self._gripper_geoms
        )
        wine_bottle_gripper_contact = _has_contact(
            sim, self._wine_bottle_geoms, self._gripper_geoms
        )
        wine_rack_gripper_contact = _has_contact(
            sim, self._wine_rack_geoms, self._gripper_geoms
        )
        black_bowl_controlled = (
            gripper_closed
            and black_bowl_gripper_contact
            and black_bowl_motion >= self.config.controlled_motion_threshold
        )
        bottom_drawer_interacted = (
            drawer_motion >= self.config.drawer_motion_threshold
        )
        state_objects = self._state_objects_for_env(env)
        black_bowl_state = state_objects[self.config.black_bowl_state_name]
        bottom_drawer_state = state_objects[self.config.bottom_drawer_state_name]
        black_bowl_in_bottom_drawer = bool(
            bottom_drawer_state.check_contain(black_bowl_state)
        )
        bottom_drawer_is_open = bool(bottom_drawer_state.is_open())
        bottom_drawer_is_close = bool(bottom_drawer_state.is_close())

        return {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "env_success": _check_success(env),
            "gripper_width": gripper_width,
            "gripper_closed": gripper_closed,
            "black_bowl_x": float(black_bowl_position[0]),
            "black_bowl_y": float(black_bowl_position[1]),
            "black_bowl_z": float(black_bowl_position[2]),
            "black_bowl_qw": float(black_bowl_quaternion[0]),
            "black_bowl_qx": float(black_bowl_quaternion[1]),
            "black_bowl_qy": float(black_bowl_quaternion[2]),
            "black_bowl_qz": float(black_bowl_quaternion[3]),
            "black_bowl_motion": black_bowl_motion,
            "black_bowl_gripper_contact": black_bowl_gripper_contact,
            "black_bowl_controlled": black_bowl_controlled,
            "bottom_drawer_qpos": drawer_qpos,
            "bottom_drawer_motion": drawer_motion,
            "bottom_drawer_gripper_contact": bottom_drawer_gripper_contact,
            "bottom_drawer_interacted": bottom_drawer_interacted,
            "bottom_drawer_is_open": bottom_drawer_is_open,
            "bottom_drawer_is_close": bottom_drawer_is_close,
            "black_bowl_in_bottom_drawer": black_bowl_in_bottom_drawer,
            "wine_bottle_gripper_contact": wine_bottle_gripper_contact,
            "wine_rack_gripper_contact": wine_rack_gripper_contact,
        }

    def _black_bowl_motion(self, position: np.ndarray) -> float:
        previous = self._previous_black_bowl_position
        self._previous_black_bowl_position = position.copy()
        if previous is None:
            return 0.0
        return float(np.linalg.norm(position - previous))

    def _start_episode(self, episode_index: int) -> None:
        if self._current_episode_index == episode_index:
            return
        self._current_episode_index = episode_index
        self._previous_black_bowl_position = None
        self._previous_drawer_qpos = None

    def _drawer_motion(self, qpos: float) -> float:
        previous = self._previous_drawer_qpos
        self._previous_drawer_qpos = qpos
        if previous is None:
            return 0.0
        return abs(qpos - previous)

    def _state_objects_for_env(self, env: Any) -> Any:
        if self._state_objects is not None:
            return self._state_objects
        base_env = getattr(env, "env", env)
        state_objects = getattr(base_env, "object_states_dict", None)
        if state_objects is None:
            raise AttributeError(
                "LIBERO task3 environment does not expose object_states_dict after reset."
            )
        required_state_names = {
            self.config.black_bowl_state_name,
            self.config.bottom_drawer_state_name,
        }
        missing_state_names = required_state_names - set(state_objects)
        if missing_state_names:
            raise ValueError(
                "LIBERO task3 semantic states are unavailable: "
                f"{sorted(missing_state_names)}"
            )
        self._state_objects = state_objects
        return state_objects


class Task4SemanticTraceRecorder:
    """Extract task4 privileged state from a live LIBERO simulator."""

    def __init__(self, env: Any, config: Task4SemanticTraceConfig | None = None):
        self.config = config or Task4SemanticTraceConfig()
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._sim = sim
        model = sim.model
        self.bodies = Task4SemanticBodies(
            porcelain_mug=_resolve_body_name(model, self.config.porcelain_mug_alias),
            white_yellow_mug=_resolve_body_name(
                model, self.config.white_yellow_mug_alias
            ),
            left_plate=_resolve_body_name(model, self.config.left_plate_alias),
            right_plate=_resolve_body_name(model, self.config.right_plate_alias),
            red_coffee_mug=_resolve_body_name(
                model, self.config.red_coffee_mug_alias
            ),
        )
        self._porcelain_mug_geoms = _geom_ids_for_body(
            model, self.bodies.porcelain_mug, self.config.porcelain_mug_alias
        )
        self._white_yellow_mug_geoms = _geom_ids_for_body(
            model, self.bodies.white_yellow_mug, self.config.white_yellow_mug_alias
        )
        self._red_coffee_mug_geoms = _geom_ids_for_body(
            model, self.bodies.red_coffee_mug, self.config.red_coffee_mug_alias
        )
        self._gripper_geoms = _gripper_geom_ids(model)
        self._state_objects: Any | None = None
        self._previous_positions: dict[str, np.ndarray] = {}
        self._current_episode_index: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the resolved task4 trace schema for reproducibility."""
        return {
            "version": "task4_semantic_trace_v1",
            "config": asdict(self.config),
            "bodies": asdict(self.bodies),
            "state_names": {
                "porcelain_mug": self.config.porcelain_mug_state_name,
                "white_yellow_mug": self.config.white_yellow_mug_state_name,
                "left_plate": self.config.left_plate_state_name,
                "right_plate": self.config.right_plate_state_name,
            },
        }

    def capture(
        self,
        env: Any,
        episode_index: int,
        frame_index: int,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture one simulator-aligned task4 semantic trace record."""
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._start_episode(int(episode_index))
        porcelain_mug_position, porcelain_mug_quaternion = _body_pose(
            sim, self.bodies.porcelain_mug
        )
        white_yellow_mug_position, white_yellow_mug_quaternion = _body_pose(
            sim, self.bodies.white_yellow_mug
        )
        porcelain_mug_motion = self._motion("porcelain_mug", porcelain_mug_position)
        white_yellow_mug_motion = self._motion(
            "white_yellow_mug", white_yellow_mug_position
        )
        gripper_width = Task1SemanticTraceRecorder._gripper_width(observation, env)
        gripper_closed = gripper_width <= self.config.gripper_width_threshold
        porcelain_mug_gripper_contact = _has_contact(
            sim, self._porcelain_mug_geoms, self._gripper_geoms
        )
        white_yellow_mug_gripper_contact = _has_contact(
            sim, self._white_yellow_mug_geoms, self._gripper_geoms
        )
        red_coffee_mug_gripper_contact = _has_contact(
            sim, self._red_coffee_mug_geoms, self._gripper_geoms
        )
        porcelain_mug_controlled = (
            gripper_closed
            and porcelain_mug_gripper_contact
            and porcelain_mug_motion >= self.config.controlled_motion_threshold
        )
        white_yellow_mug_controlled = (
            gripper_closed
            and white_yellow_mug_gripper_contact
            and white_yellow_mug_motion >= self.config.controlled_motion_threshold
        )
        state_objects = self._state_objects_for_env(env)
        porcelain_mug_state = state_objects[self.config.porcelain_mug_state_name]
        white_yellow_mug_state = state_objects[self.config.white_yellow_mug_state_name]
        left_plate_state = state_objects[self.config.left_plate_state_name]
        right_plate_state = state_objects[self.config.right_plate_state_name]
        porcelain_mug_on_left_plate = bool(
            left_plate_state.check_ontop(porcelain_mug_state)
        )
        white_yellow_mug_on_right_plate = bool(
            right_plate_state.check_ontop(white_yellow_mug_state)
        )
        porcelain_mug_on_right_plate = bool(
            right_plate_state.check_ontop(porcelain_mug_state)
        )
        white_yellow_mug_on_left_plate = bool(
            left_plate_state.check_ontop(white_yellow_mug_state)
        )

        return {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "env_success": _check_success(env),
            "gripper_width": gripper_width,
            "gripper_closed": gripper_closed,
            "porcelain_mug_x": float(porcelain_mug_position[0]),
            "porcelain_mug_y": float(porcelain_mug_position[1]),
            "porcelain_mug_z": float(porcelain_mug_position[2]),
            "porcelain_mug_qw": float(porcelain_mug_quaternion[0]),
            "porcelain_mug_qx": float(porcelain_mug_quaternion[1]),
            "porcelain_mug_qy": float(porcelain_mug_quaternion[2]),
            "porcelain_mug_qz": float(porcelain_mug_quaternion[3]),
            "porcelain_mug_motion": porcelain_mug_motion,
            "porcelain_mug_gripper_contact": porcelain_mug_gripper_contact,
            "porcelain_mug_controlled": porcelain_mug_controlled,
            "white_yellow_mug_x": float(white_yellow_mug_position[0]),
            "white_yellow_mug_y": float(white_yellow_mug_position[1]),
            "white_yellow_mug_z": float(white_yellow_mug_position[2]),
            "white_yellow_mug_qw": float(white_yellow_mug_quaternion[0]),
            "white_yellow_mug_qx": float(white_yellow_mug_quaternion[1]),
            "white_yellow_mug_qy": float(white_yellow_mug_quaternion[2]),
            "white_yellow_mug_qz": float(white_yellow_mug_quaternion[3]),
            "white_yellow_mug_motion": white_yellow_mug_motion,
            "white_yellow_mug_gripper_contact": white_yellow_mug_gripper_contact,
            "white_yellow_mug_controlled": white_yellow_mug_controlled,
            "porcelain_mug_on_left_plate": porcelain_mug_on_left_plate,
            "white_yellow_mug_on_right_plate": white_yellow_mug_on_right_plate,
            "porcelain_mug_on_right_plate": porcelain_mug_on_right_plate,
            "white_yellow_mug_on_left_plate": white_yellow_mug_on_left_plate,
            "red_coffee_mug_gripper_contact": red_coffee_mug_gripper_contact,
        }

    def _motion(self, key: str, position: np.ndarray) -> float:
        previous = self._previous_positions.get(key)
        self._previous_positions[key] = position.copy()
        if previous is None:
            return 0.0
        return float(np.linalg.norm(position - previous))

    def _start_episode(self, episode_index: int) -> None:
        if self._current_episode_index == episode_index:
            return
        self._current_episode_index = episode_index
        self._previous_positions.clear()

    def _state_objects_for_env(self, env: Any) -> Any:
        if self._state_objects is not None:
            return self._state_objects
        base_env = getattr(env, "env", env)
        state_objects = getattr(base_env, "object_states_dict", None)
        if state_objects is None:
            raise AttributeError(
                "LIBERO task4 environment does not expose object_states_dict after reset."
            )
        required_state_names = {
            self.config.porcelain_mug_state_name,
            self.config.white_yellow_mug_state_name,
            self.config.left_plate_state_name,
            self.config.right_plate_state_name,
        }
        missing_state_names = required_state_names - set(state_objects)
        if missing_state_names:
            raise ValueError(
                "LIBERO task4 semantic states are unavailable: "
                f"{sorted(missing_state_names)}"
            )
        self._state_objects = state_objects
        return state_objects


class Task5SemanticTraceRecorder:
    """Extract task5 privileged state from a live LIBERO simulator."""

    def __init__(self, env: Any, config: Task5SemanticTraceConfig | None = None):
        self.config = config or Task5SemanticTraceConfig()
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._sim = sim
        model = sim.model
        self.bodies = Task5SemanticBodies(
            black_book=_resolve_body_name(model, self.config.black_book_alias),
            caddy=_resolve_body_name(model, self.config.caddy_alias),
            white_yellow_mug=_resolve_body_name(
                model, self.config.white_yellow_mug_alias
            ),
        )
        self._black_book_geoms = _geom_ids_for_body(
            model, self.bodies.black_book, self.config.black_book_alias
        )
        self._caddy_geoms = _geom_ids_for_body(
            model, self.bodies.caddy, self.config.caddy_alias
        )
        self._white_yellow_mug_geoms = _geom_ids_for_body(
            model, self.bodies.white_yellow_mug, self.config.white_yellow_mug_alias
        )
        self._gripper_geoms = _gripper_geom_ids(model)
        self._state_objects: Any | None = None
        self._previous_book_position: np.ndarray | None = None
        self._current_episode_index: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the resolved task5 trace schema for reproducibility."""
        return {
            "version": "task5_semantic_trace_v1",
            "config": asdict(self.config),
            "bodies": asdict(self.bodies),
            "state_names": {
                "black_book": self.config.black_book_state_name,
                "back_compartment": self.config.back_compartment_state_name,
            },
        }

    def capture(
        self,
        env: Any,
        episode_index: int,
        frame_index: int,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture one simulator-aligned task5 semantic trace record."""
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._start_episode(int(episode_index))
        book_position, book_quaternion = _body_pose(sim, self.bodies.black_book)
        back_compartment_position = _site_position(
            sim, self.config.back_compartment_site_name
        )
        book_motion = self._book_motion(book_position)
        book_back_distance = float(
            np.linalg.norm(book_position - back_compartment_position)
        )
        gripper_width = Task1SemanticTraceRecorder._gripper_width(observation, env)
        gripper_closed = gripper_width <= self.config.gripper_width_threshold
        book_gripper_contact = _has_contact(
            sim, self._black_book_geoms, self._gripper_geoms
        )
        book_caddy_contact = _has_contact(sim, self._black_book_geoms, self._caddy_geoms)
        white_yellow_mug_gripper_contact = _has_contact(
            sim, self._white_yellow_mug_geoms, self._gripper_geoms
        )
        book_controlled = (
            gripper_closed
            and book_gripper_contact
            and book_motion >= self.config.controlled_motion_threshold
        )
        book_final_insertion_zone = (
            book_back_distance <= self.config.final_insertion_distance_threshold
            and (book_controlled or book_caddy_contact)
        )
        state_objects = self._state_objects_for_env(env)
        book_state = state_objects[self.config.black_book_state_name]
        back_compartment_state = state_objects[
            self.config.back_compartment_state_name
        ]
        book_in_back_compartment = bool(back_compartment_state.check_contain(book_state))

        return {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "env_success": _check_success(env),
            "gripper_width": gripper_width,
            "gripper_closed": gripper_closed,
            "black_book_x": float(book_position[0]),
            "black_book_y": float(book_position[1]),
            "black_book_z": float(book_position[2]),
            "black_book_qw": float(book_quaternion[0]),
            "black_book_qx": float(book_quaternion[1]),
            "black_book_qy": float(book_quaternion[2]),
            "black_book_qz": float(book_quaternion[3]),
            "black_book_motion": book_motion,
            "black_book_gripper_contact": book_gripper_contact,
            "black_book_controlled": book_controlled,
            "black_book_caddy_contact": book_caddy_contact,
            "black_book_back_distance": book_back_distance,
            "black_book_final_insertion_zone": book_final_insertion_zone,
            "black_book_in_back_compartment": book_in_back_compartment,
            "white_yellow_mug_gripper_contact": white_yellow_mug_gripper_contact,
        }

    def _book_motion(self, position: np.ndarray) -> float:
        previous = self._previous_book_position
        self._previous_book_position = position.copy()
        if previous is None:
            return 0.0
        return float(np.linalg.norm(position - previous))

    def _start_episode(self, episode_index: int) -> None:
        if self._current_episode_index == episode_index:
            return
        self._current_episode_index = episode_index
        self._previous_book_position = None

    def _state_objects_for_env(self, env: Any) -> Any:
        if self._state_objects is not None:
            return self._state_objects
        base_env = getattr(env, "env", env)
        state_objects = getattr(base_env, "object_states_dict", None)
        if state_objects is None:
            raise AttributeError(
                "LIBERO task5 environment does not expose object_states_dict after reset."
            )
        required_state_names = {
            self.config.black_book_state_name,
            self.config.back_compartment_state_name,
        }
        missing_state_names = required_state_names - set(state_objects)
        if missing_state_names:
            raise ValueError(
                "LIBERO task5 semantic states are unavailable: "
                f"{sorted(missing_state_names)}"
            )
        self._state_objects = state_objects
        return state_objects


class Task6SemanticTraceRecorder:
    """Extract task6 privileged state from a live LIBERO simulator."""

    def __init__(self, env: Any, config: Task6SemanticTraceConfig | None = None):
        self.config = config or Task6SemanticTraceConfig()
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._sim = sim
        model = sim.model
        self.bodies = Task6SemanticBodies(
            porcelain_mug=_resolve_body_name(model, self.config.porcelain_mug_alias),
            chocolate_pudding=_resolve_body_name(
                model, self.config.chocolate_pudding_alias
            ),
            plate=_resolve_body_name(model, self.config.plate_alias),
            red_coffee_mug=_resolve_body_name(
                model, self.config.red_coffee_mug_alias
            ),
        )
        self._porcelain_mug_geoms = _geom_ids_for_body(
            model, self.bodies.porcelain_mug, self.config.porcelain_mug_alias
        )
        self._chocolate_pudding_geoms = _geom_ids_for_body(
            model, self.bodies.chocolate_pudding, self.config.chocolate_pudding_alias
        )
        self._red_coffee_mug_geoms = _geom_ids_for_body(
            model, self.bodies.red_coffee_mug, self.config.red_coffee_mug_alias
        )
        self._gripper_geoms = _gripper_geom_ids(model)
        self._state_objects: Any | None = None
        self._previous_positions: dict[str, np.ndarray] = {}
        self._current_episode_index: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the resolved task6 trace schema for reproducibility."""
        return {
            "version": "task6_semantic_trace_v1",
            "config": asdict(self.config),
            "bodies": asdict(self.bodies),
            "state_names": {
                "porcelain_mug": self.config.porcelain_mug_state_name,
                "chocolate_pudding": self.config.chocolate_pudding_state_name,
                "plate": self.config.plate_state_name,
                "plate_left_region": self.config.plate_left_region_state_name,
                "plate_right_region": self.config.plate_right_region_state_name,
            },
        }

    def capture(
        self,
        env: Any,
        episode_index: int,
        frame_index: int,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture one simulator-aligned task6 semantic trace record."""
        sim = getattr(env, "sim", None)
        if sim is None:
            raise AttributeError("LIBERO environment does not expose env.sim.")
        self._start_episode(int(episode_index))
        porcelain_mug_position, porcelain_mug_quaternion = _body_pose(
            sim, self.bodies.porcelain_mug
        )
        chocolate_pudding_position, chocolate_pudding_quaternion = _body_pose(
            sim, self.bodies.chocolate_pudding
        )
        porcelain_mug_motion = self._motion(
            "porcelain_mug", porcelain_mug_position
        )
        chocolate_pudding_motion = self._motion(
            "chocolate_pudding", chocolate_pudding_position
        )
        gripper_width = Task1SemanticTraceRecorder._gripper_width(observation, env)
        gripper_closed = gripper_width <= self.config.gripper_width_threshold
        porcelain_mug_gripper_contact = _has_contact(
            sim, self._porcelain_mug_geoms, self._gripper_geoms
        )
        chocolate_pudding_gripper_contact = _has_contact(
            sim, self._chocolate_pudding_geoms, self._gripper_geoms
        )
        red_coffee_mug_gripper_contact = _has_contact(
            sim, self._red_coffee_mug_geoms, self._gripper_geoms
        )
        porcelain_mug_controlled = (
            gripper_closed
            and porcelain_mug_gripper_contact
            and porcelain_mug_motion >= self.config.controlled_motion_threshold
        )
        chocolate_pudding_controlled = (
            gripper_closed
            and chocolate_pudding_gripper_contact
            and chocolate_pudding_motion >= self.config.controlled_motion_threshold
        )
        state_objects = self._state_objects_for_env(env)
        porcelain_mug_state = state_objects[self.config.porcelain_mug_state_name]
        chocolate_pudding_state = state_objects[
            self.config.chocolate_pudding_state_name
        ]
        plate_state = state_objects[self.config.plate_state_name]
        plate_left_region_state = state_objects[
            self.config.plate_left_region_state_name
        ]
        plate_right_region_state = state_objects[
            self.config.plate_right_region_state_name
        ]
        porcelain_mug_on_plate = bool(plate_state.check_ontop(porcelain_mug_state))
        chocolate_pudding_on_plate = bool(
            plate_state.check_ontop(chocolate_pudding_state)
        )
        chocolate_pudding_on_plate_left_region = bool(
            plate_left_region_state.check_ontop(chocolate_pudding_state)
        )
        chocolate_pudding_on_plate_right_region = bool(
            plate_right_region_state.check_ontop(chocolate_pudding_state)
        )

        return {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "env_success": _check_success(env),
            "gripper_width": gripper_width,
            "gripper_closed": gripper_closed,
            "porcelain_mug_x": float(porcelain_mug_position[0]),
            "porcelain_mug_y": float(porcelain_mug_position[1]),
            "porcelain_mug_z": float(porcelain_mug_position[2]),
            "porcelain_mug_qw": float(porcelain_mug_quaternion[0]),
            "porcelain_mug_qx": float(porcelain_mug_quaternion[1]),
            "porcelain_mug_qy": float(porcelain_mug_quaternion[2]),
            "porcelain_mug_qz": float(porcelain_mug_quaternion[3]),
            "porcelain_mug_motion": porcelain_mug_motion,
            "porcelain_mug_gripper_contact": porcelain_mug_gripper_contact,
            "porcelain_mug_controlled": porcelain_mug_controlled,
            "chocolate_pudding_x": float(chocolate_pudding_position[0]),
            "chocolate_pudding_y": float(chocolate_pudding_position[1]),
            "chocolate_pudding_z": float(chocolate_pudding_position[2]),
            "chocolate_pudding_qw": float(chocolate_pudding_quaternion[0]),
            "chocolate_pudding_qx": float(chocolate_pudding_quaternion[1]),
            "chocolate_pudding_qy": float(chocolate_pudding_quaternion[2]),
            "chocolate_pudding_qz": float(chocolate_pudding_quaternion[3]),
            "chocolate_pudding_motion": chocolate_pudding_motion,
            "chocolate_pudding_gripper_contact": chocolate_pudding_gripper_contact,
            "chocolate_pudding_controlled": chocolate_pudding_controlled,
            "porcelain_mug_on_plate": porcelain_mug_on_plate,
            "chocolate_pudding_on_plate": chocolate_pudding_on_plate,
            "chocolate_pudding_on_plate_left_region": (
                chocolate_pudding_on_plate_left_region
            ),
            "chocolate_pudding_on_plate_right_region": (
                chocolate_pudding_on_plate_right_region
            ),
            "red_coffee_mug_gripper_contact": red_coffee_mug_gripper_contact,
        }

    def _motion(self, key: str, position: np.ndarray) -> float:
        previous = self._previous_positions.get(key)
        self._previous_positions[key] = position.copy()
        if previous is None:
            return 0.0
        return float(np.linalg.norm(position - previous))

    def _start_episode(self, episode_index: int) -> None:
        if self._current_episode_index == episode_index:
            return
        self._current_episode_index = episode_index
        self._previous_positions.clear()

    def _state_objects_for_env(self, env: Any) -> Any:
        if self._state_objects is not None:
            return self._state_objects
        base_env = getattr(env, "env", env)
        state_objects = getattr(base_env, "object_states_dict", None)
        if state_objects is None:
            raise AttributeError(
                "LIBERO task6 environment does not expose object_states_dict after reset."
            )
        required_state_names = {
            self.config.porcelain_mug_state_name,
            self.config.chocolate_pudding_state_name,
            self.config.plate_state_name,
            self.config.plate_left_region_state_name,
            self.config.plate_right_region_state_name,
        }
        missing_state_names = required_state_names - set(state_objects)
        if missing_state_names:
            raise ValueError(
                "LIBERO task6 semantic states are unavailable: "
                f"{sorted(missing_state_names)}"
            )
        self._state_objects = state_objects
        return state_objects


def _first_stable_frame(mask: np.ndarray, stable_frames: int, start: int = 0) -> int | None:
    run_start: int | None = None
    for frame_index in range(start, len(mask)):
        if bool(mask[frame_index]):
            if run_start is None:
                run_start = frame_index
            if frame_index - run_start + 1 >= stable_frames:
                return run_start
        else:
            run_start = None
    return None


def _phase_progress(
    phase: np.ndarray,
    *,
    is_success: bool,
    num_phases: int = NUM_TASK1_PHASES,
) -> tuple[np.ndarray, np.ndarray]:
    """Build time-based phase progress from verified phase boundaries.

    Completed phases, including phase zero, increase linearly from zero to
    one. For failed episodes, the final observed phase increases only to
    FAILED_TERMINAL_PHASE_PROGRESS_CAP during its first half and then remains
    at that value.
    """
    phase_progress = np.zeros(len(phase), dtype=np.float32)
    if not len(phase):
        return phase_progress, phase_progress

    terminal_phase = int(phase[-1])
    for phase_id in np.unique(phase):
        indices = np.flatnonzero(phase == phase_id)
        is_incomplete_terminal_phase = not is_success and phase_id == terminal_phase
        if is_incomplete_terminal_phase:
            if len(indices) == 1:
                continue
            ramp_length = max(2, (len(indices) + 1) // 2)
            phase_progress[indices[:ramp_length]] = np.linspace(
                0.0,
                FAILED_TERMINAL_PHASE_PROGRESS_CAP,
                ramp_length,
                dtype=np.float32,
            )
            phase_progress[indices[ramp_length:]] = FAILED_TERMINAL_PHASE_PROGRESS_CAP
        elif len(indices) == 1:
            phase_progress[indices] = 1.0
        else:
            phase_progress[indices] = np.linspace(
                0.0, 1.0, len(indices), dtype=np.float32
            )

    global_progress = (phase.astype(np.float32) + phase_progress) / num_phases
    return phase_progress, global_progress


def build_task1_phase_labels(
    trace: pd.DataFrame,
    *,
    stable_frames: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build four monotonic task1 phases and an episode-level audit table."""
    required = {
        "episode_index",
        "frame_index",
        "is_success",
        "object_a_controlled",
        "object_b_controlled",
        "object_a_basket_contact",
        "object_b_basket_contact",
        "object_a_near_basket",
        "object_b_near_basket",
    }
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"Semantic trace missing columns: {sorted(missing)}")

    label_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for episode_index, episode_trace in trace.groupby("episode_index", sort=True):
        episode_trace = episode_trace.sort_values("frame_index").reset_index(drop=True)
        length = len(episode_trace)
        is_success = bool(episode_trace["is_success"].iloc[-1])
        controlled_a = episode_trace["object_a_controlled"].to_numpy(dtype=bool)
        controlled_b = episode_trace["object_b_controlled"].to_numpy(dtype=bool)
        in_basket_a = episode_trace["object_a_basket_contact"].to_numpy(dtype=bool)
        in_basket_b = episode_trace["object_b_basket_contact"].to_numpy(dtype=bool)
        b1 = _first_stable_frame(controlled_a | controlled_b, stable_frames)
        one_in_basket = in_basket_a | in_basket_b
        joint_transfer = controlled_a & controlled_b & (
            episode_trace["object_a_near_basket"].to_numpy(dtype=bool)
        ) & (episode_trace["object_b_near_basket"].to_numpy(dtype=bool))
        b2 = _first_stable_frame(one_in_basket | joint_transfer, stable_frames, b1 or 0)
        both_in_basket = in_basket_a & in_basket_b
        b3 = _first_stable_frame(both_in_basket, stable_frames, b2 or 0)
        b3_source = "state_stable" if b3 is not None else "unresolved"
        if b3 is None and is_success and len(both_in_basket) and both_in_basket[-1]:
            b3 = len(both_in_basket) - 1
            while b3 > 0 and both_in_basket[b3 - 1]:
                b3 -= 1
            b3_source = "state_terminal_success"
        if b3 is None and is_success and "env_success" in episode_trace:
            success_frames = np.flatnonzero(
                episode_trace["env_success"].to_numpy(dtype=bool)
            )
            if len(success_frames):
                b3 = int(success_frames[-1])
                b3_source = "env_success_terminal"
        if not is_success:
            b3 = None
            b3_source = "failed_episode"

        phase = np.zeros(length, dtype=np.int64)
        if b1 is not None:
            phase[b1:] = 1
        if b2 is not None and b1 is not None and b2 > b1:
            phase[b2:] = 2
        else:
            b2 = None
        if b3 is not None and b2 is not None and b3 > b2:
            phase[b3:] = 3
        else:
            b3 = None
            if is_success:
                success_frames = np.flatnonzero(
                    episode_trace.get(
                        "env_success",
                        pd.Series(False, index=episode_trace.index),
                    ).to_numpy(dtype=bool)
                )
                if len(success_frames) and b2 is not None and success_frames[-1] > b2:
                    b3 = int(success_frames[-1])
                    phase[b3:] = 3
                    b3_source = "env_success_terminal"
                else:
                    b3_source = "unresolved"

        phase_progress, global_progress = _phase_progress(
            phase, is_success=is_success
        )

        label_frames.append(
            pd.DataFrame(
                {
                    "episode_index": int(episode_index),
                    "frame_index": episode_trace["frame_index"].to_numpy(dtype=np.int64),
                    "phase": phase,
                    "phase_progress": phase_progress,
                    "global_progress": global_progress,
                    "semantic_source": "simulator_trace",
                    "semantic_confidence": np.where(
                        b1 is not None and (not is_success or b3 is not None),
                        "state_verified",
                        "unresolved",
                    ),
                    "is_success": is_success,
                }
            )
        )
        audit_rows.append(
            {
                "episode_index": int(episode_index),
                "episode_length": length,
                "is_success": is_success,
                "b1_frame": b1,
                "b2_frame": b2,
                "b3_frame": b3,
                "b3_source": b3_source,
                "b2_joint_transfer_detected": bool(joint_transfer.any()),
                "b3_consistent_with_success": (b3 is not None) == is_success,
                "trainable": b1 is not None and (not is_success or b3 is not None),
            }
        )
    return pd.concat(label_frames, ignore_index=True), pd.DataFrame(audit_rows)


def write_task1_semantic_artifacts(
    dataset_path: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    output_name: str = "semantic_trace_task1",
    stable_frames: int = 5,
) -> dict[str, str]:
    """Write raw trace, four-phase labels, audit rows, and trace metadata."""
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_path = meta_dir / f"{output_name}.parquet"
    labels_path = meta_dir / f"phase_progress_{output_name}.parquet"
    audit_path = meta_dir / f"{output_name}_audit.csv"
    metadata_path = meta_dir / f"{output_name}_metadata.json"
    trace = pd.DataFrame(records).sort_values(["episode_index", "frame_index"])
    labels, audit = build_task1_phase_labels(trace, stable_frames=stable_frames)
    trace.to_parquet(raw_path, index=False)
    labels.to_parquet(labels_path, index=False)
    audit.to_csv(audit_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return {
        "raw_trace": str(raw_path),
        "phase_labels": str(labels_path),
        "audit": str(audit_path),
        "metadata": str(metadata_path),
    }


def _first_true_frame(mask: np.ndarray, start: int = 0) -> int | None:
    true_frames = np.flatnonzero(mask[start:])
    if not len(true_frames):
        return None
    return int(start + true_frames[0])


def _first_boundary(
    candidates: dict[str, int | None],
) -> tuple[int | None, str]:
    present = {source: frame for source, frame in candidates.items() if frame is not None}
    if not present:
        return None, "unresolved"
    boundary = min(present.values())
    sources = sorted(source for source, frame in present.items() if frame == boundary)
    return boundary, "+".join(sources)


def build_task2_phase_labels(
    trace: pd.DataFrame,
    *,
    stable_frames: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build four monotonic phases for turning on a stove and placing moka pot."""
    required = {
        "episode_index",
        "frame_index",
        "is_success",
        "env_success",
        "moka_pot_controlled",
        "stove_button_interacted",
        "stove_turn_on",
        "moka_pot_on_cook_region",
        "frypan_gripper_contact",
    }
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"Semantic trace missing columns: {sorted(missing)}")

    label_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for episode_index, episode_trace in trace.groupby("episode_index", sort=True):
        episode_trace = episode_trace.sort_values("frame_index").reset_index(drop=True)
        length = len(episode_trace)
        is_success = bool(episode_trace["is_success"].iloc[-1])
        moka_pot_controlled = episode_trace["moka_pot_controlled"].to_numpy(dtype=bool)
        stove_button_interacted = episode_trace[
            "stove_button_interacted"
        ].to_numpy(dtype=bool)
        stove_turn_on = episode_trace["stove_turn_on"].to_numpy(dtype=bool)
        moka_pot_on_cook_region = episode_trace[
            "moka_pot_on_cook_region"
        ].to_numpy(dtype=bool)
        env_success = episode_trace["env_success"].to_numpy(dtype=bool)

        b1, b1_source = _first_boundary(
            {
                "moka_pot_controlled": _first_stable_frame(
                    moka_pot_controlled, stable_frames
                ),
                "stove_button_interaction": _first_true_frame(
                    stove_button_interacted
                ),
            }
        )
        b2_start = b1 + 1 if b1 is not None else 0
        b2, b2_source = _first_boundary(
            {
                "stove_turn_on": _first_stable_frame(
                    stove_turn_on, stable_frames, b2_start
                ),
                "moka_pot_on_cook_region": _first_stable_frame(
                    moka_pot_on_cook_region, stable_frames, b2_start
                ),
            }
        )
        if b2 is None and is_success:
            b2, b2_source = _first_boundary(
                {
                    "stove_turn_on_terminal": _first_true_frame(
                        stove_turn_on, b2_start
                    ),
                    "moka_pot_on_cook_region_terminal": _first_true_frame(
                        moka_pot_on_cook_region, b2_start
                    ),
                }
            )

        b3 = _first_stable_frame(
            stove_turn_on & moka_pot_on_cook_region,
            stable_frames,
            b2 + 1 if b2 is not None else 0,
        )
        b3_source = "state_stable" if b3 is not None else "unresolved"
        if b3 is None and is_success:
            b3 = _first_true_frame(env_success, b2 + 1 if b2 is not None else 0)
            if b3 is not None:
                b3_source = "env_success_terminal"
        if not is_success:
            b3 = None
            b3_source = "failed_episode"

        phase = np.zeros(length, dtype=np.int64)
        if b1 is not None:
            phase[b1:] = 1
        if b2 is not None and b1 is not None and b2 > b1:
            phase[b2:] = 2
        else:
            b2 = None
            b2_source = "unresolved"
        if b3 is not None and b2 is not None and b3 > b2:
            phase[b3:] = TASK2_SUCCESS_PHASE
        else:
            b3 = None
            if is_success:
                b3_source = "unresolved"

        phase_progress, global_progress = _phase_progress(
            phase, is_success=is_success, num_phases=NUM_TASK2_PHASES
        )

        label_frames.append(
            pd.DataFrame(
                {
                    "episode_index": int(episode_index),
                    "frame_index": episode_trace["frame_index"].to_numpy(dtype=np.int64),
                    "phase": phase,
                    "phase_progress": phase_progress,
                    "global_progress": global_progress,
                    "semantic_source": "simulator_trace",
                    "semantic_confidence": np.where(
                        b1 is not None and (not is_success or b3 is not None),
                        "state_verified",
                        "unresolved",
                    ),
                    "is_success": is_success,
                }
            )
        )
        first_stove_turn_on = _first_true_frame(stove_turn_on)
        first_moka_on_cook_region = _first_true_frame(moka_pot_on_cook_region)
        if first_stove_turn_on is None and first_moka_on_cook_region is None:
            subgoal_order = "none"
        elif first_moka_on_cook_region is None or (
            first_stove_turn_on is not None
            and first_stove_turn_on < first_moka_on_cook_region
        ):
            subgoal_order = "stove_then_moka_pot"
        elif first_stove_turn_on is None or (
            first_moka_on_cook_region < first_stove_turn_on
        ):
            subgoal_order = "moka_pot_then_stove"
        else:
            subgoal_order = "simultaneous"
        audit_rows.append(
            {
                "episode_index": int(episode_index),
                "episode_length": length,
                "is_success": is_success,
                "b1_frame": b1,
                "b1_source": b1_source,
                "b2_frame": b2,
                "b2_source": b2_source,
                "b3_frame": b3,
                "b3_source": b3_source,
                "subgoal_order": subgoal_order,
                "stove_turn_on_observed": bool(stove_turn_on.any()),
                "moka_pot_on_cook_region_observed": bool(
                    moka_pot_on_cook_region.any()
                ),
                "frypan_gripper_contact_observed": bool(
                    episode_trace["frypan_gripper_contact"].to_numpy(dtype=bool).any()
                ),
                "b3_consistent_with_success": (b3 is not None) == is_success,
                "trainable": b1 is not None and (not is_success or b3 is not None),
            }
        )
    return pd.concat(label_frames, ignore_index=True), pd.DataFrame(audit_rows)


def write_task2_semantic_artifacts(
    dataset_path: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    output_name: str = "semantic_trace_task2",
    stable_frames: int = 3,
) -> dict[str, str]:
    """Write raw task2 trace, phase labels, audit rows, and metadata."""
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_path = meta_dir / f"{output_name}.parquet"
    labels_path = meta_dir / f"phase_progress_{output_name}.parquet"
    audit_path = meta_dir / f"{output_name}_audit.csv"
    metadata_path = meta_dir / f"{output_name}_metadata.json"
    trace = pd.DataFrame(records).sort_values(["episode_index", "frame_index"])
    labels, audit = build_task2_phase_labels(trace, stable_frames=stable_frames)
    trace.to_parquet(raw_path, index=False)
    labels.to_parquet(labels_path, index=False)
    audit.to_csv(audit_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return {
        "raw_trace": str(raw_path),
        "phase_labels": str(labels_path),
        "audit": str(audit_path),
        "metadata": str(metadata_path),
    }


def build_task3_phase_labels(
    trace: pd.DataFrame,
    *,
    stable_frames: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build four monotonic phases for placing a bowl and closing its drawer."""
    required = {
        "episode_index",
        "frame_index",
        "is_success",
        "env_success",
        "black_bowl_controlled",
        "bottom_drawer_interacted",
        "bottom_drawer_is_close",
        "black_bowl_in_bottom_drawer",
        "wine_bottle_gripper_contact",
        "wine_rack_gripper_contact",
    }
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"Semantic trace missing columns: {sorted(missing)}")

    label_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for episode_index, episode_trace in trace.groupby("episode_index", sort=True):
        episode_trace = episode_trace.sort_values("frame_index").reset_index(drop=True)
        length = len(episode_trace)
        is_success = bool(episode_trace["is_success"].iloc[-1])
        black_bowl_controlled = episode_trace[
            "black_bowl_controlled"
        ].to_numpy(dtype=bool)
        bottom_drawer_interacted = episode_trace[
            "bottom_drawer_interacted"
        ].to_numpy(dtype=bool)
        bottom_drawer_is_close = episode_trace[
            "bottom_drawer_is_close"
        ].to_numpy(dtype=bool)
        black_bowl_in_bottom_drawer = episode_trace[
            "black_bowl_in_bottom_drawer"
        ].to_numpy(dtype=bool)
        env_success = episode_trace["env_success"].to_numpy(dtype=bool)

        b1, b1_source = _first_boundary(
            {
                "black_bowl_controlled": _first_stable_frame(
                    black_bowl_controlled, stable_frames
                ),
                "bottom_drawer_interaction": _first_true_frame(
                    bottom_drawer_interacted
                ),
            }
        )
        b2_start = b1 + 1 if b1 is not None else 0
        b2 = _first_stable_frame(
            black_bowl_in_bottom_drawer, stable_frames, b2_start
        )
        b2_source = "black_bowl_in_bottom_drawer" if b2 is not None else "unresolved"
        if b2 is None and is_success:
            b2 = _first_true_frame(black_bowl_in_bottom_drawer, b2_start)
            if b2 is not None:
                b2_source = "black_bowl_in_bottom_drawer_terminal"

        b3 = _first_stable_frame(
            black_bowl_in_bottom_drawer & bottom_drawer_is_close,
            stable_frames,
            b2 + 1 if b2 is not None else 0,
        )
        b3_source = "state_stable" if b3 is not None else "unresolved"
        if b3 is None and is_success:
            b3 = _first_true_frame(env_success, b2 + 1 if b2 is not None else 0)
            if b3 is not None:
                b3_source = "env_success_terminal"
        if not is_success:
            b3 = None
            b3_source = "failed_episode"

        phase = np.zeros(length, dtype=np.int64)
        if b1 is not None:
            phase[b1:] = 1
        if b2 is not None and b1 is not None and b2 > b1:
            phase[b2:] = 2
        else:
            b2 = None
            b2_source = "unresolved"
        if b3 is not None and b2 is not None and b3 > b2:
            phase[b3:] = TASK3_SUCCESS_PHASE
        else:
            b3 = None
            if is_success:
                b3_source = "unresolved"

        phase_progress, global_progress = _phase_progress(
            phase, is_success=is_success, num_phases=NUM_TASK3_PHASES
        )

        label_frames.append(
            pd.DataFrame(
                {
                    "episode_index": int(episode_index),
                    "frame_index": episode_trace["frame_index"].to_numpy(dtype=np.int64),
                    "phase": phase,
                    "phase_progress": phase_progress,
                    "global_progress": global_progress,
                    "semantic_source": "simulator_trace",
                    "semantic_confidence": np.where(
                        b1 is not None and (not is_success or b3 is not None),
                        "state_verified",
                        "unresolved",
                    ),
                    "is_success": is_success,
                }
            )
        )
        audit_rows.append(
            {
                "episode_index": int(episode_index),
                "episode_length": length,
                "is_success": is_success,
                "b1_frame": b1,
                "b1_source": b1_source,
                "b2_frame": b2,
                "b2_source": b2_source,
                "b3_frame": b3,
                "b3_source": b3_source,
                "drawer_closed_before_bowl_observed": bool(
                    (bottom_drawer_is_close & ~black_bowl_in_bottom_drawer).any()
                ),
                "wine_bottle_gripper_contact_observed": bool(
                    episode_trace["wine_bottle_gripper_contact"].to_numpy(dtype=bool).any()
                ),
                "wine_rack_gripper_contact_observed": bool(
                    episode_trace["wine_rack_gripper_contact"].to_numpy(dtype=bool).any()
                ),
                "b3_consistent_with_success": (b3 is not None) == is_success,
                "trainable": b1 is not None and (not is_success or b3 is not None),
            }
        )
    return pd.concat(label_frames, ignore_index=True), pd.DataFrame(audit_rows)


def write_task3_semantic_artifacts(
    dataset_path: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    output_name: str = "semantic_trace_task3",
    stable_frames: int = 3,
) -> dict[str, str]:
    """Write raw task3 trace, phase labels, audit rows, and metadata."""
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_path = meta_dir / f"{output_name}.parquet"
    labels_path = meta_dir / f"phase_progress_{output_name}.parquet"
    audit_path = meta_dir / f"{output_name}_audit.csv"
    metadata_path = meta_dir / f"{output_name}_metadata.json"
    trace = pd.DataFrame(records).sort_values(["episode_index", "frame_index"])
    labels, audit = build_task3_phase_labels(trace, stable_frames=stable_frames)
    trace.to_parquet(raw_path, index=False)
    labels.to_parquet(labels_path, index=False)
    audit.to_csv(audit_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return {
        "raw_trace": str(raw_path),
        "phase_labels": str(labels_path),
        "audit": str(audit_path),
        "metadata": str(metadata_path),
    }


def build_task4_phase_labels(
    trace: pd.DataFrame,
    *,
    stable_frames: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build four monotonic phases for placing two mugs on correct plates."""
    required = {
        "episode_index",
        "frame_index",
        "is_success",
        "env_success",
        "porcelain_mug_controlled",
        "white_yellow_mug_controlled",
        "porcelain_mug_on_left_plate",
        "white_yellow_mug_on_right_plate",
        "porcelain_mug_on_right_plate",
        "white_yellow_mug_on_left_plate",
        "red_coffee_mug_gripper_contact",
    }
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"Semantic trace missing columns: {sorted(missing)}")

    label_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for episode_index, episode_trace in trace.groupby("episode_index", sort=True):
        episode_trace = episode_trace.sort_values("frame_index").reset_index(drop=True)
        length = len(episode_trace)
        is_success = bool(episode_trace["is_success"].iloc[-1])
        porcelain_mug_controlled = episode_trace[
            "porcelain_mug_controlled"
        ].to_numpy(dtype=bool)
        white_yellow_mug_controlled = episode_trace[
            "white_yellow_mug_controlled"
        ].to_numpy(dtype=bool)
        porcelain_mug_on_left_plate = episode_trace[
            "porcelain_mug_on_left_plate"
        ].to_numpy(dtype=bool)
        white_yellow_mug_on_right_plate = episode_trace[
            "white_yellow_mug_on_right_plate"
        ].to_numpy(dtype=bool)
        env_success = episode_trace["env_success"].to_numpy(dtype=bool)

        b1, b1_source = _first_boundary(
            {
                "porcelain_mug_controlled": _first_stable_frame(
                    porcelain_mug_controlled, stable_frames
                ),
                "white_yellow_mug_controlled": _first_stable_frame(
                    white_yellow_mug_controlled, stable_frames
                ),
            }
        )
        b2_start = b1 + 1 if b1 is not None else 0
        b2, b2_source = _first_boundary(
            {
                "porcelain_mug_on_left_plate": _first_stable_frame(
                    porcelain_mug_on_left_plate, stable_frames, b2_start
                ),
                "white_yellow_mug_on_right_plate": _first_stable_frame(
                    white_yellow_mug_on_right_plate, stable_frames, b2_start
                ),
            }
        )
        if b2 is None and is_success:
            b2, b2_source = _first_boundary(
                {
                    "porcelain_mug_on_left_plate_terminal": _first_true_frame(
                        porcelain_mug_on_left_plate, b2_start
                    ),
                    "white_yellow_mug_on_right_plate_terminal": _first_true_frame(
                        white_yellow_mug_on_right_plate, b2_start
                    ),
                }
            )

        b3 = _first_stable_frame(
            porcelain_mug_on_left_plate & white_yellow_mug_on_right_plate,
            stable_frames,
            b2 + 1 if b2 is not None else 0,
        )
        b3_source = "state_stable" if b3 is not None else "unresolved"
        if b3 is None and is_success:
            b3 = _first_true_frame(env_success, b2 + 1 if b2 is not None else 0)
            if b3 is not None:
                b3_source = "env_success_terminal"
        if not is_success:
            b3 = None
            b3_source = "failed_episode"

        phase = np.zeros(length, dtype=np.int64)
        if b1 is not None:
            phase[b1:] = 1
        if b2 is not None and b1 is not None and b2 > b1:
            phase[b2:] = 2
        else:
            b2 = None
            b2_source = "unresolved"
        if b3 is not None and b2 is not None and b3 > b2:
            phase[b3:] = TASK4_SUCCESS_PHASE
        else:
            b3 = None
            if is_success:
                b3_source = "unresolved"

        phase_progress, global_progress = _phase_progress(
            phase, is_success=is_success, num_phases=NUM_TASK4_PHASES
        )

        porcelain_mug_on_right_plate = episode_trace[
            "porcelain_mug_on_right_plate"
        ].to_numpy(dtype=bool)
        white_yellow_mug_on_left_plate = episode_trace[
            "white_yellow_mug_on_left_plate"
        ].to_numpy(dtype=bool)
        label_frames.append(
            pd.DataFrame(
                {
                    "episode_index": int(episode_index),
                    "frame_index": episode_trace["frame_index"].to_numpy(dtype=np.int64),
                    "phase": phase,
                    "phase_progress": phase_progress,
                    "global_progress": global_progress,
                    "semantic_source": "simulator_trace",
                    "semantic_confidence": np.where(
                        b1 is not None and (not is_success or b3 is not None),
                        "state_verified",
                        "unresolved",
                    ),
                    "is_success": is_success,
                }
            )
        )
        audit_rows.append(
            {
                "episode_index": int(episode_index),
                "episode_length": length,
                "is_success": is_success,
                "b1_frame": b1,
                "b1_source": b1_source,
                "b2_frame": b2,
                "b2_source": b2_source,
                "b3_frame": b3,
                "b3_source": b3_source,
                "porcelain_mug_on_left_plate_observed": bool(
                    porcelain_mug_on_left_plate.any()
                ),
                "white_yellow_mug_on_right_plate_observed": bool(
                    white_yellow_mug_on_right_plate.any()
                ),
                "incorrect_plate_placement_observed": bool(
                    porcelain_mug_on_right_plate.any()
                    or white_yellow_mug_on_left_plate.any()
                ),
                "red_coffee_mug_gripper_contact_observed": bool(
                    episode_trace[
                        "red_coffee_mug_gripper_contact"
                    ].to_numpy(dtype=bool).any()
                ),
                "b3_consistent_with_success": (b3 is not None) == is_success,
                "trainable": b1 is not None and (not is_success or b3 is not None),
            }
        )
    return pd.concat(label_frames, ignore_index=True), pd.DataFrame(audit_rows)


def write_task4_semantic_artifacts(
    dataset_path: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    output_name: str = "semantic_trace_task4",
    stable_frames: int = 3,
) -> dict[str, str]:
    """Write raw task4 trace, phase labels, audit rows, and metadata."""
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_path = meta_dir / f"{output_name}.parquet"
    labels_path = meta_dir / f"phase_progress_{output_name}.parquet"
    audit_path = meta_dir / f"{output_name}_audit.csv"
    metadata_path = meta_dir / f"{output_name}_metadata.json"
    trace = pd.DataFrame(records).sort_values(["episode_index", "frame_index"])
    labels, audit = build_task4_phase_labels(trace, stable_frames=stable_frames)
    trace.to_parquet(raw_path, index=False)
    labels.to_parquet(labels_path, index=False)
    audit.to_csv(audit_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return {
        "raw_trace": str(raw_path),
        "phase_labels": str(labels_path),
        "audit": str(audit_path),
        "metadata": str(metadata_path),
    }


def build_task5_phase_labels(
    trace: pd.DataFrame,
    *,
    stable_frames: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build four monotonic phases for inserting a book into a caddy."""
    required = {
        "episode_index",
        "frame_index",
        "is_success",
        "env_success",
        "black_book_controlled",
        "black_book_final_insertion_zone",
        "black_book_in_back_compartment",
        "black_book_back_distance",
        "black_book_caddy_contact",
        "white_yellow_mug_gripper_contact",
    }
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"Semantic trace missing columns: {sorted(missing)}")

    label_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for episode_index, episode_trace in trace.groupby("episode_index", sort=True):
        episode_trace = episode_trace.sort_values("frame_index").reset_index(drop=True)
        length = len(episode_trace)
        is_success = bool(episode_trace["is_success"].iloc[-1])
        black_book_controlled = episode_trace[
            "black_book_controlled"
        ].to_numpy(dtype=bool)
        black_book_final_insertion_zone = episode_trace[
            "black_book_final_insertion_zone"
        ].to_numpy(dtype=bool)
        black_book_in_back_compartment = episode_trace[
            "black_book_in_back_compartment"
        ].to_numpy(dtype=bool)
        env_success = episode_trace["env_success"].to_numpy(dtype=bool)

        b1 = _first_stable_frame(black_book_controlled, stable_frames)
        b1_source = "black_book_controlled" if b1 is not None else "unresolved"
        b2_start = b1 + 1 if b1 is not None else 0
        b2 = _first_stable_frame(
            black_book_final_insertion_zone, stable_frames, b2_start
        )
        b2_source = "black_book_final_insertion_zone" if b2 is not None else "unresolved"
        if b2 is None and is_success:
            b2 = _first_true_frame(black_book_in_back_compartment, b2_start)
            if b2 is not None:
                b2_source = "black_book_in_back_compartment_terminal"

        b3 = _first_stable_frame(
            black_book_in_back_compartment,
            stable_frames,
            b2 + 1 if b2 is not None else 0,
        )
        b3_source = "state_stable" if b3 is not None else "unresolved"
        if b3 is None and is_success:
            b3 = _first_true_frame(env_success, b2 + 1 if b2 is not None else 0)
            if b3 is not None:
                b3_source = "env_success_terminal"
        if not is_success:
            b3 = None
            b3_source = "failed_episode"

        phase = np.zeros(length, dtype=np.int64)
        if b1 is not None:
            phase[b1:] = 1
        if b2 is not None and b1 is not None and b2 > b1:
            phase[b2:] = 2
        else:
            b2 = None
            b2_source = "unresolved"
        if b3 is not None and b2 is not None and b3 > b2:
            phase[b3:] = TASK5_SUCCESS_PHASE
        else:
            b3 = None
            if is_success:
                b3_source = "unresolved"

        phase_progress, global_progress = _phase_progress(
            phase, is_success=is_success, num_phases=NUM_TASK5_PHASES
        )

        label_frames.append(
            pd.DataFrame(
                {
                    "episode_index": int(episode_index),
                    "frame_index": episode_trace["frame_index"].to_numpy(dtype=np.int64),
                    "phase": phase,
                    "phase_progress": phase_progress,
                    "global_progress": global_progress,
                    "semantic_source": "simulator_trace",
                    "semantic_confidence": np.where(
                        b1 is not None and (not is_success or b3 is not None),
                        "state_verified",
                        "unresolved",
                    ),
                    "is_success": is_success,
                }
            )
        )
        audit_rows.append(
            {
                "episode_index": int(episode_index),
                "episode_length": length,
                "is_success": is_success,
                "b1_frame": b1,
                "b1_source": b1_source,
                "b2_frame": b2,
                "b2_source": b2_source,
                "b3_frame": b3,
                "b3_source": b3_source,
                "minimum_black_book_back_distance": float(
                    episode_trace["black_book_back_distance"].min()
                ),
                "black_book_caddy_contact_observed": bool(
                    episode_trace["black_book_caddy_contact"].to_numpy(dtype=bool).any()
                ),
                "black_book_in_back_compartment_observed": bool(
                    black_book_in_back_compartment.any()
                ),
                "white_yellow_mug_gripper_contact_observed": bool(
                    episode_trace[
                        "white_yellow_mug_gripper_contact"
                    ].to_numpy(dtype=bool).any()
                ),
                "b3_consistent_with_success": (b3 is not None) == is_success,
                "trainable": b1 is not None and (not is_success or b3 is not None),
            }
        )
    return pd.concat(label_frames, ignore_index=True), pd.DataFrame(audit_rows)


def write_task5_semantic_artifacts(
    dataset_path: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    output_name: str = "semantic_trace_task5",
    stable_frames: int = 3,
) -> dict[str, str]:
    """Write raw task5 trace, phase labels, audit rows, and metadata."""
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_path = meta_dir / f"{output_name}.parquet"
    labels_path = meta_dir / f"phase_progress_{output_name}.parquet"
    audit_path = meta_dir / f"{output_name}_audit.csv"
    metadata_path = meta_dir / f"{output_name}_metadata.json"
    trace = pd.DataFrame(records).sort_values(["episode_index", "frame_index"])
    labels, audit = build_task5_phase_labels(trace, stable_frames=stable_frames)
    trace.to_parquet(raw_path, index=False)
    labels.to_parquet(labels_path, index=False)
    audit.to_csv(audit_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return {
        "raw_trace": str(raw_path),
        "phase_labels": str(labels_path),
        "audit": str(audit_path),
        "metadata": str(metadata_path),
    }


def build_task6_phase_labels(
    trace: pd.DataFrame,
    *,
    stable_frames: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build four monotonic phases for the task6 mug and pudding goals."""
    required = {
        "episode_index",
        "frame_index",
        "is_success",
        "env_success",
        "porcelain_mug_controlled",
        "chocolate_pudding_controlled",
        "porcelain_mug_on_plate",
        "chocolate_pudding_on_plate",
        "chocolate_pudding_on_plate_left_region",
        "chocolate_pudding_on_plate_right_region",
        "red_coffee_mug_gripper_contact",
    }
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"Semantic trace missing columns: {sorted(missing)}")

    label_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for episode_index, episode_trace in trace.groupby("episode_index", sort=True):
        episode_trace = episode_trace.sort_values("frame_index").reset_index(drop=True)
        length = len(episode_trace)
        is_success = bool(episode_trace["is_success"].iloc[-1])
        porcelain_mug_controlled = episode_trace[
            "porcelain_mug_controlled"
        ].to_numpy(dtype=bool)
        chocolate_pudding_controlled = episode_trace[
            "chocolate_pudding_controlled"
        ].to_numpy(dtype=bool)
        porcelain_mug_on_plate = episode_trace["porcelain_mug_on_plate"].to_numpy(
            dtype=bool
        )
        chocolate_pudding_on_plate_right_region = episode_trace[
            "chocolate_pudding_on_plate_right_region"
        ].to_numpy(dtype=bool)
        env_success = episode_trace["env_success"].to_numpy(dtype=bool)

        b1, b1_source = _first_boundary(
            {
                "porcelain_mug_controlled": _first_stable_frame(
                    porcelain_mug_controlled, stable_frames
                ),
                "chocolate_pudding_controlled": _first_stable_frame(
                    chocolate_pudding_controlled, stable_frames
                ),
            }
        )
        b2_start = b1 + 1 if b1 is not None else 0
        b2, b2_source = _first_boundary(
            {
                "porcelain_mug_on_plate": _first_stable_frame(
                    porcelain_mug_on_plate, stable_frames, b2_start
                ),
                "chocolate_pudding_on_plate_right_region": _first_stable_frame(
                    chocolate_pudding_on_plate_right_region,
                    stable_frames,
                    b2_start,
                ),
            }
        )
        if b2 is None and is_success:
            b2, b2_source = _first_boundary(
                {
                    "porcelain_mug_on_plate_terminal": _first_true_frame(
                        porcelain_mug_on_plate, b2_start
                    ),
                    "chocolate_pudding_on_plate_right_region_terminal": (
                        _first_true_frame(
                            chocolate_pudding_on_plate_right_region, b2_start
                        )
                    ),
                }
            )

        final_goal = porcelain_mug_on_plate & chocolate_pudding_on_plate_right_region
        b3 = _first_stable_frame(
            final_goal,
            stable_frames,
            b2 + 1 if b2 is not None else 0,
        )
        b3_source = "state_stable" if b3 is not None else "unresolved"
        if b3 is None and is_success:
            b3 = _first_true_frame(env_success, b2 + 1 if b2 is not None else 0)
            if b3 is not None:
                b3_source = "env_success_terminal"
        if not is_success:
            b3 = None
            b3_source = "failed_episode"

        phase = np.zeros(length, dtype=np.int64)
        if b1 is not None:
            phase[b1:] = 1
        if b2 is not None and b1 is not None and b2 > b1:
            phase[b2:] = 2
        else:
            b2 = None
            b2_source = "unresolved"
        if b3 is not None and b2 is not None and b3 > b2:
            phase[b3:] = TASK6_SUCCESS_PHASE
        else:
            b3 = None
            if is_success:
                b3_source = "unresolved"

        phase_progress, global_progress = _phase_progress(
            phase, is_success=is_success, num_phases=NUM_TASK6_PHASES
        )

        chocolate_pudding_on_plate = episode_trace[
            "chocolate_pudding_on_plate"
        ].to_numpy(dtype=bool)
        chocolate_pudding_on_plate_left_region = episode_trace[
            "chocolate_pudding_on_plate_left_region"
        ].to_numpy(dtype=bool)
        label_frames.append(
            pd.DataFrame(
                {
                    "episode_index": int(episode_index),
                    "frame_index": episode_trace["frame_index"].to_numpy(
                        dtype=np.int64
                    ),
                    "phase": phase,
                    "phase_progress": phase_progress,
                    "global_progress": global_progress,
                    "semantic_source": "simulator_trace",
                    "semantic_confidence": np.where(
                        b1 is not None and (not is_success or b3 is not None),
                        "state_verified",
                        "unresolved",
                    ),
                    "is_success": is_success,
                }
            )
        )
        audit_rows.append(
            {
                "episode_index": int(episode_index),
                "episode_length": length,
                "is_success": is_success,
                "b1_frame": b1,
                "b1_source": b1_source,
                "b2_frame": b2,
                "b2_source": b2_source,
                "b3_frame": b3,
                "b3_source": b3_source,
                "porcelain_mug_on_plate_observed": bool(porcelain_mug_on_plate.any()),
                "chocolate_pudding_on_plate_right_region_observed": bool(
                    chocolate_pudding_on_plate_right_region.any()
                ),
                "chocolate_pudding_incorrect_placement_observed": bool(
                    chocolate_pudding_on_plate.any()
                    or chocolate_pudding_on_plate_left_region.any()
                ),
                "red_coffee_mug_gripper_contact_observed": bool(
                    episode_trace["red_coffee_mug_gripper_contact"]
                    .to_numpy(dtype=bool)
                    .any()
                ),
                "b3_consistent_with_success": (b3 is not None) == is_success,
                "trainable": b1 is not None and (not is_success or b3 is not None),
            }
        )
    return pd.concat(label_frames, ignore_index=True), pd.DataFrame(audit_rows)


def write_task6_semantic_artifacts(
    dataset_path: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    output_name: str = "semantic_trace_task6",
    stable_frames: int = 3,
) -> dict[str, str]:
    """Write raw task6 trace, phase labels, audit rows, and metadata."""
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_path = meta_dir / f"{output_name}.parquet"
    labels_path = meta_dir / f"phase_progress_{output_name}.parquet"
    audit_path = meta_dir / f"{output_name}_audit.csv"
    metadata_path = meta_dir / f"{output_name}_metadata.json"
    trace = pd.DataFrame(records).sort_values(["episode_index", "frame_index"])
    labels, audit = build_task6_phase_labels(trace, stable_frames=stable_frames)
    trace.to_parquet(raw_path, index=False)
    labels.to_parquet(labels_path, index=False)
    audit.to_csv(audit_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return {
        "raw_trace": str(raw_path),
        "phase_labels": str(labels_path),
        "audit": str(audit_path),
        "metadata": str(metadata_path),
    }


def build_task7_phase_labels(trace: pd.DataFrame, *, stable_frames: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Task7 phases with a pre-containment final-transfer boundary."""
    required = {"episode_index", "frame_index", "is_success", "env_success", "object_a_controlled", "object_b_controlled", "object_a_basket_contact", "object_b_basket_contact", "object_a_near_basket", "object_b_near_basket"}
    missing = required - set(trace.columns)
    if missing:
        raise ValueError(f"Semantic trace missing columns: {sorted(missing)}")
    labels, audits = [], []
    for episode_index, episode in trace.groupby("episode_index", sort=True):
        episode = episode.sort_values("frame_index").reset_index(drop=True)
        success = bool(episode["is_success"].iloc[-1])
        controlled_a = episode["object_a_controlled"].to_numpy(bool)
        controlled_b = episode["object_b_controlled"].to_numpy(bool)
        contained_a = episode["object_a_basket_contact"].to_numpy(bool)
        contained_b = episode["object_b_basket_contact"].to_numpy(bool)
        near_a = episode["object_a_near_basket"].to_numpy(bool)
        near_b = episode["object_b_near_basket"].to_numpy(bool)
        env_success = episode["env_success"].to_numpy(bool)
        b1 = _first_stable_frame(controlled_a | controlled_b, stable_frames)
        b2_start = b1 + 1 if b1 is not None else 0
        b2, b2_source = _first_boundary({
            "object_a_in_basket": _first_stable_frame(contained_a, stable_frames, b2_start),
            "object_b_in_basket": _first_stable_frame(contained_b, stable_frames, b2_start),
            "object_a_final_transfer": _first_stable_frame(controlled_a & near_a, stable_frames, b2_start),
            "object_b_final_transfer": _first_stable_frame(controlled_b & near_b, stable_frames, b2_start),
        })
        final_goal = contained_a & contained_b
        b3 = _first_stable_frame(final_goal, stable_frames, b2 + 1 if b2 is not None else 0)
        b3_source = "state_stable" if b3 is not None else "unresolved"
        if b3 is None and success:
            b3 = _first_true_frame(env_success, b2 + 1 if b2 is not None else 0)
            b3_source = "env_success_terminal" if b3 is not None else "unresolved"
        if not success:
            b3, b3_source = None, "failed_episode"
        phase = np.zeros(len(episode), dtype=np.int64)
        if b1 is not None:
            phase[b1:] = 1
        if b2 is not None and b1 is not None and b2 > b1:
            phase[b2:] = 2
        else:
            b2, b2_source = None, "unresolved"
        if b3 is not None and b2 is not None and b3 > b2:
            phase[b3:] = 3
        else:
            b3 = None
            if success:
                b3_source = "unresolved"
        phase_progress, global_progress = _phase_progress(
            phase, is_success=success
        )
        labels.append(pd.DataFrame({"episode_index": int(episode_index), "frame_index": episode["frame_index"].to_numpy(np.int64), "phase": phase, "phase_progress": phase_progress, "global_progress": global_progress, "semantic_source": "simulator_trace", "semantic_confidence": "state_verified" if b1 is not None and (not success or b3 is not None) else "unresolved", "is_success": success}))
        audits.append({"episode_index": int(episode_index), "episode_length": len(episode), "is_success": success, "b1_frame": b1, "b2_frame": b2, "b2_source": b2_source, "b3_frame": b3, "b3_source": b3_source, "b2_joint_transfer_detected": bool(b2_source and "final_transfer" in b2_source), "b3_consistent_with_success": (b3 is not None) == success, "trainable": b1 is not None and (not success or b3 is not None)})
    return pd.concat(labels, ignore_index=True), pd.DataFrame(audits)


def write_task7_semantic_artifacts(dataset_path: str | Path, records: list[dict[str, Any]], metadata: dict[str, Any], *, output_name: str = "semantic_trace_task7", stable_frames: int = 5) -> dict[str, str]:
    """Write Task7 raw trace, labels, audit, and metadata."""
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.DataFrame(records).sort_values(["episode_index", "frame_index"])
    labels, audit = build_task7_phase_labels(trace, stable_frames=stable_frames)
    raw_path, labels_path = meta_dir / f"{output_name}.parquet", meta_dir / f"phase_progress_{output_name}.parquet"
    audit_path, metadata_path = meta_dir / f"{output_name}_audit.csv", meta_dir / f"{output_name}_metadata.json"
    trace.to_parquet(raw_path, index=False); labels.to_parquet(labels_path, index=False); audit.to_csv(audit_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as file: json.dump(metadata, file, indent=2, ensure_ascii=False)
    return {"raw_trace": str(raw_path), "phase_labels": str(labels_path), "audit": str(audit_path), "metadata": str(metadata_path)}


def build_task9_phase_labels(trace: pd.DataFrame, *, stable_frames: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"episode_index", "frame_index", "is_success", "env_success", "mug_controlled", "mug_in_heating_region", "microwave_is_close", "porcelain_mug_gripper_contact"}
    missing = required - set(trace.columns)
    if missing: raise ValueError(f"Semantic trace missing columns: {sorted(missing)}")
    labels, audits = [], []
    for index, episode in trace.groupby("episode_index", sort=True):
        episode = episode.sort_values("frame_index").reset_index(drop=True); success = bool(episode.is_success.iloc[-1])
        controlled, inside, closed, env_success = (episode[name].to_numpy(bool) for name in ("mug_controlled", "mug_in_heating_region", "microwave_is_close", "env_success"))
        b1 = _first_stable_frame(controlled, stable_frames); start = b1 + 1 if b1 is not None else 0
        b2 = _first_stable_frame(inside, stable_frames, start); b2_source = "mug_in_heating_region" if b2 is not None else "unresolved"
        b3 = _first_stable_frame(inside & closed, stable_frames, b2 + 1 if b2 is not None else 0); b3_source = "state_stable" if b3 is not None else "unresolved"
        if b3 is None and success: b3 = _first_true_frame(env_success, b2 + 1 if b2 is not None else 0); b3_source = "env_success_terminal" if b3 is not None else "unresolved"
        if not success: b3, b3_source = None, "failed_episode"
        phase = np.zeros(len(episode), dtype=np.int64)
        if b1 is not None: phase[b1:] = 1
        if b2 is not None and b1 is not None and b2 > b1: phase[b2:] = 2
        else: b2, b2_source = None, "unresolved"
        if b3 is not None and b2 is not None and b3 > b2: phase[b3:] = 3
        else: b3 = None; b3_source = "unresolved" if success else b3_source
        progress, global_progress = _phase_progress(phase, is_success=success)
        local_progress = np.zeros(len(episode), dtype=np.float32)
        distance = episode["mug_microwave_distance"].to_numpy(dtype=np.float32)
        if b1 is not None:
            reference_distance = float(np.median(distance[b1 : min(b1 + stable_frames, len(distance))]))
            target_distance = float(np.min(distance[b1:]))
            if reference_distance > target_distance:
                phase_one = phase == 1
                local_progress[phase_one] = np.clip((reference_distance - distance[phase_one]) / (reference_distance - target_distance), 0.0, 1.0)
        if b2 is not None:
            door_qpos = episode["microwave_joint_qpos"].to_numpy(dtype=np.float32)
            start_qpos = float(np.median(door_qpos[b2 : min(b2 + stable_frames, len(door_qpos))]))
            denominator = abs(start_qpos)
            phase_two = phase == 2
            if denominator > 1e-6:
                local_progress[phase_two] = np.clip(1.0 - np.abs(door_qpos[phase_two]) / denominator, 0.0, 1.0) * inside[phase_two]
        local_progress[phase == 3] = 1.0
        local_peak = pd.Series(local_progress).groupby(phase).cummax().to_numpy(dtype=np.float32)
        local_source = np.where(phase == 1, "mug_to_microwave_distance", np.where(phase == 2, "microwave_door_joint", np.where(phase == 3, "task_success", "phase_zero")))
        labels.append(pd.DataFrame({"episode_index": int(index), "frame_index": episode.frame_index.to_numpy(np.int64), "phase": phase, "phase_progress": progress, "global_progress": global_progress, "phase_local_progress": local_progress, "phase_local_progress_peak": local_peak, "phase_local_progress_source": local_source, "phase_local_progress_confidence": "state_verified", "semantic_source": "simulator_trace", "semantic_confidence": "state_verified" if b1 is not None and (not success or b3 is not None) else "unresolved", "is_success": success}))
        audits.append({"episode_index": int(index), "episode_length": len(episode), "is_success": success, "b1_frame": b1, "b2_frame": b2, "b2_source": b2_source, "b3_frame": b3, "b3_source": b3_source, "microwave_closed_before_mug_observed": bool((closed & ~inside).any()), "porcelain_mug_gripper_contact_observed": bool(episode.porcelain_mug_gripper_contact.to_numpy(bool).any()), "b3_consistent_with_success": (b3 is not None) == success, "trainable": b1 is not None and (not success or b3 is not None)})
    return pd.concat(labels, ignore_index=True), pd.DataFrame(audits)


def write_task9_semantic_artifacts(dataset_path: str | Path, records: list[dict[str, Any]], metadata: dict[str, Any], *, output_name: str = "semantic_trace_task9", stable_frames: int = 3) -> dict[str, str]:
    dataset_path = Path(dataset_path); meta_dir = dataset_path / "meta"; meta_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.DataFrame(records).sort_values(["episode_index", "frame_index"]); labels, audit = build_task9_phase_labels(trace, stable_frames=stable_frames)
    raw_path = meta_dir / f"{output_name}.parquet"; labels_path = meta_dir / f"phase_progress_{output_name}.parquet"; audit_path = meta_dir / f"{output_name}_audit.csv"; metadata_path = meta_dir / f"{output_name}_metadata.json"
    trace.to_parquet(raw_path, index=False); labels.to_parquet(labels_path, index=False); audit.to_csv(audit_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as file: json.dump(metadata, file, indent=2, ensure_ascii=False)
    return {"raw_trace": str(raw_path), "phase_labels": str(labels_path), "audit": str(audit_path), "metadata": str(metadata_path)}
