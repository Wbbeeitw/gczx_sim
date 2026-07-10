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
TASK1_SUCCESS_PHASE = 3


@dataclass(frozen=True)
class Task1SemanticTraceConfig:
    """Task1 simulator-state extraction settings."""

    object_a_alias: str = "cream_cheese"
    object_b_alias: str = "butter"
    basket_alias: str = "basket"
    gripper_width_threshold: float = 0.035
    controlled_motion_threshold: float = 0.001
    basket_near_threshold: float = 0.12
    stable_frames: int = 5


@dataclass(frozen=True)
class Task1SemanticBodies:
    """Resolved MuJoCo body names used by a task1 trace."""

    object_a: str
    object_b: str
    basket: str


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


def _phase_progress(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase_progress = np.zeros(len(phase), dtype=np.float32)
    for phase_id in np.unique(phase):
        indices = np.flatnonzero(phase == phase_id)
        if len(indices) > 1:
            phase_progress[indices] = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
    global_progress = (phase.astype(np.float32) + phase_progress) / NUM_TASK1_PHASES
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
        in_basket_a = (
            episode_trace["object_a_basket_contact"].to_numpy(dtype=bool)
            & episode_trace["object_a_near_basket"].to_numpy(dtype=bool)
        )
        in_basket_b = (
            episode_trace["object_b_basket_contact"].to_numpy(dtype=bool)
            & episode_trace["object_b_near_basket"].to_numpy(dtype=bool)
        )
        b1 = _first_stable_frame(controlled_a | controlled_b, stable_frames)
        one_in_basket = in_basket_a | in_basket_b
        joint_transfer = controlled_a & controlled_b & (
            episode_trace["object_a_near_basket"].to_numpy(dtype=bool)
        ) & (episode_trace["object_b_near_basket"].to_numpy(dtype=bool))
        b2 = _first_stable_frame(one_in_basket | joint_transfer, stable_frames, b1 or 0)
        b3 = _first_stable_frame(in_basket_a & in_basket_b, stable_frames, b2 or 0)
        if not is_success:
            b3 = None

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

        phase_progress, global_progress = _phase_progress(phase)
        terminal_incomplete = not is_success or b3 is None
        if terminal_incomplete and length:
            terminal_phase = int(phase[-1])
            terminal_start = int(np.flatnonzero(phase == terminal_phase)[-1])
            while terminal_start > 0 and phase[terminal_start - 1] == terminal_phase:
                terminal_start -= 1
            phase_progress[terminal_start:] = 0.0
            global_progress[terminal_start:] = terminal_phase / NUM_TASK1_PHASES

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
