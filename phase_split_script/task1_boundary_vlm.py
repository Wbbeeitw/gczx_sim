"""Boundary-based VLM annotation for LIBERO-10 Task 1.

This script avoids whole-video phase segmentation. Instead, it predicts three
monotonic boundary events:

    B1: first clear robot-controlled grasp / pickup / drag of a target object
    B2: first late-stage milestone (one object stably in basket, or clear final
        joint-transfer setup at the basket)
    B3: task completion (both objects stably in basket)

The workflow is:
    1. Use the existing rule annotator to produce coarse boundary candidates.
    2. Refine each candidate locally with a small VLM window around it.
    3. Convert the final boundaries into phase labels:
           phase 0: [0, B1)
           phase 1: [B1, B2) or to end if B2 absent
           phase 2: [B2, B3) or to end if B3 absent
           phase 3: [B3, end]

This is faster and more stable than asking the VLM to segment the whole video
into a complete phase timeline in one shot.
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from tqdm import tqdm

try:
    from common_annotator import compute_progress_per_segment
    from qwen_client import call_qwen_vl, DEFAULT_MODEL
    from task1 import annotate_task1
except ModuleNotFoundError:
    from phase_split_script.common_annotator import compute_progress_per_segment
    from phase_split_script.qwen_client import call_qwen_vl, DEFAULT_MODEL
    from phase_split_script.task1 import annotate_task1


TASK_DESCRIPTION = "Put both the cream cheese box and the butter in the basket"
PROMPT_VERSION = "task1_boundary_v2_pickup_search"
NUM_PHASES = 4
SUCCESS_PHASE = 3
DEFAULT_VIDEO_FPS = 10.0
MAX_WINDOW_IMAGES = 9
MAX_REFINE_ATTEMPTS = 3

_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class BoundarySpec:
    key: str
    phase_id: int
    window_seconds: float
    description: str


BOUNDARY_SPECS: tuple[BoundarySpec, ...] = (
    BoundarySpec(
        key="b1",
        phase_id=1,
        window_seconds=4.0,
        description=(
            "Earliest moment after reset-settling when the robot first clearly "
            "grasps, lifts, or deliberately drags at least one target object under "
            "robot control. Passive drop, bounce, gravity-driven slide, or a brief "
            "incidental touch at the start does NOT count."
        ),
    ),
    BoundarySpec(
        key="b2",
        phase_id=2,
        window_seconds=5.0,
        description=(
            "Earliest late-stage milestone: one target object is already stably "
            "inside the basket, OR both objects are under a clear final "
            "joint-transfer setup at the basket, but the task is not complete."
        ),
    ),
    BoundarySpec(
        key="b3",
        phase_id=3,
        window_seconds=5.0,
        description=(
            "Earliest moment when BOTH target objects are stably inside the basket "
            "and the task is complete."
        ),
    ),
)


PHASE_DEFINITIONS: dict[int, str] = {
    0: "reset-settling or pre-manipulation",
    1: "first robot-controlled pickup / drag before any stable in-basket result",
    2: "late unfinished stage after one in-basket result or clear final setup",
    3: "both objects stably inside the basket / completion",
}


class BoundaryDecision(BaseModel):
    """Local VLM decision for one boundary window."""

    status: Literal["before", "within", "after", "absent"] = Field(
        description="Relative location of the boundary with respect to the shown window."
    )
    frame_index: int | None = Field(
        default=None,
        description="Chosen frame index when status == 'within'.",
    )
    reasoning: str = Field(default="", description="Short explanation.")


@dataclass
class BoundaryRefinement:
    key: str
    coarse_frame: int | None
    final_frame: int | None
    status: str
    source: str
    attempts: list[dict[str, Any]]


@dataclass
class EpisodeBoundaryResult:
    episode_index: int
    df: pd.DataFrame | None
    source: str
    boundaries: dict[str, int | None] | None = None
    error: str | None = None


def _load_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cache[int(entry["episode_index"])] = entry
    return cache


def _append_cache(cache_path: Path, entry: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_collection_summary(dataset_path: Path) -> dict[str, Any]:
    summary_path = dataset_path / "collection_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _parse_collection_summary_success(summary: dict[str, Any]) -> dict[int, bool] | None:
    if not isinstance(summary, dict):
        return None

    if "episodes" in summary and isinstance(summary["episodes"], list):
        result: dict[int, bool] = {}
        for entry in summary["episodes"]:
            if isinstance(entry, dict) and "episode_index" in entry:
                result[int(entry["episode_index"])] = bool(entry.get("is_success", False))
        return result if result else None

    if "is_success" in summary and isinstance(summary["is_success"], list):
        return {i: bool(v) for i, v in enumerate(summary["is_success"])}

    result = {}
    for key, value in summary.items():
        if isinstance(key, str) and key.isdigit() and isinstance(value, dict):
            result[int(key)] = bool(value.get("is_success", False))
    return result if result else None


def _infer_episode_success(
    dataset_path: Path,
    episode_index: int,
    ep_file: Path | None = None,
) -> bool | None:
    summary = _load_collection_summary(dataset_path)
    parsed = _parse_collection_summary_success(summary)
    if parsed is not None and episode_index in parsed:
        return parsed[episode_index]

    if ep_file is not None and ep_file.exists():
        try:
            df = pd.read_parquet(ep_file)
            if "is_success" in df.columns:
                return bool(df["is_success"].iloc[-1])
            if "reward" in df.columns:
                return bool(df["reward"].iloc[-1] > 0.0)
            if "next.done" in df.columns:
                return bool(df["next.done"].iloc[-1])
        except Exception:
            pass

    return None


def _infer_video_path(
    dataset_path: Path,
    episode_index: int,
    video_key: str = "image",
) -> Path | None:
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        video_path_pattern = info.get("video_path")
        if video_path_pattern:
            formatted = (
                video_path_pattern
                .replace("{episode_chunk:03d}", f"{episode_index // 1000:03d}")
                .replace("{episode_index:06d}", f"{episode_index:06d}")
                .replace("{video_key}", video_key)
            )
            candidate = dataset_path / formatted
            if candidate.exists():
                return candidate

    chunk = episode_index // 1000
    candidate = (
        dataset_path
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )
    if candidate.exists():
        return candidate

    videos_root = dataset_path / "videos"
    if not videos_root.exists():
        return None

    patterns = [
        f"episode_{episode_index:06d}.mp4",
        f"episode_{episode_index:03d}.mp4",
        f"episode_{episode_index}.mp4",
        f"*{episode_index}*.mp4",
    ]
    for pattern in patterns:
        for match in videos_root.rglob(pattern):
            return match
    return None


def _load_video(video_path: Path) -> tuple[list[Image.Image], float]:
    try:
        import av
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyAV is required to decode rollout videos. Install the embodied "
            "dependencies or run inside the dataset annotation environment."
        ) from e

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else DEFAULT_VIDEO_FPS
    frames: list[Image.Image] = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(Image.fromarray(arr))
    container.close()
    return frames, fps


def _resize_to_height(img: Image.Image, target_height: int) -> Image.Image:
    if img.height == target_height:
        return img
    target_width = max(1, int(round(img.width * target_height / img.height)))
    return img.resize((target_width, target_height), Image.Resampling.BILINEAR)


def _compose_multiview_frame(
    main_frame: Image.Image,
    wrist_frame: Image.Image | None,
    frame_index: int,
    timestamp_seconds: float,
) -> Image.Image:
    main_rgb = main_frame.convert("RGB")
    header_h = 28
    gap = 6
    target_h = main_rgb.height

    views: list[tuple[str, Image.Image]] = [("main", main_rgb)]
    if wrist_frame is not None:
        views.append(("wrist", _resize_to_height(wrist_frame.convert("RGB"), target_h)))

    body_w = sum(img.width for _, img in views) + gap * (len(views) - 1)
    canvas = Image.new("RGB", (body_w, target_h + header_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, body_w, header_h), fill=(0, 0, 0))
    draw.text(
        (8, 7),
        f"frame={frame_index:04d}  t={timestamp_seconds:05.1f}s",
        fill=(255, 255, 255),
    )

    x = 0
    for label, img in views:
        canvas.paste(img, (x, header_h))
        draw.rectangle((x, header_h, x + img.width - 1, header_h + 17), fill=(0, 0, 0))
        draw.text((x + 6, header_h + 3), label, fill=(255, 255, 255))
        x += img.width + gap

    return canvas


def _build_sampled_images(
    main_frames: list[Image.Image],
    wrist_frames: list[Image.Image] | None,
    sampled_indices: list[int],
    fps: float,
) -> list[Image.Image]:
    sampled_images: list[Image.Image] = []
    for frame_index in sampled_indices:
        wrist_frame = None
        if wrist_frames is not None and frame_index < len(wrist_frames):
            wrist_frame = wrist_frames[frame_index]
        sampled_images.append(
            _compose_multiview_frame(
                main_frame=main_frames[frame_index],
                wrist_frame=wrist_frame,
                frame_index=frame_index,
                timestamp_seconds=frame_index / max(fps, 1e-6),
            )
        )
    return sampled_images


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stack = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue
    raise ValueError(f"Could not extract JSON from response: {text!r}")


def _parse_boundary_decision(text: str) -> dict[str, Any]:
    parsed = _extract_json(text)
    return BoundaryDecision.model_validate(parsed).model_dump()


def _first_frame_at_or_above(phase: np.ndarray, threshold: int) -> int | None:
    indices = np.where(phase >= threshold)[0]
    if len(indices) == 0:
        return None
    return int(indices[0])


def _coarse_boundaries_from_legacy(
    legacy_phase: np.ndarray,
    is_success: bool | None,
) -> dict[str, int | None]:
    b1 = _first_frame_at_or_above(legacy_phase, 1)
    b2 = _first_frame_at_or_above(legacy_phase, 2)
    b3 = _first_frame_at_or_above(legacy_phase, 5) if is_success is not False else None

    if b2 is None and is_success is True:
        b2 = _first_frame_at_or_above(legacy_phase, 4)
        if b2 is None:
            b2 = _first_frame_at_or_above(legacy_phase, 5)

    return {"b1": b1, "b2": b2, "b3": b3}


def _default_b1_fallback_frame(episode_length: int) -> int | None:
    if episode_length <= 0:
        return None
    if episode_length == 1:
        return 0
    return min(episode_length - 1, max(1, int(round(episode_length * 0.02))))


def _initial_center_for_boundary(
    spec: BoundarySpec,
    coarse_frame: int | None,
    episode_length: int,
    fps: float,
    is_success: bool | None,
) -> int | None:
    if episode_length <= 0:
        return None

    if coarse_frame is not None:
        coarse_frame = int(max(0, min(coarse_frame, episode_length - 1)))
        if spec.key == "b1":
            early_bias = max(1, int(round(1.0 * fps)))
            return max(0, coarse_frame - early_bias)
        return coarse_frame

    if spec.key == "b1":
        ratio = 0.18
    elif spec.key == "b2":
        ratio = 0.58 if is_success is True else 0.68
    elif spec.key == "b3":
        if is_success is False:
            return None
        ratio = 0.88
    else:
        return None

    return int(round((episode_length - 1) * ratio))


def _search_shift_frames(
    spec: BoundarySpec,
    fps: float,
    has_coarse: bool,
) -> int:
    base = max(1, int(round(spec.window_seconds * fps / 3.0)))
    if has_coarse:
        return base
    return max(base, int(round(spec.window_seconds * fps * 0.75)))


def _window_sample_indices(
    center_frame: int,
    episode_length: int,
    fps: float,
    window_seconds: float,
    num_samples: int = MAX_WINDOW_IMAGES,
) -> list[int]:
    if episode_length <= 0:
        return []
    half_window = max(1, int(round(window_seconds * fps / 2.0)))
    start = max(0, center_frame - half_window)
    end = min(episode_length - 1, center_frame + half_window)
    if end <= start:
        return [start]
    return np.linspace(start, end, num=num_samples, dtype=int).tolist()


def _build_boundary_prompt(
    spec: BoundarySpec,
    sampled_indices: list[int],
    fps: float,
    is_success: bool | None,
    enable_reasoning: bool,
) -> str:
    think_prefix = "/think\n" if enable_reasoning else "/no_think\n"
    if is_success is True:
        outcome_hint = "Environment outcome hint: this episode SUCCEEDED."
    elif is_success is False:
        outcome_hint = "Environment outcome hint: this episode FAILED."
    else:
        outcome_hint = "Environment outcome hint: unknown."

    frame_list = ", ".join(str(idx) for idx in sampled_indices)
    return (
        f"{think_prefix}You are verifying one temporal boundary in a robot manipulation video.\n\n"
        f'Task: "{TASK_DESCRIPTION}"\n'
        f"{outcome_hint}\n\n"
        f"Boundary to locate: {spec.key.upper()}\n"
        f"Boundary meaning: {spec.description}\n\n"
        "You will see a short chronological window of frames around a candidate "
        "boundary location. Each image header already shows the exact frame index "
        "and timestamp.\n\n"
        "Important rules:\n"
        "  - Judge robot-caused task progress, not passive environment reset motion.\n"
        "  - For B1 specifically, initial object drop / bounce / gravity slide at "
        "reset is NEVER enough.\n"
        "  - For B1, trigger as soon as one target object is clearly grasped, "
        "lifted, carried, or deliberately dragged under robot control; do NOT wait "
        "for a long stable hold.\n"
        "  - A brief accidental tap or uncontrolled bump is NOT enough for B1.\n"
        "  - Use 'within' only when one of the shown frames is the earliest frame where the boundary becomes true.\n"
        "  - Use 'before' if the boundary already happened before the first shown frame.\n"
        "  - Use 'after' if the boundary has still NOT happened by the last shown frame but likely occurs later.\n"
        "  - Use 'absent' only if you are confident this boundary never happens in the episode.\n\n"
        f"Shown frame indices: [{frame_list}]\n\n"
        "Output ONLY JSON in this exact format:\n"
        "{\n"
        '  "status": "before" | "within" | "after" | "absent",\n'
        '  "frame_index": 123 or null,\n'
        '  "reasoning": "short explanation"\n'
        "}\n\n"
        "If status == 'within', frame_index must be one of the shown frame indices."
    )


def _query_boundary_window(
    spec: BoundarySpec,
    sampled_images: list[Image.Image],
    sampled_indices: list[int],
    fps: float,
    model: str,
    is_success: bool | None,
    enable_reasoning: bool,
) -> BoundaryDecision:
    prompt = _build_boundary_prompt(
        spec=spec,
        sampled_indices=sampled_indices,
        fps=fps,
        is_success=is_success,
        enable_reasoning=enable_reasoning,
    )
    parsed = call_qwen_vl(
        sampled_images,
        prompt,
        model=model,
        enable_reasoning=enable_reasoning,
        response_parser=_parse_boundary_decision,
        max_tokens=512,
    )
    decision = BoundaryDecision.model_validate(parsed)
    if decision.status == "within":
        if decision.frame_index is None:
            raise ValueError("Boundary response marked 'within' but frame_index is null.")
        if decision.frame_index not in sampled_indices:
            raise ValueError(
                f"Boundary frame_index {decision.frame_index} is not one of shown indices "
                f"{sampled_indices}"
            )
    return decision


def _refine_boundary(
    spec: BoundarySpec,
    coarse_frame: int | None,
    main_frames: list[Image.Image],
    wrist_frames: list[Image.Image] | None,
    fps: float,
    episode_length: int,
    model: str,
    is_success: bool | None,
    enable_reasoning: bool,
) -> BoundaryRefinement:
    center = _initial_center_for_boundary(
        spec=spec,
        coarse_frame=coarse_frame,
        episode_length=episode_length,
        fps=fps,
        is_success=is_success,
    )
    if center is None:
        return BoundaryRefinement(
            key=spec.key,
            coarse_frame=coarse_frame,
            final_frame=None,
            status="absent",
            source="coarse_absent",
            attempts=[],
        )

    seeded_search = coarse_frame is None
    attempts: list[dict[str, Any]] = []
    for _ in range(MAX_REFINE_ATTEMPTS):
        sampled_indices = _window_sample_indices(
            center_frame=center,
            episode_length=episode_length,
            fps=fps,
            window_seconds=spec.window_seconds,
            num_samples=MAX_WINDOW_IMAGES,
        )
        sampled_images = _build_sampled_images(main_frames, wrist_frames, sampled_indices, fps)
        decision = _query_boundary_window(
            spec=spec,
            sampled_images=sampled_images,
            sampled_indices=sampled_indices,
            fps=fps,
            model=model,
            is_success=is_success,
            enable_reasoning=enable_reasoning,
        )
        attempts.append({
            "sampled_indices": sampled_indices,
            "decision": decision.model_dump(),
        })

        if decision.status == "within":
            return BoundaryRefinement(
                key=spec.key,
                coarse_frame=coarse_frame,
                final_frame=decision.frame_index,
                status="within",
                source="vlm_seed_search" if seeded_search else "vlm_refined",
                attempts=attempts,
            )
        if decision.status == "absent":
            return BoundaryRefinement(
                key=spec.key,
                coarse_frame=coarse_frame,
                final_frame=None,
                status="absent",
                source="vlm_absent_after_search" if seeded_search else "vlm_absent",
                attempts=attempts,
            )
        shift = _search_shift_frames(spec, fps, has_coarse=not seeded_search)
        if decision.status == "before":
            center = max(0, sampled_indices[0] - shift)
        else:
            center = min(episode_length - 1, sampled_indices[-1] + shift)

    return BoundaryRefinement(
        key=spec.key,
        coarse_frame=coarse_frame,
        final_frame=coarse_frame if coarse_frame is not None else None,
        status="fallback" if coarse_frame is not None else "search_exhausted",
        source="coarse_fallback" if coarse_frame is not None else "vlm_search_exhausted",
        attempts=attempts,
    )


def _finalize_boundaries(
    coarse: dict[str, int | None],
    refinements: dict[str, BoundaryRefinement],
    is_success: bool | None,
    episode_length: int,
) -> dict[str, int | None]:
    final: dict[str, int | None] = {}
    for spec in BOUNDARY_SPECS:
        refinement = refinements.get(spec.key)
        if refinement is not None and refinement.final_frame is not None:
            final[spec.key] = refinement.final_frame
        elif refinement is not None and refinement.status == "absent":
            final[spec.key] = None
        else:
            final[spec.key] = coarse.get(spec.key)

    if is_success is False:
        final["b3"] = None
    elif is_success is True and final["b3"] is None and episode_length > 0:
        final["b3"] = episode_length - 1

    if final["b1"] is None and (final["b2"] is not None or final["b3"] is not None):
        fallback_b1 = coarse.get("b1")
        if fallback_b1 is None:
            fallback_b1 = _default_b1_fallback_frame(episode_length)
        final["b1"] = fallback_b1

    if final["b2"] is not None and final["b1"] is not None and final["b2"] <= final["b1"]:
        final["b2"] = None

    if final["b3"] is not None:
        prev = final["b2"] if final["b2"] is not None else final["b1"]
        if prev is None:
            final["b1"] = coarse.get("b1", 0) or 0
            prev = final["b1"]
        if prev is not None and final["b3"] <= prev:
            if prev < episode_length - 1:
                final["b3"] = prev + 1
            else:
                final["b3"] = prev
        if final["b2"] is not None and final["b2"] >= final["b3"]:
            final["b2"] = None

    for key, value in list(final.items()):
        if value is not None:
            final[key] = int(max(0, min(value, episode_length - 1)))

    return final


def _sync_refinements_with_final(
    coarse: dict[str, int | None],
    refinements: dict[str, BoundaryRefinement],
    final: dict[str, int | None],
    is_success: bool | None,
    episode_length: int,
) -> None:
    b1_ref = refinements.get("b1")
    if final.get("b1") is not None and (b1_ref is None or b1_ref.final_frame is None):
        source = "linked_fallback"
        if coarse.get("b1") is None:
            source = "default_b1_fallback"
        refinements["b1"] = BoundaryRefinement(
            key="b1",
            coarse_frame=coarse.get("b1"),
            final_frame=final["b1"],
            status=source,
            source=source,
            attempts=[] if b1_ref is None else b1_ref.attempts,
        )

    b3_ref = refinements.get("b3")
    if (
        is_success is True
        and episode_length > 0
        and final.get("b3") == episode_length - 1
        and (b3_ref is None or b3_ref.final_frame is None)
    ):
        refinements["b3"] = BoundaryRefinement(
            key="b3",
            coarse_frame=coarse.get("b3"),
            final_frame=final["b3"],
            status="success_tail_fallback",
            source="success_tail_fallback",
            attempts=[] if b3_ref is None else b3_ref.attempts,
        )


def boundaries_to_phase_array(
    episode_length: int,
    boundaries: dict[str, int | None],
) -> np.ndarray:
    phase = np.zeros(episode_length, dtype=np.int64)
    if episode_length <= 0:
        return phase

    b1 = boundaries.get("b1")
    b2 = boundaries.get("b2")
    b3 = boundaries.get("b3")
    if b1 is not None:
        phase[b1:] = 1
    if b2 is not None:
        phase[b2:] = 2
    if b3 is not None:
        phase[b3:] = 3
    return phase


def build_phase_df_from_boundaries(
    episode_index: int,
    episode_length: int,
    boundaries: dict[str, int | None],
    is_success: bool | None,
) -> pd.DataFrame:
    phase = boundaries_to_phase_array(episode_length, boundaries)
    phase_progress, global_progress, overall_progress = compute_progress_per_segment(
        phase, NUM_PHASES
    )

    if len(phase) > 0:
        terminal_incomplete = is_success is False or (
            is_success is not True and phase[-1] != SUCCESS_PHASE
        )
        if terminal_incomplete:
            segment_start = len(phase) - 1
            while segment_start > 0 and phase[segment_start - 1] == phase[-1]:
                segment_start -= 1
            phase_progress[segment_start:] = 0.0
            global_progress[segment_start:] = phase[-1] / NUM_PHASES

    out = pd.DataFrame({
        "episode_index": np.full(episode_length, episode_index, dtype=np.int64),
        "frame_index": np.arange(episode_length, dtype=np.int64),
        "phase": phase.astype(np.int64),
        "phase_progress": phase_progress,
        "global_progress": global_progress,
        "progress": overall_progress,
    })
    if is_success is not None:
        out["is_success"] = pd.array([is_success] * episode_length, dtype="boolean")
    return out


def _boundary_summary_row(
    episode_index: int,
    episode_length: int,
    is_success: bool | None,
    coarse: dict[str, int | None],
    refinements: dict[str, BoundaryRefinement],
    final: dict[str, int | None],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_length": episode_length,
        "is_success": is_success,
    }
    for spec in BOUNDARY_SPECS:
        refinement = refinements.get(spec.key)
        row[f"{spec.key}_coarse"] = coarse.get(spec.key)
        row[f"{spec.key}_final"] = final.get(spec.key)
        row[f"{spec.key}_status"] = refinement.status if refinement else None
        row[f"{spec.key}_source"] = refinement.source if refinement else "coarse_only"
    return row


def _legacy_phase_for_episode(ep_file: Path) -> np.ndarray:
    df = pd.read_parquet(ep_file)
    actions = np.stack(df["actions"].values)
    state = (
        np.stack(df["state"].values)
        if "state" in df.columns
        else np.empty((len(df), 0))
    )
    phase, _ = annotate_task1(actions, state, len(df))
    return phase.astype(np.int64)


def _annotate_episode(
    dataset_path: Path,
    ep_file: Path,
    model: str,
    video_key: str,
    wrist_video_key: str | None,
    cache_path: Path,
    cache: dict[int, dict[str, Any]],
    enable_reasoning: bool,
    coarse_only: bool,
) -> EpisodeBoundaryResult:
    df = pd.read_parquet(ep_file)
    if len(df) == 0:
        return EpisodeBoundaryResult(
            episode_index=-1, df=None, source="boundary_vlm", error="empty episode"
        )

    episode_index = int(df["episode_index"].iloc[0])
    episode_length = len(df)
    is_success = _infer_episode_success(dataset_path, episode_index, ep_file)

    cached = cache.get(episode_index)
    if cached is not None:
        if (
            cached.get("model") != model
            or cached.get("video_key") != video_key
            or cached.get("wrist_video_key") != wrist_video_key
            or cached.get("enable_reasoning") != enable_reasoning
            or cached.get("coarse_only") != coarse_only
            or cached.get("prompt_version") != PROMPT_VERSION
        ):
            cached = None
    if cached is not None:
        boundaries = cached["final_boundaries"]
        out_df = build_phase_df_from_boundaries(
            episode_index=episode_index,
            episode_length=episode_length,
            boundaries=boundaries,
            is_success=is_success,
        )
        return EpisodeBoundaryResult(
            episode_index=episode_index,
            df=out_df,
            source="boundary_vlm",
            boundaries=boundaries,
        )

    legacy_phase = _legacy_phase_for_episode(ep_file)
    coarse = _coarse_boundaries_from_legacy(legacy_phase, is_success)

    refinements: dict[str, BoundaryRefinement] = {}
    if coarse_only:
        for spec in BOUNDARY_SPECS:
            refinements[spec.key] = BoundaryRefinement(
                key=spec.key,
                coarse_frame=coarse.get(spec.key),
                final_frame=coarse.get(spec.key),
                status="coarse_only",
                source="coarse_only",
                attempts=[],
            )
    else:
        video_path = _infer_video_path(dataset_path, episode_index, video_key=video_key)
        if video_path is None:
            return EpisodeBoundaryResult(
                episode_index=episode_index,
                df=None,
                source="boundary_vlm",
                error="video not found",
            )
        main_frames, fps = _load_video(video_path)
        wrist_frames: list[Image.Image] | None = None
        if wrist_video_key:
            wrist_video_path = _infer_video_path(
                dataset_path, episode_index, video_key=wrist_video_key
            )
            if wrist_video_path is not None:
                wrist_frames, _ = _load_video(wrist_video_path)

        for spec in BOUNDARY_SPECS:
            if spec.key == "b3" and is_success is False:
                refinements[spec.key] = BoundaryRefinement(
                    key=spec.key,
                    coarse_frame=None,
                    final_frame=None,
                    status="absent",
                    source="failed_episode",
                    attempts=[],
                )
                continue
            try:
                refinements[spec.key] = _refine_boundary(
                    spec=spec,
                    coarse_frame=coarse.get(spec.key),
                    main_frames=main_frames,
                    wrist_frames=wrist_frames,
                    fps=fps,
                    episode_length=episode_length,
                    model=model,
                    is_success=is_success,
                    enable_reasoning=enable_reasoning,
                )
            except Exception as e:
                refinements[spec.key] = BoundaryRefinement(
                    key=spec.key,
                    coarse_frame=coarse.get(spec.key),
                    final_frame=coarse.get(spec.key),
                    status="fallback",
                    source=f"coarse_after_error:{e}",
                    attempts=[],
                )

    final_boundaries = _finalize_boundaries(
        coarse=coarse,
        refinements=refinements,
        is_success=is_success,
        episode_length=episode_length,
    )
    _sync_refinements_with_final(
        coarse=coarse,
        refinements=refinements,
        final=final_boundaries,
        is_success=is_success,
        episode_length=episode_length,
    )
    out_df = build_phase_df_from_boundaries(
        episode_index=episode_index,
        episode_length=episode_length,
        boundaries=final_boundaries,
        is_success=is_success,
    )

    entry = {
        "episode_index": episode_index,
        "episode_length": episode_length,
        "task_description": TASK_DESCRIPTION,
        "model": model,
        "video_key": video_key,
        "wrist_video_key": wrist_video_key,
        "enable_reasoning": enable_reasoning,
        "coarse_only": coarse_only,
        "prompt_version": PROMPT_VERSION,
        "is_success": is_success,
        "coarse_boundaries": coarse,
        "final_boundaries": final_boundaries,
        "refinements": {
            key: {
                "coarse_frame": value.coarse_frame,
                "final_frame": value.final_frame,
                "status": value.status,
                "source": value.source,
                "attempts": value.attempts,
            }
            for key, value in refinements.items()
        },
    }
    _append_cache(cache_path, entry)
    with _CACHE_LOCK:
        cache[episode_index] = entry

    return EpisodeBoundaryResult(
        episode_index=episode_index,
        df=out_df,
        source="boundary_vlm",
        boundaries=final_boundaries,
    )


def _cache_to_summary_table(cache: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_index, entry in sorted(cache.items()):
        refinements_raw = entry.get("refinements", {})
        refinements = {
            key: BoundaryRefinement(
                key=key,
                coarse_frame=value.get("coarse_frame"),
                final_frame=value.get("final_frame"),
                status=value.get("status", ""),
                source=value.get("source", ""),
                attempts=value.get("attempts", []),
            )
            for key, value in refinements_raw.items()
        }
        rows.append(
            _boundary_summary_row(
                episode_index=episode_index,
                episode_length=int(entry.get("episode_length", 0)),
                is_success=entry.get("is_success"),
                coarse=entry.get("coarse_boundaries", {}),
                refinements=refinements,
                final=entry.get("final_boundaries", {}),
            )
        )
    return pd.DataFrame(rows)


def annotate_dataset_with_boundary_vlm(
    dataset_path: str | Path,
    output_name: str = "phase_progress_semantic_boundary_vlm",
    model: str = DEFAULT_MODEL,
    video_key: str = "image",
    wrist_video_key: str | None = "wrist_image",
    enable_reasoning: bool = False,
    coarse_only: bool = False,
) -> tuple[Path, Path]:
    dataset_path = Path(dataset_path)
    data_root = dataset_path / "data"
    ep_files = sorted(data_root.rglob("episode_*.parquet"))
    print(f"Found {len(ep_files)} episodes in {data_root}")

    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cache_path = meta_dir / f"{output_name}_boundaries.jsonl"
    summary_path = meta_dir / f"{output_name}_boundaries.csv"
    cache = _load_cache(cache_path)
    print(f"Loaded {len(cache)} cached boundary responses from {cache_path}")

    records: list[pd.DataFrame] = []
    errors: list[tuple[int, str]] = []
    for ep_file in tqdm(ep_files, desc="Boundary annotation"):
        result = _annotate_episode(
            dataset_path=dataset_path,
            ep_file=ep_file,
            model=model,
            video_key=video_key,
            wrist_video_key=wrist_video_key,
            cache_path=cache_path,
            cache=cache,
            enable_reasoning=enable_reasoning,
            coarse_only=coarse_only,
        )
        if result.df is not None:
            records.append(result.df)
        elif result.error is not None:
            errors.append((result.episode_index, result.error))

    if not records:
        raise RuntimeError("No boundary annotations were produced.")

    out_df = pd.concat(records, ignore_index=True)
    out_df = out_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    out_path = meta_dir / f"{output_name}.parquet"
    out_df.to_parquet(out_path, index=False)

    summary_df = _cache_to_summary_table(cache)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved {len(out_df)} frame labels to {out_path}")
    print(f"Saved boundary summary to {summary_path}")
    print("Phase distribution:")
    print(out_df["phase"].value_counts().sort_index())
    if errors:
        print("Errors:")
        for episode_index, error in errors:
            print(f"  episode {episode_index}: {error}")

    return out_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boundary-based VLM annotation for LIBERO-10 Task 1."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to LeRobot dataset root.",
    )
    parser.add_argument(
        "--output_name",
        default="phase_progress_semantic_boundary_vlm",
        help="Output parquet / boundary summary prefix under dataset/meta/.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="DashScope or local vLLM model name.",
    )
    parser.add_argument(
        "--video_key",
        default="image",
        help="Main video key.",
    )
    parser.add_argument(
        "--wrist_video_key",
        default="wrist_image",
        help="Optional wrist video key; set empty string to disable.",
    )
    parser.add_argument(
        "--enable_reasoning",
        default="false",
        choices=("true", "false"),
        help="Whether to enable model reasoning/thinking.",
    )
    parser.add_argument(
        "--coarse_only",
        action="store_true",
        help="Skip VLM refinement and emit pure rule-based boundary proposals.",
    )
    args = parser.parse_args()

    wrist_video_key = args.wrist_video_key or None
    out_path, summary_path = annotate_dataset_with_boundary_vlm(
        dataset_path=args.dataset_path,
        output_name=args.output_name,
        model=args.model,
        video_key=args.video_key,
        wrist_video_key=wrist_video_key,
        enable_reasoning=args.enable_reasoning == "true",
        coarse_only=args.coarse_only,
    )
    print(f"Done: {out_path}")
    print(f"Boundary summary: {summary_path}")


if __name__ == "__main__":
    main()
