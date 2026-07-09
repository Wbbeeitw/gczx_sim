"""Whole-video VLM phase annotation for LIBERO-10 Task 1.

Unlike the frame-by-frame ``qwen_annotator.py``, this script sends a sampled
sequence of frames from an entire episode to a Qwen-VL model served via vLLM
(or DashScope) and asks for a contiguous phase timeline.  It explicitly
supports failed episodes and phase regressions/repetitions (e.g. the robot
drops an object and retries the same phase).

The output is a standard ``meta/<output_name>.parquet`` sidecar compatible
with ``RevaluePhaseDataset``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm

from common_annotator import compute_progress_per_segment
from qwen_client import call_qwen_vl, DEFAULT_MODEL
from task1 import annotate_task1


# --------------------------------------------------------------------------- #
# Task-specific definitions
# --------------------------------------------------------------------------- #
TASK_DESCRIPTION = "Put both the cream cheese box and the butter in the basket"

NUM_PHASES = 6

PHASE_DEFINITIONS: dict[int, str] = {
    0: "approaching and grasping the first object (cream cheese box)",
    1: "transporting the first object to the basket",
    2: "placing the first object into the basket",
    3: "approaching and grasping the second object (butter)",
    4: "transporting the second object to the basket",
    5: "placing the second object into the basket / task completion",
}

PHASE_NAME_TO_ID = {name.lower(): phase_id for phase_id, name in PHASE_DEFINITIONS.items()}

# LeRobot LIBERO videos are recorded at 10 fps.
DEFAULT_VIDEO_FPS = 10.0


# --------------------------------------------------------------------------- #
# Pydantic models for VLM output
# --------------------------------------------------------------------------- #
class PhaseSegment(BaseModel):
    """One contiguous phase segment returned by the VLM."""

    name: str = Field(description="Phase name from the closed vocabulary.")
    start_seconds: float = Field(description="Segment start time in seconds.")
    end_seconds: float = Field(description="Segment end time in seconds.")


class EpisodeAnnotation(BaseModel):
    """Structured VLM annotation for one episode."""

    reasoning: str = Field(default="", description="Brief visual reasoning.")
    success: bool | None = Field(
        default=None,
        description="Whether the model thinks the episode succeeded.",
    )
    segments: list[PhaseSegment] = Field(
        description="Contiguous phase segments covering the whole video."
    )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
_CACHE_LOCK = threading.Lock()


def _load_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    """Load per-episode VLM cache keyed by episode_index."""
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
    """Append a single cache entry to the JSONL cache (thread-safe)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Episode outcome helpers
# --------------------------------------------------------------------------- #
def _load_collection_summary(dataset_path: Path) -> dict[str, Any]:
    """Load collection_summary.json if it exists."""
    summary_path = dataset_path / "collection_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _parse_collection_summary_success(summary: dict[str, Any]) -> dict[int, bool] | None:
    """Extract per-episode success flags from a collection summary.

    Tries several common layouts:
      - {"episodes": [{"episode_index": 0, "is_success": true}, ...]}
      - {"is_success": [true, false, ...]}
      - {"0": {"is_success": true}, ...}
    """
    if not isinstance(summary, dict):
        return None

    if "episodes" in summary and isinstance(summary["episodes"], list):
        result: dict[int, bool] = {}
        for entry in summary["episodes"]:
            if isinstance(entry, dict) and "episode_index" in entry:
                result[int(entry["episode_index"])] = bool(entry.get("is_success", False))
        return result if result else None

    if "is_success" in summary and isinstance(summary["is_success"], list):
        return {
            i: bool(v)
            for i, v in enumerate(summary["is_success"])
        }

    # Try dict-of-dicts keyed by episode index string.
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
    """Best-effort inference of episode success from available metadata."""
    summary = _load_collection_summary(dataset_path)
    parsed = _parse_collection_summary_success(summary)
    if parsed is not None and episode_index in parsed:
        return parsed[episode_index]

    # Fall back to episode parquet columns.
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


# --------------------------------------------------------------------------- #
# Video helpers
# --------------------------------------------------------------------------- #
def _infer_video_path(
    dataset_path: Path,
    episode_index: int,
    video_key: str = "image",
) -> Path | None:
    """Infer the video file path for an episode."""
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
    if videos_root.exists():
        patterns = [
            f"episode_{episode_index:06d}.mp4",
            f"episode_{episode_index:03d}.mp4",
            f"episode_{episode_index}.mp4",
            f"*{episode_index}*.mp4",
        ]
        for pattern in patterns:
            for candidate in videos_root.rglob(pattern):
                return candidate
    return None


def _load_video(video_path: Path) -> tuple[list[Image.Image], float]:
    """Decode all frames from a video file and return PIL images plus fps."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else DEFAULT_VIDEO_FPS
    frames: list[Image.Image] = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(Image.fromarray(arr))
    container.close()
    return frames, fps


def _sample_frame_indices(
    episode_length: int,
    sample_fps: float,
    video_fps: float,
) -> list[int]:
    """Return sampled frame indices at ``sample_fps``, including first/last."""
    if episode_length <= 0:
        return []
    interval = max(1, int(round(video_fps / sample_fps)))
    indices = list(range(0, episode_length, interval))
    if indices[-1] != episode_length - 1:
        indices.append(episode_length - 1)
    return indices


# --------------------------------------------------------------------------- #
# Prompt and parsing
# --------------------------------------------------------------------------- #
def _seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def _build_prompt(
    task_description: str,
    phase_definitions: dict[int, str],
    episode_length: int,
    duration_seconds: float,
    sample_fps: float,
    is_success: bool | None,
) -> str:
    """Build the whole-video prompt with outcome hint and phase vocabulary."""
    if is_success is True:
        outcome_hint = (
            "Environment outcome hint: this episode SUCCEEDED. "
            "The robot should complete all phases and end with phase 5."
        )
    elif is_success is False:
        outcome_hint = (
            "Environment outcome hint: this episode FAILED. "
            "The robot did NOT complete the task. Annotate ONLY the phases that "
            "actually happen visually; the last observed phase must extend to the "
            "end of the video. Do NOT invent later phases."
        )
    else:
        outcome_hint = (
            "Environment outcome hint: unknown. Judge from the video whether the "
            "task was completed. If it failed, annotate only observed phases."
        )

    phase_lines = "\n".join(
        f"  {phase_id}: {desc}" for phase_id, desc in sorted(phase_definitions.items())
    )

    return (
        "You are an expert robotics video analyst. I will show you a sampled "
        "sequence of frames from a single LIBERO-10 episode, in chronological order.\n\n"
        f'Task: "{task_description}"\n\n'
        f"Video statistics:\n"
        f"  Total frames: {episode_length}\n"
        f"  Duration: {duration_seconds:.2f} seconds\n"
        f"  Frames are sampled every {1.0 / sample_fps:.1f} seconds "
        f"({sample_fps} fps) for annotation.\n\n"
        f"{outcome_hint}\n\n"
        "Annotate the video with the following closed phase vocabulary. "
        "You may repeat phases if the robot retries an action (e.g. drops an object "
        "and re-grasps it). The phase sequence must reflect what actually happens visually, "
        "not what ideally should happen.\n\n"
        "Phase vocabulary (use these exact names):\n"
        f"{phase_lines}\n\n"
        "Important rules:\n"
        "  1. Every frame must belong to exactly one contiguous phase segment.\n"
        "  2. Segments must be contiguous in time: the end of one segment equals "
        "     the start of the next.\n"
        "  3. The first segment must start at 00:00.000 and the last segment must "
        "     end at the video duration.\n"
        "  4. If the robot retries a phase, output the same phase name again.\n"
        "  5. Use only names from the vocabulary above.\n\n"
        "Output format:\n"
        "First, briefly describe what you observe (1-3 sentences). Then output ONLY "
        "a JSON object in this exact format (no markdown code fences):\n\n"
        "{\n"
        '  "reasoning": "...",\n'
        '  "success": true or false,\n'
        '  "segments": [\n'
        '    {"name": "<phase name>", "start_seconds": 0.0, "end_seconds": 5.2},\n'
        "    ...\n"
        "  ]\n"
        "}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the largest JSON object from model text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try the whole text first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object.
    # Use a greedy regex that balances braces.
    for match in re.finditer(r"\{.*\}", text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract JSON from response: {text!r}")


def _parse_episode_annotation(text: str) -> EpisodeAnnotation:
    """Parse and validate the VLM annotation response."""
    parsed = _extract_json(text)
    return EpisodeAnnotation.model_validate(parsed)


def _resolve_phase_id(name: str) -> int:
    """Map a phase name (or numeric string) to a phase id."""
    name = name.strip().lower()

    # Exact match.
    if name in PHASE_NAME_TO_ID:
        return PHASE_NAME_TO_ID[name]

    # Numeric id.
    if name.isdigit():
        phase_id = int(name)
        if 0 <= phase_id < NUM_PHASES:
            return phase_id

    # Fuzzy match: does the response contain a phase id prefix like "phase 0"?
    match = re.search(r"\bphase\s*(\d+)\b", name)
    if match:
        phase_id = int(match.group(1))
        if 0 <= phase_id < NUM_PHASES:
            return phase_id

    # Substring match against vocabulary descriptions.
    for vocab_name, phase_id in PHASE_NAME_TO_ID.items():
        if name in vocab_name or vocab_name in name:
            return phase_id

    raise ValueError(f"Unknown phase name: {name!r}")


# --------------------------------------------------------------------------- #
# Validation and conversion
# --------------------------------------------------------------------------- #
def _validate_and_fix_segments(
    annotation: EpisodeAnnotation,
    duration_seconds: float,
    tolerance: float = 1.0,
) -> list[PhaseSegment]:
    """Validate segment contiguity/coverage and apply small snapping fixes."""
    segments = annotation.segments
    if not segments:
        raise ValueError("No segments returned by VLM.")

    # Map names to ids and validate.
    resolved: list[tuple[int, float, float]] = []
    for seg in segments:
        phase_id = _resolve_phase_id(seg.name)
        start = max(0.0, seg.start_seconds)
        end = min(duration_seconds, seg.end_seconds)
        if end < start:
            start, end = end, start
        resolved.append((phase_id, start, end))

    # Sort by start time.
    resolved.sort(key=lambda x: x[1])

    # Snap first start to 0 and last end to duration.
    if abs(resolved[0][1]) > tolerance:
        raise ValueError(
            f"First segment does not start near 0: {resolved[0][1]}"
        )
    resolved[0] = (resolved[0][0], 0.0, resolved[0][2])

    if abs(resolved[-1][2] - duration_seconds) > tolerance:
        raise ValueError(
            f"Last segment does not end near duration {duration_seconds}: "
            f"{resolved[-1][2]}"
        )
    resolved[-1] = (resolved[-1][0], resolved[-1][1], duration_seconds)

    # Merge/adjust tiny gaps/overlaps to enforce contiguity.
    fixed: list[PhaseSegment] = []
    for i, (phase_id, start, end) in enumerate(resolved):
        if i == 0:
            fixed.append(PhaseSegment(name=PHASE_DEFINITIONS[phase_id], start_seconds=start, end_seconds=end))
            continue

        prev = fixed[-1]
        gap = start - prev.end_seconds
        if abs(gap) <= tolerance:
            # Snap to prev end.
            start = prev.end_seconds
        elif gap > tolerance:
            raise ValueError(
                f"Gap between segments {i-1} and {i}: {prev.end_seconds} -> {start}"
            )
        else:
            # Overlap larger than tolerance: truncate previous segment.
            prev.end_seconds = start

        if start >= end:
            # Degenerate segment after fixing: skip it but issue a warning.
            print(f"[warn] segment {i} became degenerate after contiguity fix; skipping")
            continue

        fixed.append(PhaseSegment(name=PHASE_DEFINITIONS[phase_id], start_seconds=start, end_seconds=end))

    if not fixed:
        raise ValueError("All segments became degenerate after validation.")

    # Ensure the last segment still reaches the end.
    fixed[-1].end_seconds = duration_seconds
    return fixed


def _segments_to_frame_phase(
    segments: list[PhaseSegment],
    episode_length: int,
    fps: float,
) -> np.ndarray:
    """Convert validated segments to a per-frame phase array."""
    phase = np.zeros(episode_length, dtype=int)
    for seg in segments:
        phase_id = _resolve_phase_id(seg.name)
        start_frame = min(int(seg.start_seconds * fps), episode_length - 1)
        end_frame = min(int(np.ceil(seg.end_seconds * fps)), episode_length)
        end_frame = max(end_frame, start_frame + 1)
        phase[start_frame:end_frame] = phase_id
    return phase


# --------------------------------------------------------------------------- #
# Rule-based fallback
# --------------------------------------------------------------------------- #
def _rule_based_annotation(ep_file: Path, episode_length: int) -> np.ndarray:
    """Run the task1 rule-based annotator as a fallback."""
    df = pd.read_parquet(ep_file)
    actions = np.stack(df["actions"].values)
    state = (
        np.stack(df["state"].values)
        if "state" in df.columns
        else np.empty((len(df), 0))
    )
    phase, _ = annotate_task1(actions, state, len(df))
    return phase


# --------------------------------------------------------------------------- #
# Episode-level VLM annotation
# --------------------------------------------------------------------------- #
@dataclass
class AnnotateEpisodeResult:
    """Result of annotating one episode."""

    episode_index: int
    df: pd.DataFrame | None
    source: str  # "vlm" or "rule_fallback"
    error: str | None = None


def _build_episode_df(
    episode_index: int,
    phase: np.ndarray,
) -> pd.DataFrame:
    """Build the standard frame-level annotation dataframe for one episode."""
    episode_length = len(phase)
    phase_progress, global_progress, overall_progress = compute_progress_per_segment(
        phase, NUM_PHASES
    )
    return pd.DataFrame({
        "episode_index": np.full(episode_length, episode_index, dtype=np.int64),
        "frame_index": np.arange(episode_length, dtype=np.int64),
        "phase": phase.astype(np.int64),
        "phase_progress": phase_progress,
        "global_progress": global_progress,
        "progress": overall_progress,
    })


def _annotate_episode(
    dataset_path: Path,
    ep_file: Path,
    model: str,
    sample_fps: float,
    video_key: str,
    wrist_video_key: str | None,
    cache_path: Path,
    cache: dict[int, dict[str, Any]],
    enable_reasoning: bool,
    fallback_on_failure: bool,
) -> AnnotateEpisodeResult:
    """Annotate one episode with whole-video VLM, with rule-based fallback."""
    df = pd.read_parquet(ep_file)
    if len(df) == 0:
        return AnnotateEpisodeResult(
            episode_index=-1, df=None, source="vlm", error="empty episode"
        )

    episode_index = int(df["episode_index"].iloc[0])
    episode_length = len(df)

    cached = cache.get(episode_index)
    if cached is not None:
        # Reject stale cache entries whose config does not match the current run.
        if cached.get("sample_fps") != sample_fps or cached.get("model") != model:
            print(
                f"[info] stale cache for episode {episode_index} "
                f"(sample_fps/model mismatch); re-annotating"
            )
            cached = None
    if cached is not None:
        try:
            annotation = EpisodeAnnotation.model_validate(cached["response"])
            video_path = _infer_video_path(dataset_path, episode_index, video_key=video_key)
            _, fps = _load_video(video_path) if video_path else ([], DEFAULT_VIDEO_FPS)
            duration = episode_length / fps
            segments = _validate_and_fix_segments(annotation, duration)
            phase = _segments_to_frame_phase(segments, episode_length, fps)
            return AnnotateEpisodeResult(
                episode_index=episode_index,
                df=_build_episode_df(episode_index, phase),
                source="vlm",
            )
        except Exception as e:
            print(f"[warn] cache hit for episode {episode_index} but invalid: {e}")

    is_success = _infer_episode_success(dataset_path, episode_index, ep_file)

    video_path = _infer_video_path(dataset_path, episode_index, video_key=video_key)
    if video_path is None:
        if fallback_on_failure:
            print(f"[warn] video not found for episode {episode_index}; using rule fallback")
            phase = _rule_based_annotation(ep_file, episode_length)
            return AnnotateEpisodeResult(
                episode_index=episode_index,
                df=_build_episode_df(episode_index, phase),
                source="rule_fallback",
                error="video not found",
            )
        return AnnotateEpisodeResult(
            episode_index=episode_index,
            df=None,
            source="vlm",
            error="video not found",
        )

    frames, fps = _load_video(video_path)
    duration = episode_length / fps

    sampled_indices = _sample_frame_indices(episode_length, sample_fps, fps)
    sampled_images = [frames[i] for i in sampled_indices]

    prompt = _build_prompt(
        task_description=TASK_DESCRIPTION,
        phase_definitions=PHASE_DEFINITIONS,
        episode_length=episode_length,
        duration_seconds=duration,
        sample_fps=sample_fps,
        is_success=is_success,
    )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            parsed = call_qwen_vl(
                sampled_images,
                prompt,
                model=model,
                enable_reasoning=enable_reasoning,
                response_parser=_parse_episode_annotation,
            )
            annotation = EpisodeAnnotation.model_validate(parsed)
            segments = _validate_and_fix_segments(annotation, duration)
            phase = _segments_to_frame_phase(segments, episode_length, fps)
            entry = {
                "episode_index": episode_index,
                "task_description": TASK_DESCRIPTION,
                "model": model,
                "sample_fps": sample_fps,
                "is_success": is_success,
                "response": annotation.model_dump(),
            }
            _append_cache(cache_path, entry)
            with _CACHE_LOCK:
                cache[episode_index] = entry

            return AnnotateEpisodeResult(
                episode_index=episode_index,
                df=_build_episode_df(episode_index, phase),
                source="vlm",
            )
        except Exception as e:
            last_error = e
            print(f"[warn] VLM annotation failed for episode {episode_index} (attempt {attempt + 1}): {e}")
            # On second attempt, strengthen the failure hint.
            if attempt == 0 and is_success is False:
                prompt += (
                    "\n\nReminder: this episode FAILED. Do NOT output phases that were "
                    "not visually executed. The final segment must extend to the video end."
                )

    if fallback_on_failure:
        print(f"[warn] falling back to rule-based annotation for episode {episode_index}")
        phase = _rule_based_annotation(ep_file, episode_length)
        return AnnotateEpisodeResult(
            episode_index=episode_index,
            df=_build_episode_df(episode_index, phase),
            source="rule_fallback",
            error=str(last_error),
        )

    return AnnotateEpisodeResult(
        episode_index=episode_index,
        df=None,
        source="vlm",
        error=str(last_error),
    )


# --------------------------------------------------------------------------- #
# Dataset-level runner
# --------------------------------------------------------------------------- #
def annotate_dataset_with_vlm(
    dataset_path: str | Path,
    output_name: str = "phase_progress_semantic_vlm",
    model: str = DEFAULT_MODEL,
    sample_fps: float = 0.5,
    video_key: str = "image",
    wrist_video_key: str | None = None,
    max_workers: int = 2,
    enable_reasoning: bool = True,
    fallback_on_failure: bool = True,
) -> Path:
    """Annotate an entire LeRobot dataset with whole-video VLM phase labels.

    Args:
        dataset_path: Path to LeRobot dataset root.
        output_name: Output parquet filename under meta/.
        model: DashScope or local vLLM model name.
        sample_fps: Frame sampling rate sent to the VLM (original video fps / interval).
        video_key: Key for the main camera video.
        wrist_video_key: Key for the wrist camera video, or None to disable.
            Not used for whole-video annotation yet, but kept for interface parity.
        max_workers: Number of episodes processed in parallel.
        enable_reasoning: Whether to request model reasoning/thinking.
        fallback_on_failure: Whether to fall back to rule-based annotation when
            the VLM fails or returns invalid output.

    Returns:
        Path to the written parquet file.
    """
    dataset_path = Path(dataset_path)
    data_root = dataset_path / "data"
    ep_files = sorted(data_root.rglob("episode_*.parquet"))
    print(f"Found {len(ep_files)} episodes in {data_root}")

    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cache_path = meta_dir / f"qwen_phase_cache_{output_name}.jsonl"
    cache = _load_cache(cache_path)
    print(f"Loaded {len(cache)} cached VLM responses from {cache_path}")

    records: list[pd.DataFrame] = []
    fallback_count = 0
    error_count = 0

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _annotate_episode,
                    dataset_path,
                    ep_file,
                    model,
                    sample_fps,
                    video_key,
                    wrist_video_key,
                    cache_path,
                    cache,
                    enable_reasoning,
                    fallback_on_failure,
                ): ep_file
                for ep_file in ep_files
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Annotating episodes"
            ):
                try:
                    result = future.result()
                    if result.df is not None:
                        records.append(result.df)
                        if result.source == "rule_fallback":
                            fallback_count += 1
                    else:
                        error_count += 1
                        print(f"[warn] episode {result.episode_index}: {result.error}")
                except Exception as e:
                    ep_file = futures[future]
                    print(f"[warn] failed to annotate {ep_file}: {e}")
                    error_count += 1
    else:
        for ep_file in tqdm(ep_files, desc="Annotating episodes"):
            result = _annotate_episode(
                dataset_path,
                ep_file,
                model,
                sample_fps,
                video_key,
                wrist_video_key,
                cache_path,
                cache,
                enable_reasoning,
                fallback_on_failure,
            )
            if result.df is not None:
                records.append(result.df)
                if result.source == "rule_fallback":
                    fallback_count += 1
            else:
                error_count += 1
                print(f"[warn] episode {result.episode_index}: {result.error}")

    if not records:
        raise RuntimeError("No episodes were successfully annotated.")

    out_df = pd.concat(records, ignore_index=True)
    out_df = out_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    out_path = meta_dir / f"{output_name}.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"\nSaved {len(out_df)} frame labels to {out_path}")
    print(f"VLM annotated: {len(records) - fallback_count}")
    print(f"Rule fallback: {fallback_count}")
    print(f"Errors: {error_count}")
    print("Phase distribution:")
    print(out_df["phase"].value_counts().sort_index())
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whole-video VLM phase annotation for LIBERO-10 Task 1."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to LeRobot dataset root (contains data/ and meta/).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="phase_progress_semantic_vlm",
        help="Output parquet name (saved under dataset/meta/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL),
        help="Qwen model name (local vLLM or DashScope).",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=0.5,
        help="Sampling rate of frames sent to the VLM (default 0.5 fps).",
    )
    parser.add_argument(
        "--video_key",
        type=str,
        default="image",
        help="Video key for the main camera.",
    )
    parser.add_argument(
        "--wrist_video_key",
        type=str,
        default=None,
        help="Video key for the wrist camera (currently unused).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Number of episodes to annotate in parallel.",
    )
    parser.add_argument(
        "--enable_reasoning",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Enable model reasoning/thinking.",
    )
    parser.add_argument(
        "--no_fallback",
        action="store_true",
        help="Disable rule-based fallback on VLM failure.",
    )
    args = parser.parse_args()

    out_path = annotate_dataset_with_vlm(
        dataset_path=args.dataset_path,
        output_name=args.output_name,
        model=args.model,
        sample_fps=args.sample_fps,
        video_key=args.video_key,
        wrist_video_key=args.wrist_video_key,
        max_workers=args.max_workers,
        enable_reasoning=args.enable_reasoning,
        fallback_on_failure=not args.no_fallback,
    )
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
