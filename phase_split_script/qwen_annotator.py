"""Generic Qwen-VL based phase annotation for LIBERO rollout datasets.

Decodes episode videos, samples frames at a fixed interval, calls Qwen-VL for
phase labels, enforces temporal monotonicity, and writes a standard
``meta/phase_progress_semantic.parquet`` sidecar.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common_annotator import compute_progress
from qwen_client import call_qwen_vl, DEFAULT_MODEL


_CACHE_LOCK = threading.Lock()


def _load_info(dataset_path: Path) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _infer_video_path(
    dataset_path: Path,
    episode_index: int,
    video_key: str = "image",
) -> Path | None:
    """Infer the video file path for an episode."""
    info = _load_info(dataset_path)
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


def _load_video_frames(video_path: Path) -> list[Image.Image]:
    """Decode all frames from a video file into RGB PIL images."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames: list[Image.Image] = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(Image.fromarray(arr))
    container.close()
    return frames


def _build_prompt(
    task_description: str,
    phase_definitions: dict[int, str],
    prev_phase: int | None,
) -> str:
    """Build a Qwen prompt from the task description and phase definitions."""
    lines = [
        f'Task: "{task_description}"',
        "",
        "You are annotating a robot manipulation video. "
        "Classify the current frame into exactly one of these phases:",
        "",
    ]
    for phase_id, desc in sorted(phase_definitions.items()):
        lines.append(f"{phase_id}: {desc}")
    lines.append("")

    if prev_phase is not None:
        lines.append(f"Previous sampled frame was phase: {prev_phase}")
        lines.append("The current phase must be >= the previous phase.")
        lines.append("")

    lines.append(
        "Look at the robot arm, the gripper, and the objects. "
        "Output ONLY JSON in this exact format:"
    )
    lines.append('{"phase": <int>, "confidence": "high" or "medium" or "low"}')
    return "\n".join(lines)


def _load_cache(cache_path: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Load cached Qwen responses keyed by (episode_index, frame_index, task_description)."""
    cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (
                int(entry["episode_index"]),
                int(entry["frame_index"]),
                str(entry.get("task_description", "")),
            )
            cache[key] = entry
    return cache


def _append_cache(cache_path: Path, entry: dict[str, Any]) -> None:
    """Append a single cache entry to the JSONL cache (thread-safe)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _annotate_one_sample(
    episode_index: int,
    frame_index: int,
    images: list[Image.Image],
    task_description: str,
    phase_definitions: dict[int, str],
    prev_phase: int | None,
    model: str,
    cache_path: Path,
    cache: dict[tuple[int, int, str], dict[str, Any]],
) -> int:
    """Annotate a single sampled frame, using cache if available.

    Returns the phase number.
    """
    key = (episode_index, frame_index, task_description)
    cached = cache.get(key)
    if cached is not None:
        return int(cached["phase"])

    prompt = _build_prompt(task_description, phase_definitions, prev_phase)
    parsed = call_qwen_vl(images, prompt, model=model)

    num_phases = len(phase_definitions)
    phase = int(parsed.get("phase", 0))
    phase = max(0, min(phase, num_phases - 1))

    entry = {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_description": task_description,
        "model": model,
        "response": parsed,
        "phase": phase,
    }
    _append_cache(cache_path, entry)

    with _CACHE_LOCK:
        cache[key] = entry

    return phase



def _sampled_indices(episode_length: int, sample_interval: int) -> list[int]:
    """Return sampled frame indices, always including the last frame."""
    indices = list(range(0, episode_length, sample_interval))
    if indices[-1] != episode_length - 1:
        indices.append(episode_length - 1)
    return indices


def _sampled_phase_array(
    sampled_results: list[tuple[int, int]],
    num_phases: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a monotonic phase array from sampled (frame_index, phase) pairs."""
    sampled_results = sorted(set(sampled_results), key=lambda x: x[0])
    if not sampled_results:
        return np.array([], dtype=int), np.array([], dtype=int)

    frame_indices = np.array([x[0] for x in sampled_results], dtype=int)
    phases = np.array([x[1] for x in sampled_results], dtype=int)

    # Enforce monotonicity: phase can stay the same or increase.
    for i in range(1, len(phases)):
        phases[i] = max(phases[i], phases[i - 1])

    phases = np.clip(phases, 0, num_phases - 1)
    return frame_indices, phases


def _interpolate_phases(
    frame_indices: np.ndarray,
    phases: np.ndarray,
    episode_length: int,
) -> np.ndarray:
    """Fill phase labels for all frames using forward-fill from sampled frames."""
    full_phase = np.zeros(episode_length, dtype=int)
    if len(frame_indices) == 0:
        return full_phase

    for i in range(len(frame_indices)):
        start = frame_indices[i]
        end = frame_indices[i + 1] if i + 1 < len(frame_indices) else episode_length
        full_phase[start:end] = phases[i]

    return full_phase


def _annotate_episode(
    dataset_path: Path,
    ep_file: Path,
    task_description: str,
    phase_definitions: dict[int, str],
    num_phases: int,
    sample_interval: int,
    model: str,
    cache_path: Path,
    cache: dict[tuple[int, int], dict[str, Any]],
    video_key: str,
    wrist_video_key: str | None,
) -> pd.DataFrame | None:
    """Annotate one episode using Qwen-VL (sequential samples within episode)."""
    df = pd.read_parquet(ep_file)
    if len(df) == 0:
        return None

    episode_index = int(df["episode_index"].iloc[0])
    episode_length = len(df)

    main_video_path = _infer_video_path(dataset_path, episode_index, video_key=video_key)
    if main_video_path is None:
        print(f"[warn] main video not found for episode {episode_index}")
        return None

    main_frames = _load_video_frames(main_video_path)

    wrist_frames: list[Image.Image] | None = None
    if wrist_video_key:
        wrist_video_path = _infer_video_path(
            dataset_path, episode_index, video_key=wrist_video_key
        )
        if wrist_video_path is not None:
            wrist_frames = _load_video_frames(wrist_video_path)

    sampled_indices = _sampled_indices(episode_length, sample_interval)
    sampled_results: list[tuple[int, int]] = []
    prev_phase: int | None = None

    for frame_idx in sampled_indices:
        images = [main_frames[frame_idx]]
        if wrist_frames is not None:
            images.append(wrist_frames[frame_idx])

        phase = _annotate_one_sample(
            episode_index=episode_index,
            frame_index=frame_idx,
            images=images,
            task_description=task_description,
            phase_definitions=phase_definitions,
            prev_phase=prev_phase,
            model=model,
            cache_path=cache_path,
            cache=cache,
        )
        sampled_results.append((frame_idx, phase))
        prev_phase = phase

    frame_indices, phases = _sampled_phase_array(sampled_results, num_phases)
    full_phase = _interpolate_phases(frame_indices, phases, episode_length)
    phase_progress, global_progress, overall_progress = compute_progress(
        full_phase, num_phases
    )

    return pd.DataFrame({
        "episode_index": np.full(episode_length, episode_index, dtype=np.int64),
        "frame_index": np.arange(episode_length, dtype=np.int64),
        "phase": full_phase.astype(np.int64),
        "phase_progress": phase_progress,
        "global_progress": global_progress,
        "progress": overall_progress,
    })


def annotate_dataset_with_qwen(
    dataset_path: str | Path,
    task_description: str,
    phase_definitions: dict[int, str],
    num_phases: int,
    output_name: str = "phase_progress_semantic",
    sample_interval: int = 10,
    model: str = DEFAULT_MODEL,
    video_key: str = "image",
    wrist_video_key: str | None = "wrist_image",
    max_workers: int = 2,
    cache_name: str = "qwen_phase_cache.jsonl",
) -> Path:
    """Annotate an entire LeRobot dataset using Qwen-VL.

    Args:
        dataset_path: Path to LeRobot dataset root.
        task_description: Natural language task instruction.
        phase_definitions: Mapping from phase id to description string.
        num_phases: Number of phases (should match len(phase_definitions)).
        output_name: Output parquet filename under meta/.
        sample_interval: Call Qwen every N frames.
        model: DashScope model name.
        video_key: Key for the main camera video.
        wrist_video_key: Key for the wrist camera video, or None to disable.
        max_workers: Number of episodes processed in parallel.
        cache_name: Cache filename under meta/.

    Returns:
        Path to the written parquet file.
    """
    dataset_path = Path(dataset_path)
    data_root = dataset_path / "data"
    ep_files = sorted(data_root.rglob("episode_*.parquet"))
    print(f"Found {len(ep_files)} episodes in {data_root}")

    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cache_path = meta_dir / cache_name
    cache = _load_cache(cache_path)
    print(f"Loaded {len(cache)} cached Qwen responses from {cache_path}")

    total_samples = sum(
        len(_sampled_indices(len(pd.read_parquet(ep)), sample_interval))
        for ep in ep_files
    )
    print(f"Total samples to annotate: ~{total_samples} ({len(cache)} already cached)")

    records: list[pd.DataFrame] = []

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _annotate_episode,
                    dataset_path,
                    ep_file,
                    task_description,
                    phase_definitions,
                    num_phases,
                    sample_interval,
                    model,
                    cache_path,
                    cache,
                    video_key,
                    wrist_video_key,
                ): ep_file
                for ep_file in ep_files
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Annotating episodes"
            ):
                try:
                    ep_df = future.result()
                    if ep_df is not None:
                        records.append(ep_df)
                except Exception as e:
                    ep_file = futures[future]
                    print(f"[warn] failed to annotate {ep_file}: {e}")
    else:
        for ep_file in tqdm(ep_files, desc="Annotating episodes"):
            ep_df = _annotate_episode(
                dataset_path,
                ep_file,
                task_description,
                phase_definitions,
                num_phases,
                sample_interval,
                model,
                cache_path,
                cache,
                video_key,
                wrist_video_key,
            )
            if ep_df is not None:
                records.append(ep_df)

    if not records:
        raise RuntimeError("No episodes were successfully annotated.")

    out_df = pd.concat(records, ignore_index=True)
    out_df = out_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    out_path = meta_dir / f"{output_name}.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"\nSaved {len(out_df)} frame labels to {out_path}")
    print("Phase distribution:")
    print(out_df["phase"].value_counts().sort_index())
    return out_path
