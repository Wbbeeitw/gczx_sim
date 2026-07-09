"""Visualize phase annotations on top of episode videos.

Reads a LeRobot dataset with `meta/phase_progress_semantic.parquet` and renders
a few demo videos with phase overlay and progress bar.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from PIL import Image


# Color map for phases (BGR for OpenCV).
PHASE_COLORS = {
    0: (0, 200, 255),    # orange
    1: (0, 255, 0),      # green
    2: (255, 0, 0),      # blue
    3: (255, 0, 255),    # magenta
    4: (0, 255, 255),    # yellow
    5: (128, 0, 128),    # purple
    6: (0, 128, 255),    # light red
}


def load_collection_summary(dataset_path: Path) -> dict:
    summary_path = dataset_path / "collection_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def list_video_files(dataset_path: Path) -> list[Path]:
    videos_root = dataset_path / "videos"
    if not videos_root.exists():
        return []
    return sorted(videos_root.rglob("*.mp4"))


def infer_video_path(
    dataset_path: Path, episode_index: int, video_key: str = "image"
) -> Path | None:
    """Infer the video file path for an episode.

    First uses meta/info.json video_path pattern, then falls back to recursive search.
    """
    info = load_info(dataset_path)
    video_path_pattern = info.get("video_path")

    if video_path_pattern:
        # LeRobot info.json uses braces like {episode_chunk:03d}
        formatted = (
            video_path_pattern
            .replace("{episode_chunk:03d}", f"{episode_index // 1000:03d}")
            .replace("{episode_index:06d}", f"{episode_index:06d}")
            .replace("{video_key}", video_key)
        )
        candidate = dataset_path / formatted
        if candidate.exists():
            return candidate

    # Fallback: standard LeRobot layout.
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

    # Fallback: recursive search for any matching episode file.
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
        for candidate in videos_root.rglob(pattern):
            return candidate
    return None


def overlay_phase_info(
    frame: np.ndarray,
    phase: int,
    phase_progress: float,
    global_progress: float,
    frame_index: int,
    episode_index: int,
    task: str | None = None,
) -> np.ndarray:
    """Draw compact phase text, progress bar, and frame info on the frame.

    The overlay is tuned for 256x256 LIBERO videos so that all text fits
    within the frame width.
    """
    h, w = frame.shape[:2]
    color = PHASE_COLORS.get(phase, (255, 255, 255))

    # Black translucent top bar (compact for small resolution).
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Compact two-line info.
    line1 = f"ep={episode_index} frame={frame_index} phase={phase}"
    line2 = f"p_phase={phase_progress:.2f} p_global={global_progress:.2f}"
    cv2.putText(frame, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, line2, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Optional task text (very small, truncated to frame width).
    if task:
        task_text = str(task)
        max_chars = max(20, w // 6)
        if len(task_text) > max_chars:
            task_text = task_text[: max_chars - 3] + "..."
        cv2.putText(frame, task_text, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    # Progress bar at the bottom.
    bar_h = 14
    bar_y = h - bar_h - 8
    bar_w = w - 30
    cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    filled_w = int(bar_w * global_progress)
    cv2.rectangle(frame, (15, bar_y), (15 + filled_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + bar_h), (255, 255, 255), 1)

    # Current phase marker on the bar.
    marker_x = 15 + filled_w
    cv2.circle(frame, (marker_x, bar_y + bar_h // 2), 5, (255, 255, 255), -1)
    cv2.circle(frame, (marker_x, bar_y + bar_h // 2), 5, color, 1)

    return frame


def render_episode(
    dataset_path: Path,
    annotations: pd.DataFrame,
    episode_index: int,
    output_path: Path,
    task: str | None = None,
    video_key: str = "image",
) -> None:
    """Render one episode video with phase overlay."""
    video_path = infer_video_path(dataset_path, episode_index, video_key=video_key)
    if video_path is None:
        available = list_video_files(dataset_path)[:10]
        print(f"[warn] video not found for episode {episode_index}")
        if available:
            print(f"[debug] available videos (first {len(available)}):")
            for p in available:
                print(f"  - {p.relative_to(dataset_path)}")
        else:
            print("[debug] no .mp4 files found under videos/")
        return

    ep_ann = annotations[annotations["episode_index"] == episode_index].sort_values("frame_index")
    if ep_ann.empty:
        print(f"[warn] no annotations for episode {episode_index}")
        return

    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
    except Exception as e:
        print(f"[warn] cannot open video {video_path}: {e}")
        return

    fps = float(video_stream.average_rate) if video_stream.average_rate else 10.0
    width = int(video_stream.width)
    height = int(video_stream.height)

    # Decode all frames and apply overlay.
    frames_bgr: list[np.ndarray] = []
    frame_idx = 0
    for frame in container.decode(video_stream):
        img = frame.to_ndarray(format="bgr24")

        ann_rows = ep_ann[ep_ann["frame_index"] == frame_idx]
        if not ann_rows.empty:
            row = ann_rows.iloc[0]
            img = overlay_phase_info(
                img,
                phase=int(row["phase"]),
                phase_progress=float(row["phase_progress"]),
                global_progress=float(row["global_progress"]),
                frame_index=frame_idx,
                episode_index=episode_index,
                task=task,
            )

        frames_bgr.append(img)
        frame_idx += 1

    container.close()

    if not frames_bgr:
        print(f"[warn] no frames decoded for episode {episode_index}")
        return

    # Try to write an mp4 video first, fall back to GIF if no encoder is available.
    video_saved = False
    try:
        output_container = av.open(str(output_path), "w")
        output_stream = None
        for codec in ("libx264", "h264", "mpeg4", "libsvtav1"):
            try:
                output_stream = output_container.add_stream(codec, rate=fps)
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = "yuv420p"
                break
            except Exception:
                continue
        if output_stream is None:
            raise RuntimeError("no available video encoder")

        for img in frames_bgr:
            out_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            for packet in output_stream.encode(out_frame):
                output_container.mux(packet)
        for packet in output_stream.encode():
            output_container.mux(packet)

        output_container.close()
        video_saved = True
        print(f"[saved] {output_path}")
    except Exception as e:
        print(f"[warn] mp4 encoding failed: {e}")

    if not video_saved:
        gif_path = output_path.with_suffix(".gif")
        try:
            # Convert BGR to RGB for PIL.
            pil_frames = [Image.fromarray(img[:, :, ::-1]) for img in frames_bgr]
            duration_ms = max(1, int(1000 / fps))
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
            )
            print(f"[saved] {gif_path}")
        except Exception as e2:
            print(f"[warn] gif encoding also failed: {e2}")


def select_demo_episodes(
    dataset_path: Path,
    annotations: pd.DataFrame,
    num_success: int | None,
    num_failure: int | None,
    seed: int | None,
) -> list[int]:
    """Select demo episodes, optionally stratified by success/failure.

    Args:
        dataset_path: Path to LeRobot dataset root.
        annotations: Phase annotation dataframe.
        num_success: Number of successful episodes to render, or None to disable
            stratification.
        num_failure: Number of failed episodes to render, or None to disable
            stratification.
        seed: Random seed for episode selection. If None, selection is
            deterministic (sorted order).

    Returns:
        List of selected episode indices.
    """
    summary = load_collection_summary(dataset_path)
    all_eps = sorted(annotations["episode_index"].unique())

    # Try to read per-episode success from parquet if available.
    if "is_success" in annotations.columns:
        success_eps = sorted(annotations[annotations["is_success"]]["episode_index"].unique())
        failure_eps = sorted(annotations[~annotations["is_success"]]["episode_index"].unique())
    else:
        success_eps = []
        failure_eps = []

    # Fall back to collection_summary.json if available.
    if not success_eps and not failure_eps and summary and isinstance(summary, dict):
        parsed: dict[int, bool] | None = None
        if "episodes" in summary and isinstance(summary["episodes"], list):
            parsed = {
                int(e["episode_index"]): bool(e.get("is_success", False))
                for e in summary["episodes"]
                if isinstance(e, dict) and "episode_index" in e
            }
        elif "is_success" in summary and isinstance(summary["is_success"], list):
            parsed = {i: bool(v) for i, v in enumerate(summary["is_success"])}
        if parsed:
            success_eps = sorted([ep for ep, ok in parsed.items() if ok and ep in all_eps])
            failure_eps = sorted([ep for ep, ok in parsed.items() if not ok and ep in all_eps])

    if num_success is None and num_failure is None:
        # Backward-compatible behavior: first success/failure mix.
        num_total = 4
        selected: list[int] = []
        selected.extend(success_eps[: num_total // 2])
        selected.extend(failure_eps[: num_total - len(selected)])
        for ep in all_eps:
            if ep not in selected:
                selected.append(ep)
            if len(selected) >= num_total:
                break
        return selected[:num_total]

    if seed is not None:
        rng = np.random.default_rng(seed)
        success_eps = list(rng.permutation(success_eps))
        failure_eps = list(rng.permutation(failure_eps))
        all_eps = list(rng.permutation(all_eps))

    selected = []
    if num_success is not None:
        selected.extend(success_eps[:num_success])
    if num_failure is not None:
        selected.extend(failure_eps[:num_failure])

    # Fill remaining slots from all episodes without duplication.
    requested_total = (num_success or 0) + (num_failure or 0)
    for ep in all_eps:
        if ep not in selected:
            selected.append(ep)
        if len(selected) >= requested_total:
            break

    return selected[:requested_total]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize phase annotations on episode videos.")
    parser.add_argument("--dataset_path", required=True, help="Path to LeRobot dataset root.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for demo videos. Defaults to <repo_root>/phase_split_script/output_demo.",
    )
    parser.add_argument(
        "--annotation_name",
        type=str,
        default="phase_progress_semantic",
        help="Name of the annotation parquet under meta/ (without .parquet extension).",
    )
    parser.add_argument(
        "--num_success",
        type=int,
        default=None,
        help="Number of successful episodes to render. Enables stratified selection.",
    )
    parser.add_argument(
        "--num_failure",
        type=int,
        default=None,
        help="Number of failed episodes to render. Enables stratified selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for episode selection. If not set, selection is deterministic.",
    )
    parser.add_argument("--video_key", default="image", help="Video key to render (image or wrist_image).")
    parser.add_argument("--task", default=None, help="Optional task description to overlay.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    annotations_path = dataset_path / "meta" / f"{args.annotation_name}.parquet"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")

    annotations = pd.read_parquet(annotations_path)

    if args.output_dir is None:
        repo_root = Path(__file__).resolve().parent
        output_dir = repo_root / "output_demo" / args.annotation_name
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear old demo videos to avoid confusion.
    for old_mp4 in output_dir.glob("*.mp4"):
        old_mp4.unlink()

    episodes = select_demo_episodes(
        dataset_path, annotations, args.num_success, args.num_failure, args.seed
    )

    task = args.task
    if task is None:
        summary = load_collection_summary(dataset_path)
        task = summary.get("task_suite_name", "")

    for ep_idx in episodes:
        output_path = output_dir / f"{args.annotation_name}_episode_{ep_idx:03d}_phases.mp4"
        render_episode(
            dataset_path=dataset_path,
            annotations=annotations,
            episode_index=ep_idx,
            output_path=output_path,
            task=task,
            video_key=args.video_key,
        )

    print(f"\nDone. Rendered {len(episodes)} demo videos to {output_dir}")


if __name__ == "__main__":
    main()
