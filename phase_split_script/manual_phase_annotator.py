"""Simple keyboard-driven phase annotator for LIBERO episodes.

Dependencies (install manually):
    pip install opencv-python numpy pandas

Usage:
    python manual_phase_annotator.py \
        --dataset_path /data/libero_long/task1 \
        --episode_index 3 \
        --output_path manual_labels/episode_003.json

Keyboard controls:
    0-5      set current frame to phase 0-5
    ← / →    previous / next frame (1 frame)
    ↑ / ↓    previous / next frame (10 frames)
    space    play / pause
    s        save labels
    q        quit (saves automatically if changed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# Color map for phases (BGR for OpenCV).
PHASE_COLORS = {
    0: (0, 200, 255),    # orange
    1: (0, 255, 0),      # green
    2: (255, 0, 0),      # blue
    3: (255, 0, 255),    # magenta
    4: (0, 255, 255),    # yellow
    5: (128, 0, 128),    # purple
}

DEFAULT_PHASE = 0
SEEK_SMALL = 1
SEEK_LARGE = 10


def load_video_path(dataset_path: Path, episode_index: int, video_key: str = "image") -> Path | None:
    """Infer the video file path for an episode."""
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            import json as _json
            info = _json.load(f)
        pattern = info.get("video_path", "")
        if pattern:
            formatted = (
                pattern
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
        for pattern in [
            f"episode_{episode_index:06d}.mp4",
            f"episode_{episode_index:03d}.mp4",
            f"episode_{episode_index}.mp4",
        ]:
            for candidate in videos_root.rglob(pattern):
                return candidate
    return None


def load_frames(video_path: Path) -> list[np.ndarray]:
    """Load all frames from a video file as BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    return frames


def switches_to_phase_array(switches: list[dict], length: int) -> np.ndarray:
    """Convert a list of phase switches to a dense phase array."""
    phase = np.full(length, DEFAULT_PHASE, dtype=int)
    if not switches:
        return phase
    switches = sorted(switches, key=lambda x: int(x["frame_index"]))
    for i, sw in enumerate(switches):
        start = int(sw["frame_index"])
        end = switches[i + 1]["frame_index"] if i + 1 < len(switches) else length
        ph = int(sw["phase"])
        phase[start:end] = ph
    return phase


def phase_array_to_switches(phase: np.ndarray) -> list[dict]:
    """Convert a dense phase array to a list of phase switches."""
    switches = [{"frame_index": 0, "phase": int(phase[0])}]
    for i in range(1, len(phase)):
        if phase[i] != phase[i - 1]:
            switches.append({"frame_index": i, "phase": int(phase[i])})
    return switches


def render_frame(
    frame: np.ndarray,
    episode_index: int,
    frame_index: int,
    total_frames: int,
    phase: int,
    is_playing: bool,
) -> np.ndarray:
    """Draw overlay info on the frame."""
    h, w = frame.shape[:2]
    color = PHASE_COLORS.get(phase, (255, 255, 255))

    # Top black bar.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Text info.
    line1 = f"ep={episode_index}  frame={frame_index}/{total_frames - 1}  phase={phase}"
    line2 = "keys: 0-5=phase  arrows=seek  space=play  s=save  q=quit"
    if is_playing:
        line2 += "  [PLAYING]"
    cv2.putText(frame, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, line2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # Bottom progress bar.
    bar_h = 12
    bar_y = h - bar_h - 8
    bar_w = w - 30
    progress = frame_index / max(total_frames - 1, 1)
    filled_w = int(bar_w * progress)
    cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (15, bar_y), (15 + filled_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + bar_h), (255, 255, 255), 1)

    return frame


def save_labels(output_path: Path, episode_index: int, switches: list[dict]) -> None:
    """Save phase switches to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "episode_index": int(episode_index),
        "phase_switches": switches,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


def load_existing_labels(output_path: Path) -> list[dict]:
    """Load existing phase switches if available."""
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("phase_switches", [])
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually annotate phases for a single episode.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--episode_index", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--video_key", type=str, default="image")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS when space is pressed.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    video_path = load_video_path(dataset_path, args.episode_index, video_key=args.video_key)
    if video_path is None:
        raise FileNotFoundError(f"Video not found for episode {args.episode_index}")

    print(f"Loading video: {video_path}")
    frames = load_frames(video_path)
    total_frames = len(frames)
    print(f"Loaded {total_frames} frames")

    existing_switches = load_existing_labels(output_path)
    phase = switches_to_phase_array(existing_switches, total_frames)

    window_name = f"Phase Annotator - Episode {args.episode_index}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    is_playing = False
    changed = False

    while True:
        displayed = render_frame(
            frames[frame_idx].copy(),
            args.episode_index,
            frame_idx,
            total_frames,
            int(phase[frame_idx]),
            is_playing,
        )
        cv2.imshow(window_name, displayed)

        if is_playing:
            delay = max(1, int(1000 / args.fps))
            key = cv2.waitKey(delay) & 0xFF
            if key == ord(" "):
                is_playing = False
            elif key != 255:
                is_playing = False
        else:
            key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            if changed:
                save_labels(output_path, args.episode_index, phase_array_to_switches(phase))
            break
        elif key == ord("s"):
            save_labels(output_path, args.episode_index, phase_array_to_switches(phase))
            changed = False
        elif key == ord(" "):
            is_playing = not is_playing
        elif key == 83:  # right arrow
            frame_idx = min(total_frames - 1, frame_idx + SEEK_SMALL)
        elif key == 81:  # left arrow
            frame_idx = max(0, frame_idx - SEEK_SMALL)
        elif key == 82:  # up arrow
            frame_idx = min(total_frames - 1, frame_idx + SEEK_LARGE)
        elif key == 84:  # down arrow
            frame_idx = max(0, frame_idx - SEEK_LARGE)
        elif ord("0") <= key <= ord("5"):
            new_phase = key - ord("0")
            if phase[frame_idx] != new_phase:
                # Apply new phase from current frame to the next switch or end.
                phase[frame_idx:] = new_phase
                changed = True
            if is_playing:
                is_playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
