"""Stack existing critic-trajectory PNGs into one vertical figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def _resolve_png(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    if not path.is_file():
        raise FileNotFoundError(f"Panel PNG not found: {path}")
    return path


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _stack_panels(
    panel_paths: list[Path],
    output_path: Path,
    dpi: int,
    gap_px: int,
) -> tuple[int, int]:
    images = [Image.open(path).convert("RGB") for path in panel_paths]
    width = max(image.width for image in images)
    images = [_resize_to_width(image, width) for image in images]
    height = sum(image.height for image in images) + gap_px * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    current_y = 0
    for image in images:
        canvas.paste(image, (0, current_y))
        current_y += image.height + gap_px
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=(dpi, dpi))
    return canvas.size


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stack two or more wide critic-trajectory PNGs vertically."
    )
    parser.add_argument(
        "--panel",
        action="append",
        required=True,
        help="Panel PNG path or output stem; pass in top-to-bottom order.",
    )
    parser.add_argument("--output", required=True, help="Final PNG path.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--gap-px",
        type=int,
        default=24,
        help="White gap between panels in output pixels; defaults to 24.",
    )
    return parser


def main() -> None:
    """Load panels and write their unmodified vertical composition."""
    args = _build_parser().parse_args()
    if len(args.panel) < 2:
        raise ValueError("Pass at least two --panel arguments.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")
    if args.gap_px < 0:
        raise ValueError("--gap-px must be non-negative.")
    panel_paths = [_resolve_png(raw_path) for raw_path in args.panel]
    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    width, height = _stack_panels(
        panel_paths,
        output_path,
        dpi=args.dpi,
        gap_px=args.gap_px,
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "panels": [str(path) for path in panel_paths],
                "width": width,
                "height": height,
                "dpi": args.dpi,
                "gap_px": args.gap_px,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
