"""Tests for the critic-trajectory PNG stacker."""

import importlib.util
import sys
from pathlib import Path

from PIL import Image


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "recap"
    / "process"
    / "compose_critic_trajectory_figure.py"
)
SPEC = importlib.util.spec_from_file_location("compose_critic_figure", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stack_two_panels_without_restyling(tmp_path: Path) -> None:
    panel_paths = []
    for index, size in enumerate(((100, 40), (80, 30))):
        panel_path = tmp_path / f"panel{index}.png"
        Image.new("RGB", size, (50 * index, 80, 120)).save(panel_path)
        panel_paths.append(panel_path)

    output = tmp_path / "composite.png"
    dimensions = MODULE._stack_panels(
        panel_paths,
        output,
        dpi=600,
        gap_px=10,
    )

    assert output.is_file()
    assert dimensions == (100, 40 + 38 + 10)
    with Image.open(output) as image:
        assert image.getpixel((5, 5)) == (0, 80, 120)
        assert image.getpixel((5, 55)) == (50, 80, 120)
        assert image.getpixel((5, image.height - 5)) == (50, 80, 120)
