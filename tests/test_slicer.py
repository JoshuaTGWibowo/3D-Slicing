from __future__ import annotations

import json
from pathlib import Path

import pytest

from PIL import Image

import trimesh

from convex_slicer.cli import DEFAULT_PARAMS
from convex_slicer.slicer import ConvexSlicer


def test_slicer_generates_expected_number_of_frames(tmp_path: Path):
    cube = trimesh.creation.box(extents=(1.0, 1.0, 2.0))
    stl_path = tmp_path / "cube.stl"
    cube.export(stl_path)

    slicer = ConvexSlicer(DEFAULT_PARAMS, pitch=0.05)
    output_dir = tmp_path / "frames"
    result = slicer.slice(stl_path, output_dir)

    assert result.num_frames == 44
    first_frame = output_dir / "frame_0000.bmp"
    last_frame = output_dir / "frame_0043.bmp"
    assert first_frame.exists()
    assert last_frame.exists()

    with Image.open(first_frame) as frame:
        assert frame.size == (4096, 2160)
        assert frame.mode == "1"


    metadata_path = output_dir / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["pitch"] == pytest.approx(0.05)
    assert metadata["num_frames"] == 44
    assert metadata["image_width"] == 4096
    assert metadata["image_height"] == 2160
    assert metadata["bit_depth"] == 1
    assert metadata["pixels_per_mm"] == pytest.approx(50.0)
    assert metadata["voxel_size"] == pytest.approx(0.01)
