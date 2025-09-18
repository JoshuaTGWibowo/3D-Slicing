from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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

    assert result.num_frames == 40
    first_frame = output_dir / "frame_0000.bmp"
    last_frame = output_dir / "frame_0039.bmp"
    assert first_frame.exists()
    assert last_frame.exists()

    with Image.open(first_frame) as frame:
        assert frame.size == (4096, 2160)
        assert frame.mode == "RGB"
        first_array = np.asarray(frame)

    with Image.open(last_frame) as frame:
        assert frame.mode == "RGB"
        last_array = np.asarray(frame)

    first_nonzero = first_array.reshape(-1, 3)[np.any(first_array.reshape(-1, 3) != 0, axis=1)]
    last_nonzero = last_array.reshape(-1, 3)[np.any(last_array.reshape(-1, 3) != 0, axis=1)]
    assert first_nonzero.size > 0
    assert last_nonzero.size > 0
    assert last_nonzero.mean() > first_nonzero.mean()


    metadata_path = output_dir / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["pitch"] == pytest.approx(0.05)
    assert metadata["num_frames"] == 40
    assert metadata["image_width"] == 4096
    assert metadata["image_height"] == 2160
    assert metadata["bit_depth"] == 8
    assert metadata["color_mode"] == "RGB"
