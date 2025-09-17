"""Command line interface for convex slicer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

from .bmp import save_bitmap
from .meniscus import MeniscusParameters, compute_meniscus
from .slicing import SliceParameters, SliceVolume, build_grid
from .stl import center_mesh, compute_bounds, load_stl

DEFAULT_PITCH = 0.05  # mm


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convex slicer MVP")
    parser.add_argument("stl", type=Path, help="Input STL model")
    parser.add_argument("output", type=Path, help="Output directory for BMP frames")
    parser.add_argument("--pitch", type=float, default=DEFAULT_PITCH, help="Layer pitch in mm")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to write slicing metadata JSON",
    )
    return parser.parse_args(argv)


def run(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    params = MeniscusParameters(
        print_head_diameter=5.42,
        rim_start_height=0.75,
        contact_angle_deg=45.0,
        surface_tension=73.0,
        density=1000.0,
        gravity=9.81,
        bezier_k1=0.25,
        bezier_k2=0.75,
    )

    meniscus = compute_meniscus(params)

    triangles = load_stl(str(args.stl))
    triangles = center_mesh(triangles)
    bounds = compute_bounds(triangles)
    model_height = bounds[1][2] - bounds[0][2]

    if model_height <= 0:
        raise ValueError("Model height is zero after centering")

    radius = meniscus.radius
    max_radius = max(
        abs(bounds[0][0]),
        abs(bounds[1][0]),
        abs(bounds[0][1]),
        abs(bounds[1][1]),
    )
    if max_radius > radius:
        raise ValueError(
            f"Model exceeds print head radius: model radius {max_radius:.3f} mm vs {radius:.3f} mm"
        )

    voxel = float(args.pitch)
    grid = build_grid(triangles, voxel=voxel, radius=radius)
    slice_params = SliceParameters(pitch=voxel, meniscus=meniscus)
    volume = SliceVolume(grid, slice_params)

    num_frames = volume.num_frames(model_height)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(num_frames):
        mask = volume.frame_mask(idx, model_height)
        save_bitmap(output_dir / f"frame_{idx:04d}.bmp", mask)

    if args.metadata:
        metadata = {
            "pitch": voxel,
            "num_frames": num_frames,
            "model_height": model_height,
            "meniscus": meniscus.as_dict(),
            "bounds": bounds,
        }
        args.metadata.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
