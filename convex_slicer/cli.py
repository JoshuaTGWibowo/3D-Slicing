"""Command line interface for the convex slicer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .parameters import PrintingParameters
from .slicer import ConvexSlicer, DEFAULT_VOXEL_SIZE


DEFAULT_PARAMS = PrintingParameters(
    print_head_diameter=5.42,
    rim_start_height=0.75,
    contact_angle_deg=45.0,
    surface_tension=73.0,
    density=1000.0,
    gravity=9.81,
    bezier_k1=0.25,
    bezier_k2=0.75,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convex slicer MVP")
    parser.add_argument("stl", type=Path, help="Path to the input STL model")
    parser.add_argument(
        "output",
        type=Path,
        help="Directory where the slice frames will be written",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.05,
        help="Vertical pitch between slices in millimetres (default: 0.05)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=DEFAULT_VOXEL_SIZE,
        help=(
            "Override the voxel size used for rasterisation in millimetres "
            f"(default: {DEFAULT_VOXEL_SIZE})"
        ),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default printing parameters",
    )
    return parser


def load_parameters(path: Path | None) -> PrintingParameters:
    if path is None:
        return DEFAULT_PARAMS
    with Path(path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return PrintingParameters(**data)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    params = load_parameters(args.params)
    slicer = ConvexSlicer(params, pitch=args.pitch, voxel_size=args.voxel_size)
    slicer.slice(args.stl, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
