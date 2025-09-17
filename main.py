"""Command line interface for the convex slicing MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

from convex_slicer import ConvexSlicer, PrintParameters


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convex slicer MVP")
    parser.add_argument("stl", type=Path, help="Path to the STL file to slice")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where the generated bitmaps will be written",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.05,
        help="In-plane sampling resolution in millimetres",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.05,
        help="Vertical pitch between successive slices in millimetres",
    )
    parser.add_argument(
        "--print-head-diameter",
        type=float,
        default=5.42,
        help="Print head diameter in millimetres",
    )
    parser.add_argument(
        "--rim-start-height",
        type=float,
        default=0.75,
        help="Initial rim height relative to the base in millimetres",
    )
    parser.add_argument(
        "--contact-angle",
        type=float,
        default=45.0,
        help="Steady-state contact angle in degrees",
    )
    parser.add_argument(
        "--surface-tension",
        type=float,
        default=73.0,
        help="Surface tension in mN/m",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1000.0,
        help="Fluid density in kg/m^3",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="Acceleration due to gravity in m/s^2",
    )
    parser.add_argument(
        "--bezier-k1",
        type=float,
        default=0.25,
        help="First Bézier control weighting",
    )
    parser.add_argument(
        "--bezier-k2",
        type=float,
        default=0.75,
        help="Second Bézier control weighting",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    parameters = PrintParameters(
        print_head_diameter=args.print_head_diameter,
        rim_start_height=args.rim_start_height,
        contact_angle_deg=args.contact_angle,
        surface_tension_mN_m=args.surface_tension,
        density=args.density,
        gravity=args.gravity,
        bezier_k1=args.bezier_k1,
        bezier_k2=args.bezier_k2,
        pitch=args.pitch,
    )

    slicer = ConvexSlicer(parameters=parameters, pixel_size=args.pixel_size)
    slicer.slice(args.stl, args.output)


if __name__ == "__main__":
    main()
