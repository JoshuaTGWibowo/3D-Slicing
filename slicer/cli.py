"""Command line interface for the convex slicer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .meniscus import MeniscusParameters
from .model import ModelPreparer
from .slicing import ConvexSlicer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convex slicing MVP")
    parser.add_argument("stl", type=Path, help="Path to the STL model")
    parser.add_argument(
        "output",
        type=Path,
        help="Directory that will contain generated BMP frames",
    )
    parser.add_argument("--pitch", type=float, default=0.05, help="Vertical pitch in mm")
    parser.add_argument(
        "--print-head-diameter",
        type=float,
        default=5.42,
        help="Print head diameter in mm",
    )
    parser.add_argument(
        "--rim-height",
        type=float,
        default=0.75,
        help="Rim starting height in mm",
    )
    parser.add_argument(
        "--contact-angle",
        type=float,
        default=45.0,
        help="Contact angle in degrees",
    )
    parser.add_argument(
        "--surface-tension",
        type=float,
        default=73.0,
        help="Surface tension (mN/m)",
    )
    parser.add_argument("--density", type=float, default=1000.0, help="Density rho (kg/m^3)")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity (m/s^2)")
    parser.add_argument("--bezier-k1", type=float, default=0.25)
    parser.add_argument("--bezier-k2", type=float, default=0.75)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path for a metadata JSON summary",
    )
    return parser


def main(argv: Any | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    params = MeniscusParameters(
        print_head_diameter=args.print_head_diameter,
        rim_height=args.rim_height,
        contact_angle_deg=args.contact_angle,
        surface_tension=args.surface_tension,
        density=args.density,
        gravity=args.gravity,
        bezier_k1=args.bezier_k1,
        bezier_k2=args.bezier_k2,
    )

    preparer = ModelPreparer()
    prepared = preparer.load(args.stl)
    slicer = ConvexSlicer(prepared, params, pitch=args.pitch)
    results = slicer.slice(args.output)

    if args.metadata:
        summary: Dict[str, Any] = {
            "input": str(args.stl.resolve()),
            "output": str(args.output.resolve()),
            "pitch_mm": args.pitch,
            "frame_count": len(results),
            "apex_heights_mm": [r.apex_height for r in results],
            "translation_mm": prepared.translation.tolist(),
            "model_height_mm": prepared.height,
            "meniscus_profile": slicer.meniscus_points().tolist(),
            "capillary_length_m": params.capillary_length,
        }
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
