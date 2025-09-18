"""Core slicing logic."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon

from .parameters import PrintingParameters
from .steady_state import SteadyPhaseProfile, compute_steady_phase_profile

TARGET_WIDTH = 4096
TARGET_HEIGHT = 2160
TARGET_MODE = "L"
SUPERSAMPLE_FACTOR = 4

@dataclass
class SlicingResult:
    """Container for slicing outputs."""

    output_directory: Path
    num_frames: int
    pitch: float
    voxel_size: float


class ConvexSlicer:
    """Generate convex slices for an STL model."""

    def __init__(
        self,
        params: PrintingParameters,
        *,
        pitch: float = 0.05,
        voxel_size: Optional[float] = None,
        profile: Optional[SteadyPhaseProfile] = None,
    ) -> None:
        self.params = params
        self.pitch = float(pitch)
        self.voxel_size = float(voxel_size) if voxel_size is not None else float(pitch)
        self.profile = profile or compute_steady_phase_profile(params)

    def slice(self, stl_path: Path, output_dir: Path) -> SlicingResult:
        """Slice the provided STL model and write image frames."""

        mesh = self._load_mesh(stl_path)
        mesh = self._prepare_mesh(mesh)
        bounds = mesh.bounds

        model_height = bounds[1, 2] - self.params.rim_start_height
        scale = 1.0
        if self.profile.max_height >= model_height and self.profile.max_height > 0:
            scale = 0.8 * model_height / self.profile.max_height

        num_frames = max(int(math.ceil(max(model_height, 0.0) / self.pitch)), 1)

        z_min = self.params.rim_start_height
        z_max = bounds[1, 2]
        if z_max <= z_min:
            z_positions = np.full(num_frames, z_min, dtype=float)
        else:
            centers = z_min + (np.arange(num_frames) + 0.5) * self.pitch
            max_plane = np.nextafter(z_max, z_min)
            z_positions = np.clip(centers, z_min, max_plane)

        sections = list(
            mesh.section_multiplane(
                plane_origin=np.array([0.0, 0.0, z_min]),
                plane_normal=np.array([0.0, 0.0, 1.0]),
                heights=z_positions - z_min,
            )
        )

        transform = _compute_raster_transform(bounds, SUPERSAMPLE_FACTOR)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "pitch": self.pitch,
            "voxel_size": self.voxel_size,
            "num_frames": num_frames,
            "rim_start_height": self.params.rim_start_height,
            "print_head_radius": self.params.print_head_radius,
            "meniscus_scale": scale,
            "control_points": self.profile.control_points.tolist(),

            "image_width": TARGET_WIDTH,
            "image_height": TARGET_HEIGHT,
            "bit_depth": 8,

        }

        for frame, section in enumerate(sections):
            polygons = _extract_polygons(section)
            _save_polygons(output_dir, frame, polygons, transform)

        with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        return SlicingResult(
            output_directory=output_dir,
            num_frames=num_frames,
            pitch=self.pitch,
            voxel_size=self.voxel_size,
        )

    def _load_mesh(self, stl_path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Unsupported mesh type: expected a triangular mesh")
        return mesh

    def _prepare_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh = mesh.copy()
        bounds = mesh.bounds
        center_xy = (bounds[0, :2] + bounds[1, :2]) / 2.0
        translation = np.array([
            -center_xy[0],
            -center_xy[1],
            self.params.rim_start_height - bounds[0, 2],
        ])
        mesh.apply_translation(translation)
        return mesh


def _extract_polygons(section: Optional[trimesh.path.Path2D]) -> list[Polygon]:
    """Return all polygonal regions for a planar section."""

    if section is None:
        return []

    polygons: list[Polygon] = []
    for geom in section.polygons_full:
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(poly for poly in geom.geoms if not poly.is_empty)
    return polygons


@dataclass(frozen=True)
class _RasterTransform:
    """Affine transform from model coordinates to raster space."""

    scale: float
    offset_x: float
    offset_y: float
    min_x: float
    max_y: float
    canvas_width: int
    canvas_height: int

    def apply(self, coords: Iterable[Iterable[float]]) -> list[tuple[float, float]]:
        """Map a coordinate sequence to raster pixel space."""

        array = np.asarray(coords, dtype=float)
        if array.ndim != 2 or array.shape[1] < 2:
            return []
        x = (array[:, 0] - self.min_x) * self.scale + self.offset_x
        y = (self.max_y - array[:, 1]) * self.scale + self.offset_y
        return list(map(tuple, np.stack((x, y), axis=1)))


def _compute_raster_transform(bounds: np.ndarray, supersample: int) -> _RasterTransform:
    """Create a transform that fits the mesh bounds onto the target canvas."""

    min_x = float(bounds[0, 0])
    max_y = float(bounds[1, 1])
    width = float(bounds[1, 0] - bounds[0, 0])
    height = float(bounds[1, 1] - bounds[0, 1])

    canvas_width = max(int(round(TARGET_WIDTH * supersample)), 1)
    canvas_height = max(int(round(TARGET_HEIGHT * supersample)), 1)

    effective_width = width if width > 0 else 1.0
    effective_height = height if height > 0 else 1.0

    scale = min(canvas_width / effective_width, canvas_height / effective_height)
    offset_x = (canvas_width - width * scale) / 2.0 if width > 0 else canvas_width / 2.0
    offset_y = (canvas_height - height * scale) / 2.0 if height > 0 else canvas_height / 2.0

    return _RasterTransform(
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        min_x=min_x,
        max_y=max_y,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


def _save_polygons(
    output_dir: Path,
    index: int,
    polygons: Sequence[Polygon],
    transform: _RasterTransform,
) -> None:
    """Render the slice polygons to disk as an 8-bit grayscale BMP."""

    frame = _render_polygons(polygons, transform)
    frame.save(output_dir / f"frame_{index:04d}.bmp", format="BMP")


def _render_polygons(polygons: Sequence[Polygon], transform: _RasterTransform) -> Image.Image:
    """Rasterize a collection of polygons with supersampling."""

    canvas = Image.new(TARGET_MODE, (transform.canvas_width, transform.canvas_height), color=0)
    draw = ImageDraw.Draw(canvas)

    for polygon in polygons:
        exterior = transform.apply(polygon.exterior.coords)
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=255)
        for interior in polygon.interiors:
            hole = transform.apply(interior.coords)
            if len(hole) >= 3:
                draw.polygon(hole, fill=0)

    if transform.canvas_width == TARGET_WIDTH and transform.canvas_height == TARGET_HEIGHT:
        return canvas
    return canvas.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.LANCZOS)
