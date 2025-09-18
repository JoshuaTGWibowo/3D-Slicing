"""Core slicing logic."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, Polygon
from shapely.geometry.base import BaseGeometry

from .parameters import PrintingParameters
from .steady_state import SteadyPhaseProfile, compute_steady_phase_profile

TARGET_WIDTH = 4096
TARGET_HEIGHT = 2160
TARGET_MODE = "L"
SUPERSAMPLE_FACTOR = 2

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
        meniscus_scale, height_fn = self._meniscus_height_fn(mesh)
        warped_mesh = _warp_mesh(mesh, height_fn)
        planar_transform = _planar_transform(mesh, TARGET_WIDTH, TARGET_HEIGHT)

        warped_bounds = warped_mesh.bounds
        z_min, z_max = warped_bounds[:, 2]
        model_height = max(z_max - z_min, 0.0)
        num_frames = max(int(math.ceil(model_height / self.pitch)), 1)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "pitch": self.pitch,
            "voxel_size": self.voxel_size,
            "num_frames": num_frames,
            "rim_start_height": self.params.rim_start_height,
            "print_head_radius": self.params.print_head_radius,
            "meniscus_scale": meniscus_scale,
            "control_points": self.profile.control_points.tolist(),

            "image_width": TARGET_WIDTH,
            "image_height": TARGET_HEIGHT,
            "bit_depth": 8,

        }

        for frame in range(num_frames):
            if model_height <= 1e-6:
                plane_z = z_min
            else:
                plane_z = min(z_min + (frame + 0.5) * self.pitch, z_max - 1e-6)
            polygons = _slice_polygons(warped_mesh, plane_z)
            image = _rasterize_polygons(polygons, planar_transform)
            _save_frame(output_dir, frame, image)

        with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        return SlicingResult(
            output_directory=output_dir,
            num_frames=num_frames,
            pitch=self.pitch,
            voxel_size=self.voxel_size,
        )

    def _meniscus_height_fn(self, mesh: trimesh.Trimesh) -> tuple[float, Callable[[np.ndarray], np.ndarray]]:
        """Return a callable providing the meniscus height at ``(x, y)``."""

        model_height = mesh.bounds[1, 2] - self.params.rim_start_height
        scale = 1.0
        if self.profile.max_height >= model_height and self.profile.max_height > 0:
            scale = 0.8 * model_height / self.profile.max_height

        def height_fn(xy: np.ndarray) -> np.ndarray:
            points = np.asarray(xy, dtype=float)
            if points.ndim == 1:
                points = points[np.newaxis, :]
            radii = np.linalg.norm(points[:, :2], axis=1)
            offsets = self.profile.height(radii) * scale
            return self.params.rim_start_height + offsets

        return scale, height_fn

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


def _warp_mesh(mesh: trimesh.Trimesh, height_fn: Callable[[np.ndarray], np.ndarray]) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` warped so the meniscus surface becomes planar."""

    vertices = mesh.vertices.copy()
    base_heights = height_fn(vertices[:, :2])
    vertices[:, 2] = vertices[:, 2] - base_heights
    warped = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
    return warped


@dataclass(frozen=True)
class PlanarTransform:
    """Map XY coordinates from mesh space to raster space."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    scale: float
    x_margin: float
    y_margin: float

    def map_points(self, points: np.ndarray, *, supersample: int = 1) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[np.newaxis, :]
        s = self.scale * supersample
        x_m = self.x_margin * supersample
        y_m = self.y_margin * supersample
        px = (points[:, 0] - self.min_x) * s + x_m
        py = (self.max_y - points[:, 1]) * s + y_m
        return np.column_stack([px, py])


def _planar_transform(mesh: trimesh.Trimesh, width: int, height: int) -> PlanarTransform:
    bounds = mesh.bounds[:, :2]
    min_x, min_y = bounds[0]
    max_x, max_y = bounds[1]
    width_world = max(max_x - min_x, 1e-9)
    height_world = max(max_y - min_y, 1e-9)
    scale_x = width / width_world if width_world > 0 else 1.0
    scale_y = height / height_world if height_world > 0 else 1.0
    scale = min(scale_x, scale_y)
    x_margin = (width - width_world * scale) / 2.0
    y_margin = (height - height_world * scale) / 2.0
    return PlanarTransform(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        scale=scale,
        x_margin=x_margin,
        y_margin=y_margin,
    )


def _slice_polygons(mesh: trimesh.Trimesh, plane_z: float) -> list[Polygon]:
    """Compute polygonal cross-sections of ``mesh`` at ``plane_z``."""

    section = mesh.section(plane_origin=[0.0, 0.0, plane_z], plane_normal=[0.0, 0.0, 1.0])
    if section is None or len(section.entities) == 0:
        return []

    path2d, _ = section.to_2D()
    geometries = path2d.polygons_full
    if not geometries:
        return []

    if isinstance(geometries, (list, tuple)):
        geoms: Sequence[BaseGeometry] = tuple(geometries)
    elif isinstance(geometries, BaseGeometry):
        geoms = (geometries,)
    else:
        geoms = ()

    polygons: list[Polygon] = []
    for geom in geoms:
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, GeometryCollection):
            for sub in geom.geoms:
                if isinstance(sub, Polygon) and not sub.is_empty:
                    polygons.append(sub)
        else:
            try:
                for sub in geom.geoms:  # type: ignore[attr-defined]
                    if isinstance(sub, Polygon) and not sub.is_empty:
                        polygons.append(sub)
            except AttributeError:  # pragma: no cover - defensive fallback
                continue

    return polygons


def _rasterize_polygons(polygons: Sequence[Polygon], transform: PlanarTransform) -> Image.Image:
    """Rasterize ``polygons`` into an antialiased 8-bit image."""

    if not polygons:
        return Image.new(TARGET_MODE, (TARGET_WIDTH, TARGET_HEIGHT), color=0)

    supersample = SUPERSAMPLE_FACTOR
    high_res_size = (TARGET_WIDTH * supersample, TARGET_HEIGHT * supersample)
    canvas = Image.new(TARGET_MODE, high_res_size, color=0)
    draw = ImageDraw.Draw(canvas)

    for polygon in polygons:
        exterior = transform.map_points(np.asarray(polygon.exterior.coords), supersample=supersample)
        draw.polygon([tuple(pt) for pt in exterior], fill=255)
        for interior in polygon.interiors:
            hole = transform.map_points(np.asarray(interior.coords), supersample=supersample)
            draw.polygon([tuple(pt) for pt in hole], fill=0)

    return canvas.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.LANCZOS)


def _save_frame(output_dir: Path, index: int, image: Image.Image) -> None:
    """Write a rasterized frame to disk."""

    image.save(output_dir / f"frame_{index:04d}.bmp", format="BMP")
