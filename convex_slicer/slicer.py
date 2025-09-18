"""Core slicing logic."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import trimesh
from shapely.geometry import Polygon

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


@dataclass
class CanvasTransform:
    """Affine transform between XY coordinates and the 4K canvas."""

    xy_min: np.ndarray
    xy_max: np.ndarray
    scale: float
    offset_x: float
    offset_y: float

    @classmethod
    def from_bounds(cls, bounds: np.ndarray) -> "CanvasTransform":
        xy_min = bounds[0].astype(float)
        xy_max = bounds[1].astype(float)
        raw_extent = xy_max - xy_min
        extent = np.where(raw_extent > 0, raw_extent, 1.0)
        width_scale = TARGET_WIDTH / extent[0] if extent[0] > 0 else float("inf")
        height_scale = TARGET_HEIGHT / extent[1] if extent[1] > 0 else float("inf")
        scale = min(width_scale, height_scale)
        scaled_width = extent[0] * scale
        scaled_height = extent[1] * scale
        offset_x = (TARGET_WIDTH - scaled_width) / 2.0
        offset_y = (TARGET_HEIGHT - scaled_height) / 2.0
        return cls(xy_min=xy_min, xy_max=xy_max, scale=scale, offset_x=offset_x, offset_y=offset_y)


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

        model_height = mesh.bounds[1, 2] - self.params.rim_start_height
        scale = 1.0
        if self.profile.max_height >= model_height and self.profile.max_height > 0:
            scale = 0.8 * model_height / self.profile.max_height

        warped_mesh = _warp_mesh(mesh, self.params, self.profile, scale)
        transform = CanvasTransform.from_bounds(mesh.bounds[:, :2])

        num_frames = max(int(math.ceil(model_height / self.pitch)), 1)

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

        for frame in range(num_frames):
            plane_z = (frame + 0.5) * self.pitch
            polygons = _cross_section_polygons(warped_mesh, plane_z)
            image = _render_polygons(polygons, transform)
            image.save(output_dir / f"frame_{frame:04d}.bmp", format="BMP")

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


def _warp_mesh(
    mesh: trimesh.Trimesh,
    params: PrintingParameters,
    profile: SteadyPhaseProfile,
    scale: float,
) -> trimesh.Trimesh:
    """Warp the mesh so that convex slices become planar in ``z``."""

    vertices = mesh.vertices.copy()
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    meniscus_offsets = profile.height(radii) * scale
    total_offsets = params.rim_start_height + meniscus_offsets
    warped_vertices = vertices.copy()
    warped_vertices[:, 2] = vertices[:, 2] - total_offsets
    return trimesh.Trimesh(vertices=warped_vertices, faces=mesh.faces, process=False)


def _cross_section_polygons(mesh: trimesh.Trimesh, height: float) -> list[Polygon]:
    """Intersect ``mesh`` with the horizontal plane at ``height``."""

    if height < 0:
        return []

    section = mesh.section(plane_origin=[0.0, 0.0, float(height)], plane_normal=[0.0, 0.0, 1.0])
    if section is None or not section.entities:
        return []

    if hasattr(section, "to_2D"):
        planar, _ = section.to_2D()
    else:  # pragma: no cover - compatibility with older trimesh versions
        planar, _ = section.to_planar()
    polygons = [poly for poly in planar.polygons_full if not poly.is_empty]
    return polygons


def _render_polygons(polygons: Sequence[Polygon], transform: CanvasTransform) -> "Image.Image":
    """Rasterise polygons onto the 4K canvas with supersampling."""

    from PIL import Image, ImageDraw

    if not polygons:
        return Image.new(TARGET_MODE, (TARGET_WIDTH, TARGET_HEIGHT), color=0)

    oversample = max(int(SUPERSAMPLE_FACTOR), 1)
    canvas_size = (TARGET_WIDTH * oversample, TARGET_HEIGHT * oversample)
    canvas = Image.new(TARGET_MODE, canvas_size, color=0)
    draw = ImageDraw.Draw(canvas)

    scale = transform.scale * oversample
    offset_x = transform.offset_x * oversample
    offset_y = transform.offset_y * oversample
    xy_min = transform.xy_min
    xy_max = transform.xy_max

    def project(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for x, y in points:
            px = (x - xy_min[0]) * scale + offset_x
            py = (xy_max[1] - y) * scale + offset_y
            coords.append((px, py))
        return coords

    for polygon in polygons:
        if polygon.is_empty:
            continue
        exterior = project(polygon.exterior.coords)
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=255)
        for interior in polygon.interiors:
            hole = project(interior.coords)
            if len(hole) >= 3:
                draw.polygon(hole, fill=0)

    if oversample > 1:
        resampling = getattr(Image, "Resampling", Image)
        canvas = canvas.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=resampling.LANCZOS)

    return canvas
