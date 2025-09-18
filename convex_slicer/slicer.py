"""Core slicing logic."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from .parameters import PrintingParameters
from .steady_state import SteadyPhaseProfile, compute_steady_phase_profile

TARGET_WIDTH = 4096
TARGET_HEIGHT = 2160
TARGET_MODE = "L"
SUPERSAMPLE = 2


@dataclass
class SlicingResult:
    """Container for slicing outputs."""

    output_directory: Path
    num_frames: int
    pitch: float
    voxel_size: float


@dataclass(frozen=True)
class ProjectionParameters:
    """Parameters describing the projection of model space onto the image plane."""

    scale: float
    center_x: float
    center_y: float
    supersample: int = SUPERSAMPLE

    @property
    def canvas_width(self) -> int:
        return TARGET_WIDTH * self.supersample

    @property
    def canvas_height(self) -> int:
        return TARGET_HEIGHT * self.supersample


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
        meniscus_scale = 1.0
        if self.profile.max_height >= model_height and self.profile.max_height > 0:
            meniscus_scale = 0.8 * model_height / self.profile.max_height

        flattened_mesh = _flatten_mesh(mesh, self.profile, meniscus_scale, self.params.rim_start_height)
        num_frames = max(int(math.ceil(model_height / self.pitch)), 1)

        projection = _compute_projection(mesh)

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
            plane_height = frame * self.pitch
            geometry = _layer_geometry(flattened_mesh, plane_height)
            image = _rasterize_geometry(geometry, projection)
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


def _flatten_mesh(
    mesh: trimesh.Trimesh,
    profile: SteadyPhaseProfile,
    scale: float,
    rim_start_height: float,
) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` with the meniscus profile flattened."""

    flattened = mesh.copy()
    vertices = flattened.vertices.copy()
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    offsets = profile.height(radii) * scale + rim_start_height
    vertices[:, 2] = vertices[:, 2] - offsets
    flattened.vertices = vertices
    return flattened


def _compute_projection(mesh: trimesh.Trimesh) -> ProjectionParameters:
    """Compute projection parameters to map model coordinates to pixels."""

    bounds = mesh.bounds
    width = max(bounds[1, 0] - bounds[0, 0], 1e-6)
    height = max(bounds[1, 1] - bounds[0, 1], 1e-6)
    center_x = (bounds[0, 0] + bounds[1, 0]) / 2.0
    center_y = (bounds[0, 1] + bounds[1, 1]) / 2.0

    scale_x = (TARGET_WIDTH * SUPERSAMPLE) / width
    scale_y = (TARGET_HEIGHT * SUPERSAMPLE) / height
    scale = min(scale_x, scale_y)

    return ProjectionParameters(scale=scale, center_x=center_x, center_y=center_y)


def _layer_geometry(mesh: trimesh.Trimesh, plane_height: float):
    """Intersect ``mesh`` with a horizontal plane at ``plane_height``."""

    if mesh.is_empty:
        return None

    section = mesh.section(
        plane_origin=[0.0, 0.0, plane_height],
        plane_normal=[0.0, 0.0, 1.0],
    )
    if section is None or not section.entities:
        return None

    lines: list[LineString] = []
    for coords in section.discrete:
        if len(coords) < 3:
            continue
        xy = coords[:, :2]
        line = LineString(xy)
        if line.length > 0:
            lines.append(line)

    if not lines:
        return None

    raw_polygons = [poly for poly in polygonize(lines) if poly.area > 0.0]
    cleaned_polygons: list[Polygon] = []
    for poly in raw_polygons:
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area <= 1e-10:
            continue
        cleaned_polygons.append(poly)

    if not cleaned_polygons:
        return None

    parents = [None] * len(cleaned_polygons)
    areas = [poly.area for poly in cleaned_polygons]
    for i, poly_i in enumerate(cleaned_polygons):
        parent_index = None
        parent_area = float("inf")
        for j, poly_j in enumerate(cleaned_polygons):
            if i == j:
                continue
            if areas[j] <= areas[i]:
                continue
            if poly_j.contains(poly_i) and areas[j] < parent_area:
                parent_index = j
                parent_area = areas[j]
        parents[i] = parent_index

    children: dict[int, list[int]] = {i: [] for i in range(len(cleaned_polygons))}
    for child, parent in enumerate(parents):
        if parent is not None:
            children[parent].append(child)

    depths = [0] * len(cleaned_polygons)
    for i in range(len(cleaned_polygons)):
        depth = 0
        parent = parents[i]
        while parent is not None:
            depth += 1
            parent = parents[parent]
        depths[i] = depth

    positive_regions = []
    for i, polygon in enumerate(cleaned_polygons):
        if depths[i] % 2 != 0:
            continue
        hole_indices = [child for child in children[i] if depths[child] == depths[i] + 1]
        if hole_indices:
            holes = unary_union([cleaned_polygons[h] for h in hole_indices])
            polygon = polygon.difference(holes)
        if not polygon.is_empty:
            positive_regions.append(polygon)

    if not positive_regions:
        return None

    return unary_union(positive_regions)


def _rasterize_geometry(geometry, projection: ProjectionParameters) -> Image.Image:
    """Rasterise the layer geometry onto the supersampled canvas."""

    canvas = Image.new(TARGET_MODE, (projection.canvas_width, projection.canvas_height), color=0)
    if geometry is None or geometry.is_empty:
        return canvas.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.LANCZOS)

    draw = ImageDraw.Draw(canvas)

    for polygon in _iter_polygons(geometry):
        exterior = _project_ring(polygon.exterior.coords, projection)
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=255)
        for interior in polygon.interiors:
            hole = _project_ring(interior.coords, projection)
            if len(hole) >= 3:
                draw.polygon(hole, fill=0)

    if projection.supersample > 1:
        canvas = canvas.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.LANCZOS)
    return canvas


def _project_ring(coords: Iterable[tuple[float, float]], projection: ProjectionParameters) -> list[tuple[float, float]]:
    """Project planar coordinates to pixel coordinates."""

    coords_array = np.asarray(coords, dtype=float)
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        return []
    x = (coords_array[:, 0] - projection.center_x) * projection.scale + projection.canvas_width / 2.0
    y = (projection.center_y - coords_array[:, 1]) * projection.scale + projection.canvas_height / 2.0
    return [(float(px), float(py)) for px, py in zip(x, y)]


def _iter_polygons(geometry) -> Iterable[Polygon]:
    """Yield polygons from an arbitrary Shapely geometry."""

    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry.geoms if not poly.is_empty]
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for geom in geometry.geoms:
            polygons.extend(_iter_polygons(geom))
        return polygons
    return []
