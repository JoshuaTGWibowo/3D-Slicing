"""Core slicing logic."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh

from .parameters import PrintingParameters
from .steady_state import SteadyPhaseProfile, compute_steady_phase_profile

TARGET_WIDTH = 4096
TARGET_HEIGHT = 2160
TARGET_MODE = "RGB"

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
        voxel_grid = mesh.voxelized(self.voxel_size).fill()
        occupancy = voxel_grid.matrix.astype(bool)
        transform = voxel_grid.transform
        x_coords, y_coords, z_coords = _axis_coordinates(transform, occupancy.shape)

        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij", sparse=False)
        radii = np.sqrt(xx**2 + yy**2)
        surface_offsets = self.profile.height(radii)

        model_height = mesh.bounds[1, 2] - self.params.rim_start_height
        scale = 1.0
        if self.profile.max_height >= model_height and self.profile.max_height > 0:
            scale = 0.8 * model_height / self.profile.max_height
            surface_offsets *= scale

        base_surface = self.params.rim_start_height + surface_offsets

        num_frames = max(int(math.ceil(model_height / self.pitch)), 1)
        z_coords = np.asarray(z_coords)

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
            "color_mode": TARGET_MODE,

        }

        min_height = float(self.params.rim_start_height)
        max_height = float(self.params.rim_start_height + model_height)

        for frame in range(num_frames):
            lower = base_surface + frame * self.pitch
            upper = lower + self.pitch
            mask, height_map = _slice_height_map(occupancy, z_coords, lower, upper)
            _save_frame(
                output_dir,
                frame,
                mask,
                height_map,
                min_height=min_height,
                max_height=max_height,
            )

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


def _axis_coordinates(transform: np.ndarray, shape: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute coordinate arrays for the voxel grid axes."""

    shape = tuple(int(v) for v in shape)
    origin = transform @ np.array([0.0, 0.0, 0.0, 1.0])
    axis_vectors = (
        transform @ np.array([1.0, 0.0, 0.0, 0.0]),
        transform @ np.array([0.0, 1.0, 0.0, 0.0]),
        transform @ np.array([0.0, 0.0, 1.0, 0.0]),
    )
    x_coords = origin[0] + axis_vectors[0][0] * np.arange(shape[0])
    y_coords = origin[1] + axis_vectors[1][1] * np.arange(shape[1])
    z_coords = origin[2] + axis_vectors[2][2] * np.arange(shape[2])
    return x_coords, y_coords, z_coords


def _slice_height_map(
    occupancy: np.ndarray,
    z_coords: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a mask and height map for a slice between ``lower`` and ``upper`` surfaces."""

    z_grid = z_coords[np.newaxis, np.newaxis, :]
    within = (z_grid >= lower[..., np.newaxis]) & (z_grid < upper[..., np.newaxis])
    hits = occupancy & within
    if not np.any(hits):
        mask = np.zeros(lower.shape, dtype=bool)
        heights = np.zeros(lower.shape, dtype=float)
        return mask, heights

    height_candidates = np.where(hits, z_grid, -np.inf)
    heights = height_candidates.max(axis=2)
    mask = np.isfinite(heights)
    heights[~mask] = 0.0
    return mask, heights


def _save_frame(
    output_dir: Path,
    index: int,
    mask: np.ndarray,
    heights: np.ndarray,
    *,
    min_height: float,
    max_height: float,
) -> None:
    """Write the height map as an RGB BMP image with 4K DCI resolution."""

    frame = _render_frame(mask, heights, min_height=min_height, max_height=max_height)
    frame.save(output_dir / f"frame_{index:04d}.bmp", format="BMP")


def _render_frame(
    mask: np.ndarray,
    heights: np.ndarray,
    *,
    min_height: float,
    max_height: float,
) -> "Image.Image":
    """Project the height map onto the 4K target canvas using a color gradient."""

    from PIL import Image

    valid = mask.astype(bool)
    dynamic_range = max(max_height - min_height, 1e-9)
    normalized = np.zeros_like(heights, dtype=float)
    normalized[valid] = (heights[valid] - min_height) / dynamic_range
    normalized = np.clip(normalized, 0.0, 1.0)

    colorized = _apply_colormap(normalized)
    colorized[~valid] = 0

    array = colorized.transpose(1, 0, 2)[::-1, :, :]
    base_image = Image.fromarray(array)
    if base_image.size == (TARGET_WIDTH, TARGET_HEIGHT):
        return base_image

    width_scale = TARGET_WIDTH / base_image.width if base_image.width else 1.0
    height_scale = TARGET_HEIGHT / base_image.height if base_image.height else 1.0
    scale = min(width_scale, height_scale)
    scaled_width = max(1, min(TARGET_WIDTH, int(round(base_image.width * scale))))
    scaled_height = max(1, min(TARGET_HEIGHT, int(round(base_image.height * scale))))

    resized = base_image.resize((scaled_width, scaled_height), resample=Image.BILINEAR)
    canvas = Image.new(TARGET_MODE, (TARGET_WIDTH, TARGET_HEIGHT), color=(0, 0, 0))
    left = (TARGET_WIDTH - scaled_width) // 2
    top = (TARGET_HEIGHT - scaled_height) // 2
    canvas.paste(resized, (left, top))
    return canvas

def _apply_colormap(normalized: np.ndarray) -> np.ndarray:
    """Map normalized values in [0, 1] to an RGB gradient."""

    colormap = np.array(
        [
            [0.267004, 0.004874, 0.329415],
            [0.282327, 0.094955, 0.417331],
            [0.253935, 0.265254, 0.529983],
            [0.163625, 0.471133, 0.558148],
            [0.134692, 0.658636, 0.517649],
            [0.477504, 0.821444, 0.318195],
            [0.993248, 0.906157, 0.143936],
        ]
    )
    positions = np.linspace(0.0, 1.0, colormap.shape[0])

    flat = normalized.reshape(-1)
    red = np.interp(flat, positions, colormap[:, 0])
    green = np.interp(flat, positions, colormap[:, 1])
    blue = np.interp(flat, positions, colormap[:, 2])

    rgb = np.stack([red, green, blue], axis=1)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return rgb.reshape(normalized.shape + (3,))
