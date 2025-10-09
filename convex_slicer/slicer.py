
"""Core slicing logic with gradient-colored output (interface/height mapped to RGB)."""

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
TARGET_MODE = "RGB"  # changed from "L": we now emit 24‑bit color BMPs


@dataclass
class SlicingResult:
    """Container for slicing outputs."""

    output_directory: Path
    num_frames: int
    pitch: float
    voxel_size: float


class ConvexSlicer:
    """Generate convex slices for an STL model with gradient colors."""

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

        meniscus_peak = float(surface_offsets.max())  # how much higher the rim is than the center
        base_surface = self.params.rim_start_height + surface_offsets - meniscus_peak

        total_travel = model_height + meniscus_peak
        num_frames = max(int(math.ceil(total_travel / self.pitch)), 1)
        z_coords = np.asarray(z_coords)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Choose a global color scale so colors are comparable across frames.
        # We encode the *absolute* interface height being printed at each (x,y).
        vmin = float(base_surface.min())
        vmax = float((base_surface + total_travel).max())
        vspan = max(vmax - vmin, 1e-6)

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
            "bit_depth": 24,
            "color_mode": TARGET_MODE,
            "color_scale_min": vmin,
            "color_scale_max": vmax,
            "colormap": "viridis-like",
        }

        for frame in range(num_frames):
            lower = base_surface + frame * self.pitch
            upper = lower + self.pitch
            height_map, mask = _slice_height_map(occupancy, z_coords, lower, upper)
            # Normalise to [0, 1] using the global range
            norm = np.clip((height_map - vmin) / vspan, 0.0, 1.0)
            rgb = _apply_colormap(norm)
            # Zero-out background
            rgb[~mask] = 0
            _save_rgb_frame(output_dir, frame, rgb)

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
    """Return (height_map, mask) for a slice band.
    height_map contains the *maximum* z coordinate of occupied voxels within the band,
    per (x, y). mask is True where the band intersects the model.
    """
    z_grid = z_coords[np.newaxis, np.newaxis, :]
    within = (z_grid >= lower[..., np.newaxis]) & (z_grid < upper[..., np.newaxis])
    hits = occupancy & within

    # Use NaNs to compute per-(x,y) maxima only where we have hits
    with np.errstate(invalid="ignore"):
        z_vals = np.where(hits, z_grid, np.nan)
        height_map = np.nanmax(z_vals, axis=2)

    mask = np.isfinite(height_map)
    height_map[~mask] = 0.0
    return height_map.astype(np.float32), mask


# ---------- Color mapping utilities ----------

def _build_viridis_like_lut() -> np.ndarray:
    """Construct a small viridis-like 256x3 LUT (uint8)."""
    # Key color stops (t, r, g, b)
    stops = np.array([
        [0.00, 68,   1,  84],   # deep purple
        [0.25, 59,  82, 139],   # indigo
        [0.50, 33, 145, 140],   # teal
        [0.75, 94, 201,  98],   # green
        [1.00, 253, 231, 37],   # yellow
    ], dtype=float)

    lut = np.zeros((256, 3), dtype=np.uint8)
    ts = stops[:, 0]
    colors = stops[:, 1:4]
    for i in range(256):
        t = i / 255.0
        j = np.searchsorted(ts, t, side="right") - 1
        j = np.clip(j, 0, len(ts) - 2)
        t0, t1 = ts[j], ts[j + 1]
        c0, c1 = colors[j], colors[j + 1]
        u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        c = (1 - u) * c0 + u * c1
        lut[i] = np.clip(np.round(c), 0, 255).astype(np.uint8)
    return lut


_VIRIDIS_LUT = _build_viridis_like_lut()


def _apply_colormap(norm: np.ndarray) -> np.ndarray:
    """Map a [0,1] float array to RGB using a viridis-like LUT."""
    idx = np.clip((norm * 255.0).astype(np.uint8), 0, 255)
    return _VIRIDIS_LUT[idx]


# ---------- Rendering ----------

def _save_rgb_frame(output_dir: Path, index: int, rgb: np.ndarray) -> None:
    """Write the RGB frame as a 24-bit BMP image with 4K DCI resolution."""
    frame = _render_rgb(rgb)
    frame.save(output_dir / f"frame_{index:04d}.bmp", format="BMP")


def _render_rgb(rgb: np.ndarray) -> "Image.Image":
    """Project the RGB array (H=W=voxel grid) onto the 4K target canvas.

    The orientation mirrors the previous grayscale pipeline: transpose then
    flip vertically so on-screen matches the conventional top‑down view.
    """
    from PIL import Image

    # rgb is (X, Y, 3) – match legacy orientation
    array = rgb.transpose(1, 0, 2)[::-1, :, :]
    base_image = Image.fromarray(array, mode=TARGET_MODE)

    if base_image.size == (TARGET_WIDTH, TARGET_HEIGHT):
        return base_image

    width_scale = TARGET_WIDTH / base_image.width if base_image.width else 1.0
    height_scale = TARGET_HEIGHT / base_image.height if base_image.height else 1.0
    scale = min(width_scale, height_scale)
    scaled_width = max(1, min(TARGET_WIDTH, int(round(base_image.width * scale))))
    scaled_height = max(1, min(TARGET_HEIGHT, int(round(base_image.height * scale))))

    resized = base_image.resize((scaled_width, scaled_height), resample=Image.NEAREST)
    canvas = Image.new(TARGET_MODE, (TARGET_WIDTH, TARGET_HEIGHT), color=(0, 0, 0))
    left = (TARGET_WIDTH - scaled_width) // 2
    top = (TARGET_HEIGHT - scaled_height) // 2
    canvas.paste(resized, (left, top))
    return canvas
