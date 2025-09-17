"""Convex slicing engine."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

from .meniscus import MeniscusParameters, SteadyMeniscus
from .model import PreparedModel


@dataclass
class SliceResult:
    frame_index: int
    apex_height: float
    image_path: Path


class ConvexSlicer:
    """Perform convex slicing on an STL model."""

    def __init__(
        self,
        model: PreparedModel,
        meniscus_params: MeniscusParameters,
        pitch: float,
    ) -> None:
        self.model = model
        self.pitch = pitch
        self.meniscus = SteadyMeniscus(meniscus_params)
        self._validate_model()
        self._prepare_sampling_grid()
        self._voxelize_model()

    def _validate_model(self) -> None:
        radius = self.meniscus.params.radius
        max_radius = np.max(np.linalg.norm(self.model.mesh.vertices[:, :2], axis=1))
        if max_radius > radius:
            raise ValueError(
                "Model exceeds print head radius. Increase the print head size or scale the model."
            )

    def _prepare_sampling_grid(self) -> None:
        radius = self.meniscus.params.radius
        pitch = self.pitch
        diameter = radius * 2.0
        steps = int(np.ceil(diameter / pitch))
        x_coords = -radius + (np.arange(steps) + 0.5) * pitch
        y_coords = -radius + (np.arange(steps) + 0.5) * pitch

        self.x_coords = x_coords
        self.y_coords = y_coords
        self.radius_grid = np.sqrt(x_coords[:, None] ** 2 + y_coords[None, :] ** 2)
        self.profile_grid = self.meniscus.height_for_radius(self.radius_grid)
        self.profile_grid = np.where(
            np.isfinite(self.profile_grid), self.profile_grid, -np.inf
        )

        height = self.model.height
        z_steps = max(int(np.ceil(height / pitch)), 1)
        self.z_coords = (np.arange(z_steps) + 0.5) * pitch

    def _voxelize_model(self) -> None:
        mesh = self.model.mesh
        nx = len(self.x_coords)
        ny = len(self.y_coords)
        nz = len(self.z_coords)

        triangles = mesh.triangles
        tri_xy = triangles[:, :, :2]
        tri_z = triangles[:, :, 2]

        # Precompute barycentric denominators for efficiency.
        v0 = tri_xy[:, 1] - tri_xy[:, 0]
        v1 = tri_xy[:, 2] - tri_xy[:, 0]
        denom = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]

        columns: List[List[float]] = [[] for _ in range(nx * ny)]
        tol = 1e-9
        first_x = self.x_coords[0]
        first_y = self.y_coords[0]
        pitch = self.pitch

        for tri_idx, tri in enumerate(tri_xy):
            area = abs(denom[tri_idx]) * 0.5
            if area < tol:
                continue

            min_xy = tri.min(axis=0)
            max_xy = tri.max(axis=0)

            ix_start = max(0, int(np.floor((min_xy[0] - first_x) / pitch)))
            ix_end = min(nx - 1, int(np.ceil((max_xy[0] - first_x) / pitch)))
            iy_start = max(0, int(np.floor((min_xy[1] - first_y) / pitch)))
            iy_end = min(ny - 1, int(np.ceil((max_xy[1] - first_y) / pitch)))

            if ix_start > ix_end or iy_start > iy_end:
                continue

            tri_z_vals = tri_z[tri_idx]
            denom_value = denom[tri_idx]

            for ix in range(ix_start, ix_end + 1):
                x0 = self.x_coords[ix]
                if x0 < min_xy[0] - pitch or x0 > max_xy[0] + pitch:
                    continue
                for iy in range(iy_start, iy_end + 1):
                    y0 = self.y_coords[iy]
                    if y0 < min_xy[1] - pitch or y0 > max_xy[1] + pitch:
                        continue

                    dx = x0 - tri_xy[tri_idx, 0, 0]
                    dy = y0 - tri_xy[tri_idx, 0, 1]
                    u = (v1[tri_idx, 1] * dx - v1[tri_idx, 0] * dy) / denom_value
                    v = (-v0[tri_idx, 1] * dx + v0[tri_idx, 0] * dy) / denom_value
                    w = 1.0 - u - v
                    if min(u, v, w) < -1e-6 or max(u, v, w) > 1.0 + 1e-6:
                        continue

                    z = u * tri_z_vals[1] + v * tri_z_vals[2] + w * tri_z_vals[0]
                    columns[ix * ny + iy].append(z)

        occupancy = np.zeros((nx, ny, nz), dtype=bool)
        z_centers = self.z_coords

        for idx, intersections in enumerate(columns):
            if not intersections:
                continue
            zs = np.array(sorted(intersections))
            unique = []
            for value in zs:
                if not unique or abs(value - unique[-1]) > 1e-5:
                    unique.append(value)
            if len(unique) < 2:
                continue

            ix = idx // ny
            iy = idx % ny

            for start in range(0, len(unique) - 1, 2):
                lower = unique[start]
                upper = unique[start + 1]
                if upper < lower:
                    lower, upper = upper, lower
                mask = (z_centers >= lower - self.pitch / 2) & (z_centers <= upper + self.pitch / 2)
                occupancy[ix, iy, mask] = True

        self.occupancy = occupancy

    def _apex_heights(self) -> Sequence[float]:
        height = self.model.height
        steps = int(np.ceil(height / self.pitch))
        return [self.meniscus.params.rim_height + self.pitch * (i + 1) for i in range(steps)]

    def slice(self, output_dir: Path) -> List[SliceResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        apex_heights = list(self._apex_heights())
        results: List[SliceResult] = []
        cumulative_mask = np.zeros(self.profile_grid.shape, dtype=bool)

        for index, apex in enumerate(apex_heights):
            surface = apex + self.profile_grid
            z_mask = self.z_coords[None, None, :] <= surface[:, :, None]
            exposure = np.any(self.occupancy & z_mask, axis=2)
            incremental = exposure & ~cumulative_mask
            cumulative_mask |= exposure

            image_array = np.flipud(incremental.T.astype(np.uint8) * 255)
            image = Image.fromarray(image_array, mode="L")
            filename = output_dir / f"frame_{index:04d}.bmp"
            image.save(filename)

            results.append(SliceResult(frame_index=index, apex_height=apex, image_path=filename))

        return results

    def meniscus_points(self) -> np.ndarray:
        return self.meniscus.as_points()
