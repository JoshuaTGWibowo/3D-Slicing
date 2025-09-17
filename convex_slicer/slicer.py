"""Convex slicing implementation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import trimesh
from PIL import Image

from .meniscus import SteadyPhaseMeniscus
from .parameters import PrintParameters


@dataclass
class SliceResult:
    """Data produced for each generated slice."""

    index: int
    height: float
    image_path: Path


class ConvexSlicer:
    """Generate convex slices for a given mesh."""

    def __init__(self, parameters: PrintParameters, pixel_size: float = 0.05):
        self.parameters = parameters
        self.pixel_size = pixel_size
        self.meniscus = SteadyPhaseMeniscus(parameters)

    def slice(self, stl_path: os.PathLike[str] | str, output_dir: os.PathLike[str] | str) -> List[SliceResult]:
        """Slice the provided STL model using convex slicing.

        Parameters
        ----------
        stl_path:
            Path to the STL mesh.
        output_dir:
            Directory where the generated bitmaps and metadata will be stored.
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mesh = self._load_and_centre_mesh(stl_path)
        grid_x, grid_y, radius_grid = self._build_xy_grid()
        meniscus_heights = self.meniscus.height(radius_grid)

        height_map = self._generate_height_map(mesh, grid_x, grid_y)
        layer_heights = self._build_layer_schedule(mesh)

        results: List[SliceResult] = []
        for index, base_height in enumerate(layer_heights):
            surface = base_height + meniscus_heights
            slice_image = self._create_slice_mask(height_map, surface, meniscus_heights)
            image_path = output_path / f"frame_{index:04d}.bmp"
            self._save_bitmap(slice_image, image_path)
            results.append(SliceResult(index=index, height=float(base_height), image_path=image_path))

        self._write_metadata(results, output_path, mesh)
        return results

    # ------------------------------------------------------------------
    # Mesh preparation
    # ------------------------------------------------------------------

    def _load_and_centre_mesh(self, stl_path: os.PathLike[str] | str) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(stl_path)
        if not isinstance(mesh, trimesh.Trimesh):
            # In case the file stores a scene we merge the geometries.
            mesh = trimesh.util.concatenate(mesh.dump())
        mesh = mesh.copy()
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()

        # Centre the model in the XY plane.
        bounds = mesh.bounds
        centre_xy = (bounds[0, :2] + bounds[1, :2]) / 2.0
        mesh.apply_translation(np.array([-centre_xy[0], -centre_xy[1], 0.0]))

        # Raise the model so that the minimum Z matches the rim start height
        bounds = mesh.bounds
        z_shift = self.parameters.rim_start_height - bounds[0, 2]
        mesh.apply_translation([0.0, 0.0, z_shift])

        return mesh

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_xy_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius = self.parameters.radius
        half_extent = radius
        spacing = self.pixel_size

        count = max(1, int(math.ceil((2.0 * half_extent) / spacing)))
        if count % 2 == 0:
            count += 1
        coords = (np.arange(count) - count // 2) * spacing
        grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
        radius_grid = np.sqrt(grid_x ** 2 + grid_y ** 2)
        return grid_x, grid_y, radius_grid

    # ------------------------------------------------------------------
    # Height evaluation
    # ------------------------------------------------------------------

    def _generate_height_map(
        self, mesh: trimesh.Trimesh, grid_x: np.ndarray, grid_y: np.ndarray
    ) -> np.ndarray:
        # Cast vertical rays from above the model to gather the outer surface height.
        z_top = float(mesh.bounds[1, 2] + 5.0 * self.pixel_size)
        origins = np.column_stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.full(grid_x.size, z_top)]
        )
        directions = np.tile(np.array([0.0, 0.0, -1.0]), (origins.shape[0], 1))

        try:
            intersector = mesh.ray
            locations, ray_ids, _ = intersector.intersects_location(origins, directions)
        except BaseException:
            from trimesh.ray.ray_triangle import RayMeshIntersector

            intersector = RayMeshIntersector(mesh, exact=False)
            locations, ray_ids, _ = intersector.intersects_location(origins, directions)

        height_map = np.full(origins.shape[0], -np.inf, dtype=float)
        for ray_index, z in zip(ray_ids, locations[:, 2]):
            if z > height_map[ray_index]:
                height_map[ray_index] = z

        return height_map.reshape(grid_x.shape)

    # ------------------------------------------------------------------
    # Layer and slicing
    # ------------------------------------------------------------------

    def _build_layer_schedule(self, mesh: trimesh.Trimesh) -> Iterable[float]:
        bounds = mesh.bounds
        model_height = bounds[1, 2] - bounds[0, 2]
        pitch = self.parameters.pitch
        num_layers = max(1, int(math.ceil(model_height / pitch)))
        base = self.parameters.rim_start_height
        return (base + i * pitch for i in range(num_layers))

    def _create_slice_mask(
        self,
        height_map: np.ndarray,
        surface: np.ndarray,
        meniscus_heights: np.ndarray,
    ) -> np.ndarray:
        mask = height_map >= surface
        mask = np.where(np.isnan(meniscus_heights), False, mask)
        return mask.astype(np.uint8) * 255

    def _save_bitmap(self, mask: np.ndarray, path: Path) -> None:
        image = Image.fromarray(np.flipud(mask.T), mode="L")
        image.save(path, format="BMP")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _write_metadata(
        self, results: Iterable[SliceResult], output_path: Path, mesh: trimesh.Trimesh
    ) -> None:
        metadata = {
            "parameters": {
                "print_head_diameter_mm": self.parameters.print_head_diameter,
                "rim_start_height_mm": self.parameters.rim_start_height,
                "contact_angle_deg": self.parameters.contact_angle_deg,
                "surface_tension_mN_m": self.parameters.surface_tension_mN_m,
                "density": self.parameters.density,
                "gravity": self.parameters.gravity,
                "bezier_k1": self.parameters.bezier_k1,
                "bezier_k2": self.parameters.bezier_k2,
                "pitch_mm": self.parameters.pitch,
                "pixel_size_mm": self.pixel_size,
                "meniscus_apex_height_mm": self.meniscus.maximum_height,
            },
            "model": {
                "bounds_mm": mesh.bounds.tolist(),
                "height_mm": float(mesh.bounds[1, 2] - mesh.bounds[0, 2]),
            },
            "slices": [
                {
                    "index": result.index,
                    "height_mm": result.height,
                    "image": result.image_path.name,
                }
                for result in results
            ],
        }

        metadata_path = output_path / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

