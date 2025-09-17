"""Utilities for loading and preparing 3D models for slicing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


@dataclass
class PreparedModel:
    mesh: trimesh.Trimesh
    translation: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]

    @property
    def height(self) -> float:
        return float(self.bounds[1][2] - self.bounds[0][2])


class ModelPreparer:
    """Load, center, and align meshes for convex slicing."""

    def __init__(self, ensure_watertight: bool = True) -> None:
        self.ensure_watertight = ensure_watertight

    def load(self, path: Path) -> PreparedModel:
        mesh = trimesh.load_mesh(path, process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Expected an STL mesh containing a single solid model")

        if self.ensure_watertight and not mesh.is_watertight:
            mesh = mesh.fill_holes()

        mesh = mesh.copy()
        bounds = mesh.bounds
        center_x = (bounds[0][0] + bounds[1][0]) / 2.0
        center_y = (bounds[0][1] + bounds[1][1]) / 2.0
        min_z = bounds[0][2]
        translation = np.array([-center_x, -center_y, -min_z])
        mesh.apply_translation(translation)
        centered_bounds = mesh.bounds

        return PreparedModel(mesh=mesh, translation=translation, bounds=centered_bounds)
