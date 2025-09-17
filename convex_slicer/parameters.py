"""Data structures for convex slicing parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrintingParameters:
    """Collection of physics and geometry parameters for the slicer."""

    print_head_diameter: float
    rim_start_height: float
    contact_angle_deg: float
    surface_tension: float
    density: float
    gravity: float
    bezier_k1: float
    bezier_k2: float

    @property
    def print_head_radius(self) -> float:
        """Return half the print-head diameter."""

        return self.print_head_diameter / 2.0
