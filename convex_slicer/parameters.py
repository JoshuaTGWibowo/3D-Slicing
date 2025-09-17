"""Parameter definitions for convex slicing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrintParameters:
    """Container for the physics and geometry parameters.

    All distances are expressed in millimetres and forces in SI units.
    """

    print_head_diameter: float
    rim_start_height: float
    contact_angle_deg: float
    surface_tension_mN_m: float
    density: float
    gravity: float
    bezier_k1: float
    bezier_k2: float
    pitch: float = 0.05

    @property
    def radius(self) -> float:
        """Return the print head radius in millimetres."""

        return self.print_head_diameter / 2.0

    @property
    def contact_angle_rad(self) -> float:
        """Contact angle in radians."""

        from math import radians

        return radians(self.contact_angle_deg)

    @property
    def surface_tension(self) -> float:
        """Surface tension expressed in N/m."""

        return self.surface_tension_mN_m * 1e-3

    @property
    def capillary_length_mm(self) -> float:
        """Capillary length derived from material properties in millimetres."""

        from math import sqrt

        # capillary length computed in metres, then converted to millimetres
        length_m = sqrt(self.surface_tension / (self.density * self.gravity))
        return length_m * 1000.0

