"""Meniscus profile calculation for convex slicing."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass
class MeniscusParameters:
    """Physical parameters describing the steady-state meniscus."""

    print_head_diameter: float  # mm
    rim_height: float  # mm
    contact_angle_deg: float  # degrees
    surface_tension: float  # mN/m or equivalent units (used for diagnostics)
    density: float  # kg/m^3
    gravity: float  # m/s^2
    bezier_k1: float
    bezier_k2: float

    @property
    def radius(self) -> float:
        return self.print_head_diameter / 2.0

    @property
    def contact_angle_rad(self) -> float:
        return math.radians(self.contact_angle_deg)

    @property
    def capillary_length(self) -> float:
        """Capillary length derived from the supplied physical properties."""
        # Convert surface tension from mN/m to N/m if provided in that unit.
        gamma = self.surface_tension / 1000.0
        return math.sqrt(gamma / (self.density * self.gravity))


class SteadyMeniscus:
    """Represents the steady-state meniscus profile using a cubic Bézier curve."""

    def __init__(self, params: MeniscusParameters, samples: int = 256) -> None:
        self.params = params
        self.samples = samples
        self._sample_profile()

    def _control_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        radius = p.radius
        rim_height = p.rim_height

        p0 = np.array([0.0, 0.0])
        p1 = np.array([radius * p.bezier_k1, 0.0])

        slope = -math.tan(p.contact_angle_rad)
        r3 = radius
        z3 = -rim_height
        r2 = radius * p.bezier_k2
        z2 = z3 - slope * (r3 - r2)
        p2 = np.array([r2, z2])
        p3 = np.array([r3, z3])

        return p0, p1, p2, p3

    def _sample_profile(self) -> None:
        p0, p1, p2, p3 = self._control_points()
        t_values = np.linspace(0.0, 1.0, self.samples)
        r_samples = self._cubic_bezier(t_values, p0[0], p1[0], p2[0], p3[0])
        z_samples = self._cubic_bezier(t_values, p0[1], p1[1], p2[1], p3[1])

        # Ensure monotonicity for interpolation.
        order = np.argsort(r_samples)
        self._r_samples = r_samples[order]
        self._z_samples = z_samples[order]

    @staticmethod
    def _cubic_bezier(t: np.ndarray, p0: float, p1: float, p2: float, p3: float) -> np.ndarray:
        inv_t = 1.0 - t
        return (
            (inv_t ** 3) * p0
            + 3.0 * (inv_t ** 2) * t * p1
            + 3.0 * inv_t * (t ** 2) * p2
            + (t ** 3) * p3
        )

    def height_for_radius(self, radius: np.ndarray) -> np.ndarray:
        """Return the vertical offset of the meniscus for the supplied radius.

        The returned height is relative to the apex of the meniscus. The apex
        height itself is defined elsewhere by the slicing routine.
        """

        radius = np.asarray(radius)
        z = np.interp(radius, self._r_samples, self._z_samples, left=self._z_samples[0], right=self._z_samples[-1])
        outside = radius > self.params.radius
        if np.any(outside):
            z = np.array(z, copy=True)
            z[outside] = np.nan
        return z

    def apex_adjustment(self) -> float:
        """Return the offset between the meniscus rim and apex heights."""
        # The rim is located at radius = print head radius.
        rim_height = np.interp(self.params.radius, self._r_samples, self._z_samples)
        return -rim_height

    def as_points(self) -> np.ndarray:
        """Return sampled (r, z) pairs describing the profile."""
        return np.column_stack((self._r_samples, self._z_samples))
