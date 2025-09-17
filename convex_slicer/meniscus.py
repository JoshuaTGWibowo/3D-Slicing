"""Steady phase meniscus computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .parameters import PrintParameters


@dataclass
class MeniscusProfile:
    """Container for the steady phase meniscus data."""

    radius: float
    control_points: Tuple[float, float, float, float]
    k1: float

    def height(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the meniscus height for the supplied radial distances.

        Parameters
        ----------
        r:
            Radial distances in millimetres.

        Returns
        -------
        np.ndarray
            Height values in millimetres with NaN assigned to radii outside the
            print-head footprint.
        """

        r = np.asarray(r, dtype=float)
        t = np.empty_like(r)
        # Normalise radius and clamp to the valid range
        with np.errstate(divide="ignore", invalid="ignore"):
            normalised = r / self.radius
        np.clip(normalised, 0.0, 1.0, out=normalised)
        # Apply the k1-controlled easing to redistribute sampling density.
        exponent = 1.0 / (1.0 + max(self.k1, 1e-6))
        t[:] = normalised ** exponent
        b0, b1, b2, b3 = self.control_points
        omt = 1.0 - t
        height = (
            (omt ** 3) * b0
            + 3.0 * (omt ** 2) * t * b1
            + 3.0 * omt * (t ** 2) * b2
            + (t ** 3) * b3
        )
        height[r > self.radius] = np.nan
        return height

    @property
    def maximum_height(self) -> float:
        """Return the apex height of the meniscus."""

        return self.control_points[0]


class SteadyPhaseMeniscus:
    """Generate a steady phase meniscus model."""

    def __init__(self, parameters: PrintParameters):
        self.parameters = parameters
        self.profile = self._build_profile()

    def _build_profile(self) -> MeniscusProfile:
        params = self.parameters
        radius = params.radius
        theta = params.contact_angle_rad
        capillary_length = params.capillary_length_mm

        # Primary approximation using a spherical cap constrained by contact angle.
        sphere_radius = radius / np.sin(theta)
        cap_height = sphere_radius * (1.0 - np.cos(theta))
        # Limit the height using the capillary length to avoid unphysical domes.
        cap_height = float(min(cap_height, 2.0 * capillary_length))

        # Control points for the cubic Bézier representation of the profile.
        z0 = cap_height
        z1 = z0  # enforce horizontal tangent at the centre of the meniscus

        # Ideal z2 dictated by the contact angle at the rim
        target_z2 = (radius * np.tan(theta)) / 3.0
        # Blend between the spherical cap and the contact-angle constrained value
        z2 = (1.0 - params.bezier_k2) * z0 + params.bezier_k2 * target_z2
        z3 = 0.0

        control_points = (z0, z1, z2, z3)
        return MeniscusProfile(radius=radius, control_points=control_points, k1=params.bezier_k1)

    def height(self, r: np.ndarray) -> np.ndarray:
        """Convenience wrapper for evaluating the profile."""

        return self.profile.height(r)

    @property
    def maximum_height(self) -> float:
        return self.profile.maximum_height

