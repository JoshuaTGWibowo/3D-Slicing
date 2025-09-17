"""Computation of the steady-phase meniscus profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .parameters import PrintingParameters


def _bezier_point(t: np.ndarray, control_points: np.ndarray) -> np.ndarray:
    """Evaluate a cubic Bézier curve at ``t``.

    Parameters
    ----------
    t:
        Array of parameters in ``[0, 1]``.
    control_points:
        Array of shape ``(4, 2)`` representing ``(radius, height)`` pairs.
    """

    t = np.atleast_1d(t)
    cp = control_points
    # Bernstein basis
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t**2
    b3 = t**3
    points = (
        b0[..., None] * cp[0]
        + b1[..., None] * cp[1]
        + b2[..., None] * cp[2]
        + b3[..., None] * cp[3]
    )
    return points


@dataclass
class SteadyPhaseProfile:
    """Axisymmetric steady-phase profile for the convex slicing surface."""

    control_points: np.ndarray
    radius: float
    rim_height: float

    def height(self, r: Iterable[float]) -> np.ndarray:
        """Return the meniscus height offset relative to the center.

        Values outside the print-head radius are clamped to ``0``.
        """

        r = np.asarray(r, dtype=float)
        r_clamped = np.clip(r, 0.0, self.radius)
        t = np.zeros_like(r_clamped)
        nonzero = self.radius > 0
        if nonzero:
            t = r_clamped / self.radius
        points = _bezier_point(t, self.control_points)
        heights = points[..., 1]
        heights = np.where(r <= self.radius, heights, 0.0)
        return heights

    @property
    def max_height(self) -> float:
        """Return the maximum height offset."""

        return float(self.control_points[-1, 1])


def compute_steady_phase_profile(
    params: PrintingParameters,
    *,
    peak_scale: float = 0.6,
    capillary_fraction: float = 0.1,
) -> SteadyPhaseProfile:
    """Approximate the steady-phase meniscus using a cubic Bézier curve.

    The approximation is inspired by the steady-state model in Dynamic
    Interface Printing. The curve is axisymmetric and parameterised by the
    radial distance from the print-head centre.

    Parameters
    ----------
    params:
        Printing and material parameters.
    peak_scale:
        Relative scaling used for the peak height estimation when capillary
        effects are small.
    capillary_fraction:
        Fraction of the capillary length used when estimating the peak height.
    """

    radius = params.print_head_radius
    angle = np.deg2rad(params.contact_angle_deg)
    capillary_length_m = np.sqrt(params.surface_tension / (params.density * params.gravity))
    capillary_length_mm = capillary_length_m * 1000.0

    peak_guess = min(radius * peak_scale, capillary_length_mm * capillary_fraction)
    peak_guess = max(peak_guess, 1e-6)

    rim_offset = max(peak_guess - np.tan(angle) * radius * (1.0 - params.bezier_k2), 0.0)
    peak_height = rim_offset + np.tan(angle) * radius * (1.0 - params.bezier_k2)

    control_points = np.array(
        [
            [0.0, 0.0],
            [radius * params.bezier_k1, 0.0],
            [radius * params.bezier_k2, rim_offset],
            [radius, peak_height],
        ],
        dtype=float,
    )

    return SteadyPhaseProfile(control_points=control_points, radius=radius, rim_height=params.rim_start_height)
