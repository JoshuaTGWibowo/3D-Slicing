"""Meniscus profile computations for convex slicing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MeniscusParameters:
    print_head_diameter: float
    rim_start_height: float
    contact_angle_deg: float
    surface_tension: float
    density: float
    gravity: float
    bezier_k1: float
    bezier_k2: float


@dataclass
class MeniscusProfile:
    radius: float
    rim_height: float
    contact_angle: float
    sphere_radius: float
    center_z: float
    central_height: float
    control_points: Tuple[Tuple[float, float], ...]

    def height(self, r: float) -> float:
        """Return the meniscus height at radius ``r`` in millimetres."""

        r = abs(r)
        if r <= 0:
            return self.central_height
        if r >= self.radius:
            return self.rim_height
        return _bezier_height(self.control_points, r)

    def delta(self, r: float) -> float:
        """Return the offset needed to flatten the meniscus at radius ``r``."""

        return self.central_height - self.height(r)

    def as_dict(self) -> dict:
        return {
            "radius": self.radius,
            "rim_height": self.rim_height,
            "contact_angle_deg": math.degrees(self.contact_angle),
            "sphere_radius": self.sphere_radius,
            "center_z": self.center_z,
            "central_height": self.central_height,
            "control_points": [list(pt) for pt in self.control_points],
        }


def compute_meniscus(params: MeniscusParameters) -> MeniscusProfile:
    """Compute the steady-state meniscus profile using a spherical cap model."""

    radius = params.print_head_diameter / 2.0
    contact_angle = math.radians(params.contact_angle_deg)
    if contact_angle <= 0 or contact_angle >= math.pi / 2:
        raise ValueError("Contact angle must be between 0 and 90 degrees")

    sphere_radius = radius / math.sin(contact_angle)
    center_z = -radius / math.tan(contact_angle)
    central_height_offset = center_z + sphere_radius
    rim_height = params.rim_start_height
    central_height = rim_height + central_height_offset

    k1 = max(0.0, min(1.0, params.bezier_k1))
    k2 = max(0.0, min(1.0, params.bezier_k2))

    sample_radii = [0.0, radius * k1, radius * k2, radius]
    sample_heights = [
        rim_height + _spherical_height(r, sphere_radius, center_z) for r in sample_radii
    ]

    control_points = (
        (sample_radii[0], sample_heights[0]),
        (sample_radii[1], sample_heights[0]),
        (sample_radii[2], sample_heights[-1]),
        (sample_radii[3], sample_heights[-1]),
    )

    return MeniscusProfile(
        radius=radius,
        rim_height=rim_height,
        contact_angle=contact_angle,
        sphere_radius=sphere_radius,
        center_z=center_z,
        central_height=central_height,
        control_points=control_points,
    )


def _spherical_height(r: float, sphere_radius: float, center_z: float) -> float:
    value = sphere_radius**2 - r**2
    if value < 0:
        value = 0.0
    return math.sqrt(value) + center_z


def _bezier_height(control_points: Tuple[Tuple[float, float], ...], radius: float) -> float:
    xs = [pt[0] for pt in control_points]
    zs = [pt[1] for pt in control_points]
    target = radius
    low, high = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        x_mid = _bezier(xs, mid)
        if x_mid < target:
            low = mid
        else:
            high = mid
    t = 0.5 * (low + high)
    return _bezier(zs, t)


def _bezier(points: list[float], t: float) -> float:
    mt = 1.0 - t
    return (
        points[0] * (mt ** 3)
        + 3 * points[1] * (mt ** 2) * t
        + 3 * points[2] * mt * (t ** 2)
        + points[3] * (t ** 3)
    )
