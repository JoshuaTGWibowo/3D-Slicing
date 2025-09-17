from __future__ import annotations

import numpy as np
import pytest

from convex_slicer.parameters import PrintingParameters
from convex_slicer.steady_state import compute_steady_phase_profile


def test_profile_monotonic_increasing():
    params = PrintingParameters(
        print_head_diameter=5.42,
        rim_start_height=0.75,
        contact_angle_deg=45.0,
        surface_tension=73.0,
        density=1000.0,
        gravity=9.81,
        bezier_k1=0.25,
        bezier_k2=0.75,
    )
    profile = compute_steady_phase_profile(params)
    radii = np.linspace(0.0, params.print_head_radius, 100)
    heights = profile.height(radii)
    assert np.all(np.diff(heights) >= -1e-6)
    assert heights[0] == pytest.approx(0.0)
    assert heights[-1] > 0.0


def test_profile_clamps_outside_radius():
    params = PrintingParameters(
        print_head_diameter=5.42,
        rim_start_height=0.75,
        contact_angle_deg=45.0,
        surface_tension=73.0,
        density=1000.0,
        gravity=9.81,
        bezier_k1=0.25,
        bezier_k2=0.75,
    )
    profile = compute_steady_phase_profile(params)
    heights = profile.height(np.array([params.print_head_radius * 1.5]))
    assert heights[0] == pytest.approx(0.0)
