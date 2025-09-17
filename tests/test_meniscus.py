import numpy as np

from slicer.meniscus import MeniscusParameters, SteadyMeniscus


def test_meniscus_profile_is_monotonic():
    params = MeniscusParameters(
        print_head_diameter=5.42,
        rim_height=0.75,
        contact_angle_deg=45.0,
        surface_tension=73.0,
        density=1000.0,
        gravity=9.81,
        bezier_k1=0.25,
        bezier_k2=0.75,
    )
    meniscus = SteadyMeniscus(params, samples=32)
    points = meniscus.as_points()
    radii = points[:, 0]
    heights = points[:, 1]
    assert np.all(np.diff(radii) >= -1e-6)
    assert heights[0] > heights[-1]
