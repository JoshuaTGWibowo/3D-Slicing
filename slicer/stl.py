"""Utilities for loading STL meshes."""

from __future__ import annotations

import os
import struct
from typing import List


Triangle = List[List[float]]


def load_stl(path: str) -> List[Triangle]:
    """Load an STL file into a list of triangles."""

    with open(path, "rb") as fh:
        header = fh.read(80)
        if len(header) < 80:
            raise ValueError("STL header truncated")
        count_bytes = fh.read(4)
        if len(count_bytes) < 4:
            # Not enough data for binary count, assume ASCII.
            return _load_ascii_stl(path)
        facet_count = struct.unpack("<I", count_bytes)[0]
        expected_size = 80 + 4 + facet_count * 50
        actual_size = os.path.getsize(path)
        if actual_size == expected_size:
            return _load_binary_stl(facet_count, fh)

    # Fallback to ASCII parser.
    return _load_ascii_stl(path)


def _load_binary_stl(facet_count: int, fh) -> List[Triangle]:
    triangles: List[Triangle] = []
    for idx in range(facet_count):
        data = fh.read(50)
        if len(data) < 50:
            raise ValueError("Unexpected end of STL file")
        unpacked = struct.unpack("<12fH", data)
        v0 = unpacked[3:6]
        v1 = unpacked[6:9]
        v2 = unpacked[9:12]
        triangles.append([list(v0), list(v1), list(v2)])
    return triangles


def _load_ascii_stl(path: str) -> List[Triangle]:
    vertices: List[List[float]] = []
    current: List[List[float]] = []
    with open(path, "r", encoding="utf8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.lower().startswith("vertex"):
                parts = stripped.split()
                if len(parts) != 4:
                    raise ValueError(f"Malformed vertex line: {line!r}")
                current.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(current) == 3:
                    vertices.append(current)
                    current = []
            elif stripped.lower().startswith("endfacet"):
                current = []
    if not vertices:
        raise ValueError("No triangles found in STL file")
    return [triangle for triangle in vertices]


def compute_bounds(triangles: List[Triangle]) -> List[List[float]]:
    """Return axis-aligned bounding box for the mesh."""

    xs = [v[0] for tri in triangles for v in tri]
    ys = [v[1] for tri in triangles for v in tri]
    zs = [v[2] for tri in triangles for v in tri]
    return [
        [min(xs), min(ys), min(zs)],
        [max(xs), max(ys), max(zs)],
    ]


def center_mesh(triangles: List[Triangle]) -> List[Triangle]:
    """Center the mesh in XY and shift the base to Z=0."""

    verts = [v for tri in triangles for v in tri]
    count = len(verts)
    if count == 0:
        raise ValueError("Mesh contains no vertices")
    centroid_x = sum(v[0] for v in verts) / count
    centroid_y = sum(v[1] for v in verts) / count
    min_z = min(v[2] for v in verts)

    for tri in triangles:
        for v in tri:
            v[0] -= centroid_x
            v[1] -= centroid_y
            v[2] -= min_z
    return triangles
