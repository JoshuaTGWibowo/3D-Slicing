"""Core slicing algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .meniscus import MeniscusProfile


@dataclass
class SliceGrid:
    x_coords: List[float]
    y_coords: List[float]
    intervals: List[List[List[Tuple[float, float]]]]


@dataclass
class SliceParameters:
    pitch: float
    meniscus: MeniscusProfile


class SliceVolume:
    def __init__(self, grid: SliceGrid, params: SliceParameters):
        self.grid = grid
        self.params = params
        self._deltas = [
            [params.meniscus.delta(math.hypot(x, y)) for x in grid.x_coords]
            for y in grid.y_coords
        ]

    def num_frames(self, model_height: float) -> int:
        return int(math.ceil(model_height / self.params.pitch))

    def frame_mask(self, layer_index: int, model_height: float) -> List[List[int]]:
        pitch = self.params.pitch
        meniscus = self.params.meniscus
        base_height = meniscus.rim_height + layer_index * pitch
        width = len(self.grid.x_coords)
        height = len(self.grid.y_coords)
        mask = [[0 for _ in range(width)] for _ in range(height)]

        for iy in range(height):
            for ix in range(width):
                column_intervals = self.grid.intervals[ix][iy]
                if not column_intervals:
                    continue
                delta = self._deltas[iy][ix]
                actual_low = base_height - delta
                actual_high = actual_low + pitch
                if _column_intersects(column_intervals, actual_low, actual_high):
                    mask[iy][ix] = 255
        return mask


def _column_intersects(intervals: Sequence[Tuple[float, float]], low: float, high: float) -> bool:
    if high <= 0:
        return False
    for start, end in intervals:
        if end <= low:
            continue
        if start >= high:
            break
        return True
    return False


def build_grid(triangles: List[List[List[float]]], voxel: float, radius: float) -> SliceGrid:
    min_x = -radius
    max_x = radius
    min_y = -radius
    max_y = radius

    width = int(math.ceil((max_x - min_x) / voxel))
    height = int(math.ceil((max_y - min_y) / voxel))

    x_coords = [min_x + (i + 0.5) * voxel for i in range(width)]
    y_coords = [min_y + (j + 0.5) * voxel for j in range(height)]

    intervals: List[List[List[Tuple[float, float]]]] = [
        [list() for _ in range(height)] for _ in range(width)
    ]

    _populate_intervals(triangles, x_coords, y_coords, voxel, intervals)

    return SliceGrid(x_coords=x_coords, y_coords=y_coords, intervals=intervals)


def _populate_intervals(
    triangles: List[List[List[float]]],
    x_coords: List[float],
    y_coords: List[float],
    voxel: float,
    intervals: List[List[List[Tuple[float, float]]]],
) -> None:
    origin_z = -1.0
    width = len(x_coords)
    height = len(y_coords)
    origin_x = x_coords[0] - 0.5 * voxel
    origin_y = y_coords[0] - 0.5 * voxel

    for tri in triangles:
        v0, v1, v2 = tri
        tri_min_x = min(v0[0], v1[0], v2[0])
        tri_max_x = max(v0[0], v1[0], v2[0])
        tri_min_y = min(v0[1], v1[1], v2[1])
        tri_max_y = max(v0[1], v1[1], v2[1])

        ix_min = max(0, int(math.floor((tri_min_x - origin_x) / voxel)))
        ix_max = min(width - 1, int(math.floor((tri_max_x - origin_x) / voxel)))
        iy_min = max(0, int(math.floor((tri_min_y - origin_y) / voxel)))
        iy_max = min(height - 1, int(math.floor((tri_max_y - origin_y) / voxel)))

        for ix in range(ix_min, ix_max + 1):
            x = x_coords[ix]
            for iy in range(iy_min, iy_max + 1):
                y = y_coords[iy]
                z_hit = _ray_intersection((x, y, origin_z), tri)
                if z_hit is None:
                    continue
                intervals[ix][iy].append(z_hit)

    for ix in range(width):
        for iy in range(height):
            hits = sorted(intervals[ix][iy])
            cleaned: List[Tuple[float, float]] = []
            for i in range(0, len(hits) - 1, 2):
                start = hits[i]
                end = hits[i + 1]
                if end - start > 1e-6:
                    cleaned.append((start, end))
            intervals[ix][iy] = cleaned


def _ray_intersection(origin: Tuple[float, float, float], tri: List[List[float]]) -> float | None:
    EPS = 1e-9
    dir_vec = (0.0, 0.0, 1.0)
    v0, v1, v2 = tri
    edge1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    edge2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    h = _cross(dir_vec, edge2)
    a = _dot(edge1, h)
    if abs(a) < EPS:
        return None
    f = 1.0 / a
    s = (origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2])
    u = f * _dot(s, h)
    if u < -EPS or u > 1.0 + EPS:
        return None
    q = _cross(s, edge1)
    v = f * _dot(dir_vec, q)
    if v < -EPS or u + v > 1.0 + EPS:
        return None
    t = f * _dot(edge2, q)
    if t <= EPS:
        return None
    return origin[2] + t


def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
