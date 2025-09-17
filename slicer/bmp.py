"""Simple BMP writer for grayscale masks."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence


def save_bitmap(path: str | Path, mask: Sequence[Sequence[int]]) -> None:
    """Save a 2D mask as a 24-bit BMP image."""

    rows = list(mask)
    if not rows:
        raise ValueError("Mask must contain at least one row")
    width = len(rows[0])
    if width == 0:
        raise ValueError("Mask rows must not be empty")
    for row in rows:
        if len(row) != width:
            raise ValueError("Mask rows must have equal length")

    height = len(rows)
    path = Path(path)

    row_padding = (4 - (width * 3) % 4) % 4
    pixel_data_size = (width * 3 + row_padding) * height

    file_header = struct.pack(
        "<2sIHHI",
        b"BM",
        14 + 40 + pixel_data_size,
        0,
        0,
        14 + 40,
    )

    dib_header = struct.pack(
        "<IIIHHIIIIII",
        40,
        width,
        height,
        1,
        24,
        0,
        pixel_data_size,
        2835,
        2835,
        0,
        0,
    )

    padding_bytes = b"\x00" * row_padding

    with path.open("wb") as fh:
        fh.write(file_header)
        fh.write(dib_header)
        for row in reversed(rows):
            line = bytearray()
            for value in row:
                v = int(value) & 0xFF
                line.extend((v, v, v))
            fh.write(bytes(line))
            if padding_bytes:
                fh.write(padding_bytes)
