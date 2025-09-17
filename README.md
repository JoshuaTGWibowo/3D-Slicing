# 3D-Slicing

3D model slicing software that uses convex slicing instead of planar slicing.

## Features

- Computes a steady-state convex meniscus curve from printing parameters.
- Centers and aligns STL models before slicing.
- Performs voxel-based convex slicing using a configurable pitch (default 0.05 mm).
- Exports incremental exposure frames as 8-bit BMP images and optional metadata.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the slicer by pointing it at an STL model and an output directory:

```bash
python -m slicer.cli path/to/model.stl output_frames/ --metadata output_frames/summary.json
```

By default the slicer uses the MVP parameters:

- Print head diameter: 5.42 mm
- Rim starting height: 0.75 mm
- Contact angle: 45 °
- Surface tension: 73
- Density (rho): 1000 kg/m³
- Gravity: 9.81 m/s²
- Bézier coefficients: k1 = 0.25, k2 = 0.75
- Pitch: 0.05 mm

These can be overridden using command-line options (see `python -m slicer.cli --help`).

The generated metadata file contains the normalized meniscus profile, applied model
translation, and the apex height used for each frame.
