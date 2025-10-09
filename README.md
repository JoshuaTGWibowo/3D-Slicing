# 3D-Slicing

3D model slicing software that uses convex slicing instead of planar slicing.

## Features

- Loads STL meshes, recenters them in the print volume and aligns the base to the
  configured rim height.
- Computes a steady-phase convex meniscus using the supplied physics parameters
  (surface tension, density, gravity, print-head geometry, Bézier control
  coefficients).
- Voxelises the mesh and generates a sequence of BMP frames that follow the
  curved meniscus for each pitch increment.
  
- Writes metadata describing the slicing run (pitch, voxel size, meniscus
  control points and scaling).

## Usage

Install the dependencies and run the slicer from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m convex_slicer.cli path/to/model.stl output_directory --pitch 0.05
```

Optional arguments:

- `--voxel-size`: specify a custom voxel size (defaults to the pitch).
- `--params`: path to a JSON file overriding the default material and geometry
  parameters.

Each frame is written as a 1-bit monochrome 4K BMP named `frame_XXXX.bmp`. The
`slicing/metadata.json` file captures the parameters used for the run.

## Default parameters

The default values (matching the MVP requirements) are:

- Print head diameter: 5.42 mm
- Rim starting height: 0.75 mm
- Contact angle: 45°
- Surface tension: 73
- Density: 1000
- Gravity: 9.81
- Bézier k1: 0.25
- Bézier k2: 0.75

The slicer uses these to build a cubic Bézier approximation of the steady
meniscus profile (center low, rim high) and applies the same profile for every
layer as it marches through the height of the model using the configured pitch.
