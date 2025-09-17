# 3D-Slicing

Convex slicing MVP that loads an STL model, computes a steady phase meniscus and
produces a stack of bitmap frames aligned to the curved interface.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py path/to/model.stl --output output_directory
```

Optional arguments allow overriding the physical parameters, pitch and
in-plane sampling resolution. Generated frames are saved as monochrome BMP
images named `frame_XXXX.bmp` alongside a `metadata.json` file that captures the
slicing configuration.
