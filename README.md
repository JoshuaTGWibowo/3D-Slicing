# 3D-Slicing

Convex slicing MVP for dynamic interface printing. The tool loads an STL file,
computes the steady-state meniscus using the supplied physics parameters, and
produces a stack of BMP frames following a convex slicing strategy.

## Requirements

* Python 3.10+

## Usage

```
python -m slicer.cli <input.stl> <output-directory> [--pitch 0.05] [--metadata meta.json]
```

* The slicer centres the model in the XY plane and lifts it so the lowest point
  touches `z = 0`.
* The steady phase meniscus is computed once using the provided parameters
  (print head diameter 5.42 mm, rim starting height 0.75 mm, contact angle
  45°, surface tension 73, density 1000, gravity 9.81, Bézier k1 = 0.25,
  Bézier k2 = 0.75).
* Layers are generated from the rim starting height and increase by the pitch
  until the top of the model is reached.
* Output frames are saved as 24-bit BMP files named `frame_0000.bmp`,
  `frame_0001.bmp`, etc.

Optional metadata describing the slicing session can be written via the
`--metadata` argument.
