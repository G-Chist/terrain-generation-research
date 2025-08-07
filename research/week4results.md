## Week 4 results

- papers described, screenshots added
- apparently WPI does not have ArcGIS
---
- weierstrass-mandelbrot_numpy.py
  - converted to a standalone script
  - uses the Hadamard product to create a realistic rough terrain
---
- weierstrass-mandelbrot_torch.py
  - written using PyTorch
  - returns a .npy array file
  - much faster than the original script
    - creating the terrain and loading it to Blender takes ~3 + ~10 seconds as opposed to ~40 seconds to create it using NumPy and turn it into a Blender mesh
---
- kriging_example.py
  - uses Ordinary Kriging (spherical) to refine a real-life terrain patch