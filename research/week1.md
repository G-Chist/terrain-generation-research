## Week 1 results 

- perlin.py
  - works in Blender
  - code commented out for readability
  - runtime seems reasonable
    - may need to look into rewriting using NumPy for improvements
  - vertex data saved to data/vertices.csv
    - we can go through the list of vertices and alter the Z coordinates (height) to prevent abrupt jumps given a set of adjacent vertices
---
- worley.py
  - did not work in Blender (runtime error)
---
- perlin_numpy.py
  - works in Blender
  - works like perlin.py, but faster
    - example: the code with parameters ``vertices, faces = generate_perlin_noise(iterations=6, n_row=9, n_col=9, grad_range=1,
                                        rand_range=0.1, size=12.0, sea_level=0.0)`` took 9 seconds to run, whereas perlin.py took 19 seconds with the same parameters
