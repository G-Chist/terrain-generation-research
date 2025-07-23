## Week 2 results

- perlin_numpy.py
  - removed min/max height
  - added dig_path
    - a function that searches for a path using DFS and a height difference threshold and applies convolutions to draw paths
    - draws a natural-looking path if post-processing (blurring) is applied
  - added terracing
    - terracing + blurring generates a traversable terrain given the right parameters
---
- open_terrain_from_png.py
  - enables us to load a terrain from a .png height map into Blender and smoothen it with some convolutions