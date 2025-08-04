### Procedural city/road generation  
<https://cgl.ethz.ch/Downloads/Publications/Papers/2001/p_Par01.pdf>  
#### Can be useful for large-scale terrain realism
- The paper demonstrates how a water map, an elevation map, and a population density map can be used as inputs to apply an L-system for generating a set of roads.  
  - ![Roadmap generated from water + elevation + population density maps](images/p1i1.png)  
- By applying L-systems to an elevation map, fractals can be generated and integrated into the existing terrain, followed by blurring techniques to create more realistic large-scale terrain features.  
  - ![Highways generated from a population density map](images/p1i2.png)
---
### Learning ground traversability from simulations  
<https://idsia-robotics.github.io/files/publications/chavez-garcia2018.pdf>  
#### Can be useful for traversability estimation
- The paper demonstrates how a terrain patch can be classified as "traversable" or "not traversable" using a CNN.
  - Inputs: height map
  - Labels: a robot is simulated traversing the height map using Gazebo and ODE -> terrain is classified as "traversable" or "not traversable"
  - Output: binary classification as "traversable" or "not traversable"
  - ![CNN traversability training](images/p2i1.png)
- Both procedurally generated and real-life patches of terrain were used.
  - ![CNN output on real-life terrain](images/p2i2.png)
---

### Multifractal Terrain Generation for Evaluating Autonomous Off-Road Ground Vehicles  
<https://arxiv.org/abs/2501.02172#>  
#### _Add description here_

---

### Fast Hydraulic Erosion Simulation and Visualization on GPU  
<https://inria.hal.science/inria-00402079/document>  
#### _Add description here_

---

### Polynomial methods for fast Procedural Terrain Generation  
<https://arxiv.org/pdf/1610.03525>  
#### _Add description here_

---

### Terrain generation using genetic algorithms  
<https://www.cs.york.ac.uk/rts/docs/GECCO_2005/Conference%20proceedings/docs/p1463.pdf>  
#### _Add description here_

---

### PTRM: Perceived Terrain Realism Metric  
<https://dl.acm.org/doi/10.1145/3514244>  
#### _Add description here_

---

### Controlled Procedural Terrain Generation Using Software Agents  
<https://ianparberry.com/pubs/terrain.pdf>  
#### _Add description here_

---

### A multivariate Weierstrass–Mandelbrot function  
<https://www.researchgate.net/publication/239037074_A_Multivariate_Weierstrass-Mandelbrot_Function>  
#### _Add description here_

---

### Fractal terrain generation for vehicle simulation  
<https://www.academia.edu/84221963/Fractal_terrain_generation_for_vehicle_simulation>  
#### _Add description here_
