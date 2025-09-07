Field tests are crucial to the evaluation of the performance of an off-road vehicle.
It is important to note, however, that field tests pose certain challenges, such as high cost, high risk posed to the integrity of the vehicle's prototype, and the amount of time taken to conduct a test.
To overcome these challenges, digital simulations have been developed.

In order to efficiently complement a field test of an off-road vehicle, a digital simulation must possess certain qualities, such as physical accuracy and the capacity to include unique and traversable terrains.
Numerous digital simulations have been developed to generate high-resolution (as detailed as 5 cm per point) terrains with a controlled level of roughness.
However, the development of holistic simulation environments combining both small-scale terrain features, such as anisotropy, roughness, and the presence of protrusions, and large-scale terrain features, such as the presence of ridges, hills, and crevices, has not been explored in-depth.

The focus of this paper is on the development of a digital system designed to generate non-deformable terrains possessing realistic small-scale and large-scale features.
In order to accomplish this goal, a variety of methods was explored.
Multiple sources identify three types of algorithms used to generate terrains: noise-based algorithms such as Perlin Noise or Simplex Noise, geological simulation, such as erosion simulation, and synthesis from real-life data, such as Generative Adversarial Networks (GANs) or Neural Style Transfer (NST).

To evaluate the fitness of the methods used to generate terrains, we have compiled a set of metrics:

| Metric                   | Meaning                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| Implementation complexity| Effort required to code, integrate, and maintain the algorithm          |
| Algorithmic complexity   | Computational cost in time and memory as terrain size increases         |
| Uniqueness               | Ability to produce non-repetitive, distinctive terrain outputs          |
| Scale                    | Spatial resolution of terrain (number of meters per point)              |
| Realism                  | How closely generated terrain mimics natural geological formations      |
| Adjustability            | Extent of user control over parameters such as roughness, slope, height |
| Data dependency          | Reliance on external datasets                                           |
| Parallelizability        | Suitability for GPU or multi-core processing to improve performance     |

We have explored previous research in terrain generation to select most suitable methods for the holistic terrain generation system.

| Publication | Authors | Date | Algorithm type | Implementation complexity | Algorithmic complexity | Uniqueness | Scale | Realism | Adjustability | Data dependency | Parallelizability |
|-------------|---------|------|----------------|---------------------------|------------------------|------------|-------|---------|---------------|-----------------|-------------------|
| A step towards procedural terrain generation with GANs | C. Beckham, C. Pal | 2017 | | | | | | | | | |
| An image synthesizer | K. Perlin | 1985 | | | | | | | | | |
| Authoring Terrains with Spatialised Style | S. Perche, A. Peytavie, B. Benes, E. Galin, E. Guérin | 2023 | | | | | | | | | |
| EarthGen: Generating the World from Top-Down Views | A. Sharma et al. | 2024 | | | | | | | | | |
| Fast Hydraulic Erosion Simulation and Visualization on GPU | X. Mei, P. Decaudin, B.-G. Hu | 2007 | | | | | | | | | |
| Fractal terrain generation for vehicle simulation | J. J. Dawkins, D. M. Bevly, R. L. Jackson | 2012 | | | | | | | | | |
| Generative Terrain Authoring with Mid-air Hand Sketching in Virtual Reality | Y. Hu, K. Wang, Y. Shao, J. Plass, Z. Wang, K. Perlin | 2024 | | | | | | | | | |
| MESA: Text-Driven Terrain Generation Using Latent Diffusion and Global Copernicus Data | P. Borne--Pons, M. Czerkawski, R. Martin, R. Rouffet | 2025 | | | | | | | | | |
| Multifractal Terrain Generation for Evaluating Autonomous Off-Road Ground Vehicles | C. D. Majhor, J. P. Bos | 2025 | | | | | | | | | |
| Procedural Terrain Generation Using Generative Adversarial Networks | G. Voulgaris, I. Mademlis, I. Pitas | 2021 | | | | | | | | | |
| Procedural terrain generation with style transfer | F. Merizzi | 2024 | | | | | | | | | |
| StyleDEM: a Versatile Model for Authoring Terrains | S. Perche, A. Peytavie, B. Benes, E. Galin, E. Guérin | 2023 | | | | | | | | | |
| Terrain generation using genetic algorithms | T. J. Ong, R. Saunders, J. Keyser, J. J. Leggett | 2005 | | | | | | | | | |
| Visually Improved Erosion Algorithm for the Procedural Generation of Tile-based Terrain | F. Lim, Y. Tan, A. Bhojan | 2022 | | | | | | | | | |

