Field testing is essential for evaluating the performance of off-road vehicles under realistic conditions.
However, such tests are costly, time-consuming, and pose risks to prototype integrity.
To mitigate these issues, digital simulations have been developed as a complementary tool {Multifractal Terrain Generation for Evaluating Autonomous Off-Road Ground Vehicles}.  

In order to effectively complement field testing, a digital simulation must achieve both physical accuracy and the capacity to generate unique, traversable terrains.
Several systems have already been developed that can produce high-resolution terrains, reaching levels of detail as fine as 5 cm per point, while providing control over surface roughness {Multifractal Terrain Generation for Evaluating Autonomous Off-Road Ground Vehicles}.
Nevertheless, the integration of small-scale features such as anisotropy, roughness, and surface protrusions with large-scale formations such as ridges, hills, and crevices in a unified simulation environment remains relatively unexplored.  

The focus of this work is the development of a digital system for generating non-deformable terrains that incorporate both realistic small-scale and large-scale features. To support this objective, we reviewed the main categories of terrain generation approaches:  

1. **Noise-based algorithms.** Methods such as Perlin Noise {An image synthesizer}, multifractal noise {A multivariate Weierstrass–Mandelbrot function; On the Weierstrass-Mandelbrot fractal function}, and related fractal-based approaches {Fractal terrain generation for vehicle simulation} generate terrains procedurally with low implementation complexity.
These methods are efficient and adjustable, making them suitable for controlling roughness and local detail.
However, their ability to capture the full realism of natural geological processes is limited.  

2. **Geological simulation.** Algorithms inspired by natural processes, such as hydraulic erosion {Fast Hydraulic Erosion Simulation and Visualization on GPU, Terrain simulation using a model of stream erosion}, aim to replicate the physical forces shaping terrain.
They produce realistic landscapes and capture large-scale geomorphology.
The trade-off is lack of uniqueness and high computational cost, which can limit their scalability in large simulations.  

3. **Data-driven synthesis.** Machine learning methods, including Generative Adversarial Networks (GANs) {A step towards procedural terrain generation with GANs; Procedural Terrain Generation Using Generative Adversarial Networks}, Neural Style Transfer {Authoring Terrains with Spatialised Style; StyleDEM: a Versatile Model for Authoring Terrains; Procedural terrain generation with style transfer}, and diffusion-based models {EarthGen: Generating the World from Top-Down Views; MESA: Text-Driven Terrain Generation Using Latent Diffusion and Global Copernicus Data}, leverage real-world data to generate terrains with strong realism.
These approaches are powerful but often rely on large datasets and involve high implementation complexity.
Recent works extend this trend to user-guided interfaces, such as terrain authoring through virtual reality {Generative Terrain Authoring with Mid-air Hand Sketching in Virtual Reality}, and text-to-terrain synthesis using global satellite data {MESA: Text-Driven Terrain Generation Using Latent Diffusion and Global Copernicus Data}.  

Research in computer graphics has historically played a central role in advancing terrain generation.
Early work demonstrated the use of procedural noise for textures and surfaces {An image synthesizer}, later expanding into city-scale and planetary-scale procedural modeling {Procedural modeling of cities; Procedural Planetary Multi-resolution Terrain Generation for Games}.
These contributions emphasized scalability and artistic control, allowing environments to be generated at multiple levels of resolution.
While these methods often prioritize visual plausibility over physical accuracy, they provide insight into how holistic terrains can be generated with coherent integration of small- and large-scale features.  

To evaluate the suitability of these approaches, we define a set of metrics:  

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

Most research in robotics and off-road vehicle testing has concentrated on the generation of small-scale terrain features.
For example, Perlin Noise has been successfully applied to simulate surface irregularities sufficient to generate traversability datasets {Learning Ground Traversability From Simulations}.
More recent studies propose multifractal terrain functions that allow fine control over noise parameters to produce varied small-scale terrain surfaces {Multifractal Terrain Generation for Evaluating Autonomous Off-Road Ground Vehicles}.
These contributions highlight the effectiveness of noise-based approaches for local surface modeling.
However, they fall short of providing a holistic simulation environment that incorporates both the fine-grained detail necessary for ground interaction and the larger-scale landforms required for full environment evaluation.  

This work aims to address this gap by developing a comprehensive system capable of generating terrains that integrate both levels of detail in a unified, realistic, and controllable manner.
