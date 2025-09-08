Field testing is essential for evaluating the performance of off-road vehicles under realistic conditions.  
However, such tests are costly, time-consuming, and pose risks to prototype integrity.  
To mitigate these issues, digital simulations have been developed as a complementary tool [1].  

In order to effectively complement field testing, a digital simulation must achieve both physical accuracy and the capacity to generate unique, traversable terrains.  
Several systems have already been developed that can produce high-resolution terrains, reaching levels of detail as fine as 5 cm per point, while providing control over surface roughness [1].  
Nevertheless, the integration of small-scale features such as anisotropy, roughness, and surface protrusions with large-scale formations such as ridges, hills, and crevices in a unified simulation environment remains relatively unexplored.  

The focus of this work is the development of a digital system for generating non-deformable terrains that incorporate both realistic small-scale and large-scale features.  
To support this objective, we reviewed the main categories of terrain generation approaches:  

1. **Noise-based algorithms.**  
Methods such as Perlin Noise [2], multifractal noise [3], [4], and related fractal-based approaches [5] generate terrains procedurally with low implementation complexity.  
These methods are efficient and adjustable, making them suitable for controlling roughness and local detail.  
However, their ability to capture the full realism of natural geological processes is limited.  

2. **Geological simulation.**  
Algorithms inspired by natural processes, such as hydraulic erosion [6] and stream erosion [7], aim to replicate the physical forces shaping terrain.  
They produce somewhat realistic-looking landscapes.  
The trade-off is lack of uniqueness and high computational cost, which can limit their scalability in large simulations.  

3. **Data-driven synthesis.**  
Machine learning methods, including Generative Adversarial Networks (GANs) [8], [9], Neural Style Transfer [10], [11], [12], and diffusion-based models [13], [14], leverage real-world data to generate terrains with strong realism.  
These approaches are powerful but often rely on large datasets and involve high implementation complexity.  
Recent works extend this trend to user-guided interfaces, such as terrain authoring through virtual reality [15], and text-to-terrain synthesis using global satellite data [14].  

Research in computer graphics has played a central role in advancing terrain generation.  
Early work demonstrated the use of procedural noise for textures and surfaces [2], later expanding into city-scale and planetary-scale procedural modeling [16], [17].  
These contributions emphasized scalability and artistic control, allowing environments to be generated at multiple levels of resolution.  
While these methods often prioritize visual plausibility over physical accuracy, they provide insight into how holistic terrains can be generated with coherent integration of small-scale and large-scale features.  

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
For example, Perlin Noise has been successfully applied to simulate surface irregularities sufficient to generate traversability datasets [18].  
More recent studies propose multifractal terrain functions that allow fine control over noise parameters to produce varied small-scale terrain surfaces [1].  
These contributions highlight the effectiveness of noise-based approaches for local surface modeling.  
However, they fail to provide a holistic simulation environment that incorporates both the fine-grained detail necessary for ground interaction and the larger-scale landforms required for full environment evaluation.  

This work seeks to address the identified gap by developing a comprehensive terrain generation framework that unifies both large-scale and small-scale features within a single, realistic, and controllable simulation environment.  
The system is designed not only to meet the practical demands of off-road vehicle evaluation but also to establish a foundation for extensible research in terrain modeling and simulation.

The objectives of this work are articulated as follows:  

1. **Ensure theoretical traversability of the generated environment.**  
   Generated terrains must enable feasible navigation paths for off-road vehicles, avoiding designs that are inherently non-traversable.  

2. **Achieve realism at large scales.**  
   Terrain formations such as ridges, valleys, and hills must reflect plausible geological structures that resemble natural landscapes.  

3. **Preserve roughness and irregularity at small scales.**  
   Surface anisotropy, micro-variations, and protrusions must be present to ensure realistic wheel-terrain interaction and accurate vehicle response.  

4. **Maintain high spatial resolution.**  
   The system must produce terrains with a minimum resolution of 0.2 meters per point, supporting fine-grained vehicle-ground interaction modeling.  

5. **Provide an intuitive user interface.**  
   Researchers and practitioners must be able to configure parameters and generate terrains without requiring extensive technical expertise in algorithmic implementation.  

6. **Incorporate obstacle placement.**  
   The framework must allow integration of obstacles into the terrain to simulate realistic driving scenarios.  

7. **Enable variety in obstacle configuration.**  
   The system must support controlled diversity in obstacle types, sizes, and placements to ensure a wide range of testing conditions.

---

**References**  

[1] C. D. Majhor and J. P. Bos, “Multifractal Terrain Generation for Evaluating Autonomous Off-Road Ground Vehicles,” *Journal of Autonomous Vehicles and Systems*, vol. 5, no. 2, p. 021003, Apr. 2025, doi: 10.1115/1.4067769.  

[2] K. Perlin, “An image synthesizer,” *SIGGRAPH Comput. Graph.*, vol. 19, no. 3, pp. 287–296, July 1985, doi: 10.1145/325165.325247.  

[3] “A multivariate Weierstrass–Mandelbrot function,” *Proc. R. Soc. Lond. A*, vol. 400, no. 1819, pp. 331–350, Aug. 1985, doi: 10.1098/rspa.1985.0083.  

[4] “On the Weierstrass-Mandelbrot fractal function,” *Proc. R. Soc. Lond. A*, vol. 370, no. 1743, pp. 459–484, Apr. 1980, doi: 10.1098/rspa.1980.0044.  

[5] J. J. Dawkins, D. M. Bevly, and R. L. Jackson, “Fractal terrain generation for vehicle simulation,” *Int. J. Vehicle Autonomous Systems*, vol. 10, no. 1/2, p. 3, 2012, doi: 10.1504/IJVAS.2012.047693.  

[6] X. Mei, P. Decaudin, and B.-G. Hu, “Fast Hydraulic Erosion Simulation and Visualization on GPU,” in *15th Pacific Conf. Comput. Graph. Appl.*, Maui, HI, USA: IEEE, Oct. 2007, pp. 47–56, doi: 10.1109/PG.2007.15.  

[7] A. D. Kelley, M. C. Malin, and G. M. Nielson, “Terrain simulation using a model of stream erosion,” in *Proc. 15th Annu. Conf. Comput. Graph. Interact. Tech.*, ACM, June 1988, pp. 263–268, doi: 10.1145/54852.378519.  

[8] C. Beckham and C. Pal, “A step towards procedural terrain generation with GANs,” arXiv:1707.03383, July 11, 2017, doi: 10.48550/arXiv.1707.03383.  

[9] G. Voulgaris, I. Mademlis, and I. Pitas, “Procedural Terrain Generation Using Generative Adversarial Networks,” in *2021 29th Eur. Signal Process. Conf. (EUSIPCO)*, Dublin, Ireland: IEEE, Aug. 2021, pp. 686–690, doi: 10.23919/EUSIPCO54536.2021.9616151.  

[10] S. Perche, A. Peytavie, B. Benes, E. Galin, and E. Guérin, “Authoring Terrains with Spatialised Style,” *Comput. Graph. Forum*, vol. 42, no. 7, p. e14936, Oct. 2023, doi: 10.1111/cgf.14936.  

[11] S. Perche, A. Peytavie, B. Benes, E. Galin, and E. Guérin, “StyleDEM: a Versatile Model for Authoring Terrains,” arXiv:2304.09626, Apr. 19, 2023, doi: 10.48550/arXiv.2304.09626.  

[12] F. Merizzi, “Procedural terrain generation with style transfer,” arXiv:2403.08782, Jan. 28, 2024, doi: 10.48550/arXiv.2403.08782.  

[13] A. Sharma et al., “EarthGen: Generating the World from Top-Down Views,” arXiv:2409.01491, Sept. 7, 2024, doi: 10.48550/arXiv.2409.01491.  

[14] P. Borne--Pons, M. Czerkawski, R. Martin, and R. Rouffet, “MESA: Text-Driven Terrain Generation Using Latent Diffusion and Global Copernicus Data,” arXiv:2504.07210, 2025, doi: 10.48550/arXiv.2504.07210.  

[15] Y. Hu, K. Wang, Y. Shao, J. Plass, Z. Wang, and K. Perlin, “Generative Terrain Authoring with Mid-air Hand Sketching in Virtual Reality,” in *Proc. 30th ACM Symp. Virtual Reality Softw. Technol.*, Trier, Germany: ACM, Oct. 2024, pp. 1–10, doi: 10.1145/3641825.3687736.  

[16] Y. I. H. Parish and P. Müller, “Procedural modeling of cities,” in *Proc. 28th Annu. Conf. Comput. Graph. Interact. Tech.*, ACM, Aug. 2001, pp. 301–308, doi: 10.1145/383259.383292.  

[17] R. B. D. d’Oliveira and A. L. A. Jr, “Procedural Planetary Multi-resolution Terrain Generation for Games,” arXiv:1803.04612, Mar. 13, 2018, doi: 10.48550/arXiv.1803.04612.  

[18] R. O. Chavez-Garcia, J. Guzzi, L. M. Gambardella, and A. Giusti, “Learning Ground Traversability From Simulations,” *IEEE Robot. Autom. Lett.*, vol. 3, no. 3, pp. 1695–1702, July 2018, doi: 10.1109/LRA.2018.2801794.  
