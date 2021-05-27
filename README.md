# Pathtracer
A Monte Carlo Path Tracer utilizing NVIDIA OptiX.

- Direct Lighting
- Indirect Lighting
- Phong Sampling
- Importance Sampling
- Stratified Sampling
- Next Event Estimation
- Russian Roulette

Build using CMake and inputting a file to render.
Examples:

> make
> 
> ./build/bin/OptiXRenderer Scenes/hw2/cornell.test

![cornell](https://user-images.githubusercontent.com/84567020/119596756-a1a9c780-bd94-11eb-9ebe-3c270cb34921.png)

> make
> 
> ./build/bin/OptiXRenderer Scenes/hw3/dragon.test

![dragon](https://user-images.githubusercontent.com/84567020/119596558-437ce480-bd94-11eb-867f-d956c17c98c0.png)

More example images included in the hw# folders
