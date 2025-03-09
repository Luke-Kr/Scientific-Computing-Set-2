## Scientific-Computing-Set-2

## 2.1 Diffusion Limited Aggregation

A python implementation of DLA making use of the diffusion equation solved by SOR.  This simulation models the growth of fractal-like structures that form when the growth grows in accordance with the diffusion of nutrients.

Upon running the code an animation of the growth will be played according to the parameters defined in the file.

## 2.2 Monte Carlo Diffusion Limited Aggregation (DLA) Simulation

A Python implementation of Monte Carlo Diffusion Limited Aggregation with variable sticking probabilities. This simulation models the growth of fractal-like structures that form when particles undergoing random motion attach to a growing aggregate.

The simulation shows how different sticking probabilities affect the resulting cluster morphology, including:
- Cluster width
- Cluster height
- Cluster density

Upon running the code, some example clusters given p_r will be saved under fig/p_sweep. The resulting statistics will be saved under fig/cluster_metrics.png

## 2.3 Gray-Scott Model Simulation

A Python implementation of the Gray-Scott reaction-diffusion model. This simulation models the interaction between two chemicals, U and V, which diffuse and react with each other on a 2D grid. The model supports various boundary conditions including periodic, Dirichlet, Neumann, and Robin.

The simulation demonstrates how different parameters affect the resulting patterns, including:
- Diffusion rates (Du, Dv)
- Feed rate (f)
- Kill rate (k)
- Boundary conditions

Upon running the code, the resulting concentration of chemical V will be saved as an image under fig/gray_scott.png.

## Notebook

Run notebook.ipynb in order to have all relevant, output generating functions in one file.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Numba

