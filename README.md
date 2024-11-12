# FYS-STK4155 - Project 3 - _project title_
Main idea is to do some sort of PINN - implementation for analysis of PDE's, hopefully in two dimensions. 
## Possible Topics
- Buckely-leverett equations - replicating shock-front - solving PDE or reproducing results from dataset (https://perminc.com/resources/fundamentals-of-fluid-flow-in-porous-media/chapter-4-immiscible-displacement/buckley-leverett-theory/)
- 2D - diffusion, and/or 2D-wave equation (solutions to both using FD in MAT-MEK4270?) See https://matmek-4270.github.io/matmek4270-book/lecture13.html#the-diffusion-equation for possible solutions to compare against and how to set up using FEM 

## Constraints for regression/PINN
 - Need to be able to have an "analytical" solution for comparison. _Can that be a manufactured solution_?
 - Need to have additional method of solving the equations, FD or FEM from MAT-MEK4270. Examples equations and solver from here might be a way to go. Wave equation and Diffusion equation for u(x,t) in MATMEK4270-course
 - PDE needs to be well-posed, with a unique solution
