# FYS-STK4155 - Project 3 - Solving PDEs using Physics Informed Neural Networks (PINNs)
Repository for Project 3 in FYS-STK4155 - Solving PDEs using Physics Informed Neural Networks<br /><br />
Here you'll find the programs, project report, and some additional results produced for the second project in the subject FYS-STK4155 at UiO.<br /><br />
The folder structure is:
- **01-main**: <br />Folder for the notebooks, programs, class-files and other support files used in the project.
- **02-doc**: <br /> The written report and .tex-files used to generate the report. The report abstract, introduction and some conclusions are given below.
- **03-results**: <br />A selection of figures and results generated during the project.
- **09-archive**: <br />Preliminary files not used in the final version of the project
<br /><br />

## Running the project programs
The project programs can be found in the **01-main** folder. They are organized in different Jupyter notebooks, and can be run given that the user downloads all files in **01-main**, and keeps them in a shared directory. <br />

The notebooks imports the classes and methods from three separate files. The main classes can be found in:
- **Plain-PINN**: _network.py_ 
- **TF-PINN**: _networkFlow.py_
- **FD-solvers**: _FiniteDiff.py_

These have dependencies on depend on:
- _support.py_: Contains the activation function, plotting methods and GD-classes
- _PDEq.py_: Contains the methods for including the PDEs into the PINNs

The notebook also depends on a number of different python packages, such as _autograd_, _tensorflow_, _seaborn_, _matplotlib_ and _numpy_, so these are required to run the notebook.

## Project abstract and introduction, and main conclusions and findings
### Abstract
In this project, I am implementing two different Physics Informed Neural Networks (PINNs) with the aim of studying their applicability for finding solutions to partial differential equations (PDEs). My main goal is to see how the PINN-approach holds up against a traditional numerical method for solving PDEs, the finite difference method (FDM). The PDEs I use are the unsteady diffusion and wave equations in both one and two spatial dimensions. One of the PINNs are implemented using TensorFlow to build the network, and train it, with the other making use of Autograd together with my own feed-forward and gradient descent methods. My analysis will include methods for model selection, like standard and random grid searches, and gradient descent methods that use adaptive learning rates, such as ADAM. The analysis will also involve the impact from parameters such as network depth, learning rates, regularization, among others, as well as some of the method choices where I look at different activation functions. My finding is that the networks are capable of solving the one-dimensional equations well, but struggle when the problem is extended to two dimensions. As an example, for the 1D diffusion equation, the PINNs obtained global errors comparable to the FDM-solver, namely $E = 0.01105$, against the FDM's $E = 0.00247$.

### Introduction
Partial differential equations (PDEs) are powerful tool for the study of natural phenomena and physical systems, as they seek to describe and model the underlying behavior of a system using a specific relationships between an unknown function of two or more independent variables, and its derivatives. 

Standard methods for finding solutions to PDEs such as the _Finite Difference Method_ (FDM) and _Finite Element Method_ (FEM) are powerful and well-proven, but the methods also become more and more involved as the complexity of the system grows. They also require an implementation that is more or less tailored to a specific problem.

Several methods that utilize the strengths of neural networks (NNs) have been proposed in the recent years as models to solve PDEs. One of the main strengths is that a NN usually is more general purpose than the traditional approaches. This is especially noticeable for non-linear systems, and for system where shocks and strong convection is present, where FDM and FEM are known to struggle.

In this project, I have focused on an approach commonly called _Physics-Informed Neural Networks_ (PINNs), which can be seen as a specialization of a standard, fully-connected _Feed-Forward Neural Network_ (FFNN). The PINNs uses information from the PDEs to create a generative model that is able to predict the behavior of the physical system.

My aim is to show how two different approaches of implementing a PINN perform against a FDM implementation that solves the unsteady wave and diffusion equations in up to two spatial dimensions. The PINN-implementations are based on chapter 15 in ([Ch.15, FYS-STK3155/4155](https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter11.html)) and the companion programs to the paper ([M. Raissi et.al.](https://maziarraissi.github.io/PINNs/)). Both uses an _automatic differentiation_ approach for computing gradients and derivatives. The latter also uses the extensive _TensorFlow_-library when creating and training the network, while the former uses my own implementations to do the same.

For the PINNs, I will also show how different hyperparameters and the network structure choices impact the solution, as well as the choices related to different activation functions. For verification of both the PINN- and FDM- approach, I am using known analytical solutions to the two PDEs to compute performance metrics like the global error.

The report will go through the relevant theoretical concepts and methods in Sec.II, first describing the general concept of PDEs, and the building blocks of the neural network, with specific focus on how information from the physics come into the picture. Following this, I give a quick breakdown of the FDM, and finish off with a description of the model selection and assessment process.

Sec.III describes the program-, and class-structure of the implementation, as well as a quick breakdown of the basic training and solution algorithms I'm using in the project. Here, I will also give a short introduction to how to access and use the programs written as part of this project.

In the final parts, Sec.IV. and Sec.V, I will show a selection of results from the model selection process, and the simulations, and also present a comparison between the methods. Some notable highlights from the analysis is that the PINN-predictions of the 1D diffusion equation give a global $\ell^{2}$-errors, $E = \{0.02622,\ 0.01105\}$, while the FDM for the same problem yields an error of $E = 0.00247$. The 2D-cases did not work as intended, with the network scoring $E = 0.35431$, compared to the FDMs score of $E = 0.00045$, which is a significant difference.

In the appendices, I present some additional results from the model selection process and analysis, as well as provide some more detail activation functions, the PINN-modeling choices, and discretization using finite differences.

### Conclusion and findings
Some of the main conclusions from the project are:
- PINNs are a viable numerical method for solving PDEs
- They do not perform as well as the FDM on these particular PDEs, as they are linear, making the FDM-solutions very close to exact
- Using the _SiLU_ and _GELU_ activation functions gave the best results
- This particular implementation struggles with 2D equations, but this may be due to my program 
