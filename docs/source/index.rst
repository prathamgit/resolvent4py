.. Resolvent4py documentation master file, created by
   sphinx-quickstart on Mon Oct  7 23:00:02 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Resolvent4py
===========================

Resolvent4py is a petsc4py-based toolbox to perform linear analyses of 
high-dimensional dynamical systems. 
The goal of this project is to provide users with a friendly python-like
experience, while also leveraging the high-performance and parallel-computing
capabilities of the PETSc library.
Current functionalities include:

- Right and left eigendecomposition 
- Resolvent analysis (algebraic and using time-stepping techniques)
- Harmonic resolvent analysis
- Balanced truncation (time-invariant and time-periodic)
- One-Way Navier-Stokes (OWNS)


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
   quickstart.rst
   api-reference.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   auto_examples/toy_model/index



