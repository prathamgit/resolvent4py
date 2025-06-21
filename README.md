# Resolvent4py

[![Tests](https://github.com/albertopadovan/resolvent4py/actions/workflows/tests.yml/badge.svg)](https://github.com/albertopadovan/resolvent4py/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Size](https://img.shields.io/github/languages/code-size/albertopadovan/resolvent4py.svg)](https://github.com/albertopadovan/resolvent4py)


`resolvent4py` is a parallel Python toolbox to perform 
analysis, model reduction and control of high-dimensional linear systems. 
It relies on `mpi4py` for multi-processing parallelism, and it leverages 
the functionalities and data structures provided by `petsc4py` and `slepc4py`.
The goal of this project is to provide users with a friendly python-like
experience, while also leveraging the high-performance and parallel-computing
capabilities of the PETSc and SLEPc libraries.
The core of the package is an abstract class, called `LinearOperator`, which 
serves as a blueprint for user-defined child classes that can be used to
define any linear operator. 
`resolvent4py` currently ships with 5 linear operator subclasses:

- `MatrixLinearOperator`
- `LowRankLinearOperator`
- `LowRankUpdatedLinearOperator`
- `ProjectionLinearOperator`
- `ProductLinearOperator`

Once a linear operator is instantiated, `resolvent4py` currently allows for
several analyses, including:

- Right and left eigendecomposition using Arnoldi iteration (with shift and 
  invert)
- (Randomized) singular value decomposition (SVD)
- Resolvent analysis via randomized SVD (algebraic and with time stepping)
- Harmonic resolvent analysis via algebraic randomized SVD
- Balanced truncation for time-invariant linear systems using frequential Gramians

Additional functionalities (found in `resolvent4py/utils`) and available 
to the user through the `resolvent4py` namespace are:

- Support for parallel I/O through `petsc4py`
- Support for MPI communications using `mpi4py`
- Support for manipulation of PETSc matrices/vector and SLEPc BVs


## Dependencies
- `Python>=3.10`
- `numpy`
- `scipy`
- `matplotlib`
- `sphinx`
- `sphinx-rtd-theme`
- `sphinx-gallery`
- `sphinxcontrib-bibtex`
- `petsc4py`
- `slepc4py`
- `mpi4py`

## Installation instructions

### Installing PETSc, SLEPc, petsc4py and slepc4py

> **Note**  
> If you have an existing parallel build of PETSc and SLEPc and their
> 4py counterparts configured with complex scalars 
> (i.e., `--with-scalar-type=complex`) and with MUMPS (i.e.,
> `--download-mumps`) you can skip this subsection and go directly to
> "Installing `resolvent4py` and building the documentation".

All the dependencies above can be installed straightforwardly with `pip`, 
except for `petsc4py` and `slepc4py` whose installation is easier if 
`PETSc` and `SLEPc` are built from source.

- We recommend creating a clean Python environment.
- Download [PETSc](https://petsc.org/release/install/download/). Any version >= 
  3.20.0 should work. (The latest version that we tested is 3.23.3.)
- Configure PETSc using the command below,
    ```bash
    ./configure PETSC_ARCH=resolvent4py_arch --download-fblaslapack 
    --download-mumps --download-scalapack --download-parmetis 
    --download-metis --download-ptscotch --with-scalar-type=complex 
    --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3
    ```
  If some of the libraries above (e.g., `scalapack`, `metis`, etc.) are already
  available to the user, then see the `PETSc` [configuration guidelines](
  https://petsc.org/release/install/install/) for details on how to link against
  them.
- Follow the `PETSc` instructions (provided during the configuration step) to 
  build the library. Then make sure to export the environment variables
  `PETSC_DIR` and `PETSC_ARCH`.
- Install [SLEPc](https://slepc.upv.es/documentation/instal.htm). Any version >=
  3.20.0 should work. (The latest version that we tested was 3.23.1.)
- Install `mpi4py`, `petsc4py` and `slepc4py`
    ```bash
    pip install mpi4py petsc4py==petsc-version slepc4py==slepc-version
    ```
- Ensure that the installation was successful by running
    ```bash
    python -c "from mpi4py import MPI"
    python -c "from petsc4py import PETSc"
    python -c "from slepc4py import SLEPc"
    ```

## Installing `resolvent4py` and building the documentation

- Clone the repository into the local directory `resolvent4py`
- Navigate to the directory `resolvent4py` and run
    ```bash
        pip install resolvent4py[docs]
    ```
- Navigate to docs folder 
    ```bash
        cd docs
    ```
- Build the documentation and open it with the following commands
    ```bash
        ./compile_html.sh && open build/html/index.html
    ```