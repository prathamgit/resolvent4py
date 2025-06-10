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
- Resolvent analysis via randomized SVD
- Harmonic resolvent analysis via randomized SVD
- Balanced truncation for time-invariant linear systems

Additional functionalities (found in `resolvent4py/utils`) and available 
to the user through the `resolvent4py` namespace are:

- Support for parallel I/O through `petsc4py`
- Support for MPI communications using `mpi4py`
- Support for manipulation of PETSc matrices/vector and SLEPc BVs
- Support for time-stepping of linear ordinary differential equations


## Dependencies
- `Python>=3.9`
- `numpy`
- `scipy`
- `matplotlib`
- `petsc4py>=3.20`
- `slepc4py>=3.20`
- `mpi4py`

## Instructions

### Installing the dependencies

All the requirements above can be installed straightforwardly with `pip`. 
The installation of `PETSc` and `SLEPc` and their `4py` counterparts can be 
more cumbersome, and is outlined in detail below.


- Download [PETSc](https://petsc.org/release/install/download/) either by 
    cloning the repository or by downloading a tarball file. Any version >= 3.20
    should work. Development was done with version 3.20.1.
- Configure PETSc following the [configuration guidelines](
    https://petsc.org/release/install/install/). A required configuration option
    is `--with-scalar-type=complex`, and required external libraries include
    `mumps`, `scalapack`, `metis`, `parmetis`, `bison` and `ptscotch`.
    For example, our version of PETSc was configured as follows:

    ```bash
    ./configure PETSC_ARCH=resolvent4py_arch --with-cc=gcc --with-cxx=g++ 
    --with-fc=gfortran --download-mpich --download-fblaslapack 
    --download-mumps --download-scalapack --download-parmetis 
    --download-metis --download-ptscotch --download-bison 
    --with-scalar-type=complex --with-debugging=0 
    COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3
    ```
- Build with
    ```bash
    make PETSC_DIR=/your/petsc/dir PETSC_ARCH=resolvent4py_arch all 
    ```
- Check the build with
    ```bash
    make all check
    ```
- Make sure that `PETSC_DIR` and `PETSC_ARCH` have been exported 
    correctly by running
    ```bash
    echo $PETSC_DIR && echo $PETSC_ARCH
    ```
- Install [SLEPc](https://slepc.upv.es/documentation/instal.htm)
- Install `petsc4py` and `slepc4py`
    ```bash
    pip install petsc4py==petsc-version slepc4py==slepc-version 
    ```

## Installing `resolvent4py` and building the documentation

- Clone the repository into the local directory `resolvent4py`
- Navigate to the directory `resolvent4py` and run
    ```bash
        pip install .
    ```
- Unless already available, download `sphinx` to build the documentation
    ```bash
        pip install sphinx sphinx-rtd-theme
    ```
- Build the documentation and open it with
    ```bash
        ./compile_html.sh && open build/html/index.html
    ```


