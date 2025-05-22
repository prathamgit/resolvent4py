# Resolvent4py

[![Tests](https://github.com/albertopadovan/resolvent4py/actions/workflows/tests.yml/badge.svg)](https://github.com/albertopadovan/resolvent4py/actions/workflows/tests.yml)

Resolvent4py is a petsc4py-based package for the analysis, model reduction
and control of large-scale linear systems. 

## Dependencies
- `Python>=3.9`
- `numpy`
- `scipy`
- `matplotlib`
- `pymanopt`
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


