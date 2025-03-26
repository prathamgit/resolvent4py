# Resolvent4py

Resolvent4py is a petsc4py-based package for the analysis, model reduction
and control of large-scale linear systems. 

## Installation Instructions

### Install PETSc

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
- After configuring, build and check the installation
    ```bash
    make PETSC_DIR=/your/petsc/dir PETSC_ARCH=resolvent4py_arch all && make check all
    ```

- 

- Install `sphinx` using `pip install sphinx sphinx-rtd-theme` unless already 
available
- Clone the `resolvent4py` package into the local directory `resolvent4py`


