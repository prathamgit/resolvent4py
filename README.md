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

If you use `resolvent4py` in your workflow, please cite [this](https://www.sciencedirect.com/science/article/pii/S2352711025002523) paper.
<details>
<summary>BibTeX</summary>

```bibtex
@article{PADOVAN2025102286,
  title = {Resolvent4py: A parallel Python package for analysis, model reduction and control of large-scale linear systems},
  journal = {SoftwareX},
  volume = {31},
  pages = {102286},
  year = {2025},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2025.102286},
  url = {https://www.sciencedirect.com/science/article/pii/S2352711025002523},
  author = {Alberto Padovan and Vishal Anantharaman and Clarence W. Rowley and Blaine Vollmer and Tim Colonius and Daniel J. Bodony},
}
```
</details> 

## Documentation

Click [here](https://albertopadovan.github.io/resolvent4py/).

## Dependencies

- `Python>=3.10`
- `numpy`
- `scipy`
- `matplotlib`
- `mpi4py`
- `petsc4py >= 3.20` (must be installed from source, see below)
- `slepc4py >= 3.20` (must be installed from source, see below)



## Installation instructions

### Installation using `conda` and `pip` (recommended)

1. Create a new `conda` environment with, e.g., Python 3.11
    ```bash
    conda create -n resolvent4py_env python=3.11
    conda activate resolvent4py_env
    ```
    > **IMPORTANT:**
    > Please ensure that the environment variables `PETSC_DIR`, `PETSC_ARCH`
    > and `SLEPC_DIR` are unset.
    > This prevents `petsc4py` and `slepc4py` from binding to
    > unintended preexisting PETSc/SLEPc builds.
    > Likewise, ensure that any pre-installed MPI toolchains 
    > (e.g., from system modules or previous installations) are not 
    > interfering with the current build environment. That is, 
    > ensure that the output of `which mpicc` is empty. If it is not, then
    > this indicates a potential conflict.

2. Run the command
    ```
    conda search 'petsc=3.23.3=complex*' --channel conda-forge --info
    ```
   to list all `PETSc` 3.23.3 builds with support for complex scalars and MUMPS.
   This command will display metadata for each matching build, including its dependencies.
   Inspect the list of dependencies and look for "mumps" or "mumps-mpi."
   You will likely find two candidate builds that satisfy the MUMPS and complex
   scalar requirements: one built with MPICH and one with OpenMPI. 
   Select the one you prefer
   and identify the corresponding build string 
   (e.g., `complex_h69b5c76_0` on a Mac osx-arm64).
   
4. Install `PETSc` and `petsc4py` with
    ```bash
    conda install -c conda-forge petsc=3.23.3=build-string petsc4py
    ```

5. Run
    ```
    conda search 'slepc=3.23.2=complex*' --channel conda-forge --info
    ```
   to perform the same inspection as above and identify a compatible `SLEPc`
   build (e.g., `complex_he96486c_0` on a Mac osx-arm64). 
   If you chose the OpenMPI build in step 3, make sure you select
   the OpenMPI-compatible build of `SLEPc`.

6. Install `SLEPc` and `slepc4py` with
    ```bash
    conda install -c conda-forge slepc=3.23.2=build-string slepc4py
    ```
4. Install `mpi4py`
    ```bash
    conda install -c conda-forge mpi4py
    ```
5. Install `resolvent4py`
    ```bash
    pip install resolvent4py
    ```


### Installation from source

> **Note**  
> If you have an existing parallel build of PETSc and SLEPc and their
> 4py counterparts configured with complex scalars 
> (i.e., `--with-scalar-type=complex`) and with MUMPS (i.e.,
> `--download-mumps`), you can go directly to step 10 (after running 
> `pip install mpi4py`).


1. We recommend creating a clean Python environment using, e.g., `venv` or `conda`.
Ensure that you are using a Python version >= 3.10 by running 
`python --version` in your terminal.
2. Ensure valid C, C++, and Fortran compilers are available, 
along with ```make``` and ```flex```,
which can be obtained through a package management CLI.
3. Download [PETSc](https://petsc.org/release/install/download/). Any version >= 
  3.20.0 should work. (The latest version that we tested is 3.23.3.)
4. Consult the PETSc [configuration guidelines](https://petsc.org/release/install/install/)
to configure PETSc with complex scalars (i.e., `--with-scalar-type=complex`) and 
with MUMPs (i.e., `--download-mumps`).
For reference, here is the configure command (to be run inside the PETSc directory)
that has worked for us in the past,
    ```bash
    ./configure PETSC_ARCH=resolvent4py_arch --download-fblaslapack \
    --download-mumps --download-scalapack --download-parmetis \
    --download-metis --download-ptscotch --with-scalar-type=complex \
    --download-mpich --download-cmake --download-bison \
    --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3
    ```
    > **Note**  
    > Not all configure options shown in the multiline command above are necessary 
    > for all users. For example, if an MPI
    > compiler is already available, then `--download-mpich` may
    > not be necessary and you can pass configure flags like
    > `--with-cc=mpicc`. (Once again, please consult the PETSc 
    > [configuration guidelines](https://petsc.org/release/install/install/) for
    > your specific case.)
5. Follow the PETSc instructions (provided during the configuration step) to 
  build the library. Then make sure to export the environment variables
  `PETSC_DIR` and `PETSC_ARCH`.
6. If you downloaded mpich during the configuration stage, then run the following
  command to reference the correct MPI installation,
    ```bash
       export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH \
       export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
    ```
7. Install [SLEPc](https://slepc.upv.es/documentation/instal.htm). Any version >=
  3.20.0 should work. (The latest version that we tested was 3.23.1.)
8. Install `mpi4py`, `petsc4py` and `slepc4py`
    ```bash
    pip install mpi4py petsc4py==petsc-version slepc4py==slepc-version
    ```
    > **Note**  
    > It is critical that you install the versions of `petsc4py` and `slepc4py` 
    > corresponding to your PETSc and SLEPc installations. Otherwise, the
    > installation will likely fail.
    
9. Ensure that the installation was successful by running
    ```bash
    python -c "from mpi4py import MPI"
    python -c "from petsc4py import PETSc"
    python -c "from slepc4py import SLEPc"
    ```
10. Install `resolvent4py` with
    ```bash
        pip install resolvent4py
    ```