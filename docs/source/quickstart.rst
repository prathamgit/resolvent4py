Quickstart
==========

Running the examples
--------------------

- Navigate to `resolvent4py/examples/cgl/`
- Run `mpiexec -n 1 python3 -u generate_matrices.py` to create and store
    the operators associated with the linearized complex Ginzburg-Landau (CGL)
    equation
- Run `mpiexec -n 3 python3 -u demonstrate_eigendecomposition.py` to 
    compute the CGL eigenvalues using 3 MPI processors


