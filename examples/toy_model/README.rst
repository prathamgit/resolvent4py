Toy Model
=========

We use the toy model in :cite:`Padovan2020jfm` to demonstrate 
the use of :code:`resolvent4py` to perform the harmonic resolvent analysis.
The governing equations are 

    .. math::

        \begin{align}
            \dot{x} &= \mu x - \gamma y - \alpha x z - \beta x y\\
            \dot{y} &= \gamma x + \mu y - \alpha y z + \beta x^2\\
            \dot{z} &= -\alpha z + \alpha (x^2 + y^2)
        \end{align}

with parameters :math:`(\alpha, \beta, \gamma, \mu) = (1/5, 1/5, 1, 1/5)`.
For this choice of parameters, the origin is unstable and the state will settle
onto a time-periodic limit cycle.
We linearize the equations about this time-periodic solution and perform
the harmonic resolvent analysis as in :cite:`Padovan2020jfm`.


Instructions
------------

1. Generate the blocks of the harmonic balanced matrix using

   .. code-block:: bash

      mpiexec -n 1 python -u generate_matrices.py

   This script must be run in series, and its outputs will be written in
   :file:`data/`.

2. Run harmonic resolvent analysis with

   .. code-block:: bash

      mpiexec -n 2 python -u demonstrate_harmonic_resolvent.py

   This script can be run with any number of processors (although the
   dimension of the system is rather small, so there might not be any
   benefit in running it in parallel).

3. Navigate to the :file:`results/` directory to check out the results.

Scripts
-------