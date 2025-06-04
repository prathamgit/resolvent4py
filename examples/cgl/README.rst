Complex Ginzburg-Landau Equation
================================

In this suite of examples, we perform several linear analyses of the linearized 
complex Ginzburg-Landau (CGL) equation:

.. math::

    \partial_t q = \left(-\nu \partial_x + \
                \gamma\partial_x^2 + \mu\right)q,\quad q(x,t)\in\mathbb{C}.

The parameters :math:`\nu`, :math:`\gamma` and :math:`\mu` are chosen so that 
the origin :math:`q(x) = 0` is stable, as in Table 1 of :cite:`Ilak2010model`.
The spatial discretization is performed using a fourth-order central difference
scheme (see :code:`cgl.py` for details), and the discretized system may then 
be written compactly as

.. math::

    \frac{d}{dt} q = A q,\quad q\in\mathbb{C}^n,

where now :math:`q` denotes the spatially-discretized state vector.
This examples included here are:

- :code:`demonstrate_eigendecomposition.py` for linear stability analysis
- :code:`demonstrate_rsvd.py` for resolvent analysis in the frequency domain
- :code:`demonstrate_rsvd_dt.py` for resolvent analysis in the time domain
- :code:`demonstrate_balanced_truncation.py` for balanced model reduction


Instructions
------------

1. Generate the data matrices with

   .. code-block:: bash

      mpiexec -n 1 python -u generate_matrices.py

   This script must be run in series, and its outputs will be written in
   :file:`data/`.

2. Run any script :code:`demonstrate_*.py` with

   .. code-block:: bash

      mpiexec -n 2 python -u demonstrate_*.py

   These script can be run with any number of processors (although the
   dimension of the system is rather small, so there might not be any
   benefit in running it in parallel).

3. Navigate to the :file:`results/` directory to check out the results.

Scripts
-------
