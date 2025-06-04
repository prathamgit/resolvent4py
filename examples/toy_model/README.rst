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

