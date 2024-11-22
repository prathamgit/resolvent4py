class myBackTrackingLineSearcher:
    """Back-tracking line-search algorithm."""

    def __init__(
        self,
        contraction_factor=0.5,
        optimism=2,
        sufficient_decrease=1e-4,
        max_iterations=25,
        initial_step_size=1,
    ):
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.sufficient_decrease = sufficient_decrease
        self.max_iterations = max_iterations
        self.initial_step_size = initial_step_size

        self._oldf0 = None

def search(self, objective, manifold, x, d, f0, df0):
    """Function to perform backtracking line search.

    Args:
        objective: Objective function to optimize.
        manifold: The manifold to optimize over.
        x: Starting point on the manifold.
        d: Tangent vector at ``x``, i.e., a descent direction.
        df0: Directional derivative at ``x`` along ``d``.

    Returns:
        A tuple ``(step_size, newx)`` where ``step_size`` is the norm of
        the vector retracted to reach the suggested iterate ``newx`` from
        ``x``.
    """
    # Compute the norm of the search direction
    norm_d = manifold.norm(x, d)

    if self._oldf0 is not None:
        # Pick initial step size based on where we were last time.
        alpha = 2 * (f0 - self._oldf0) / df0
        # Look a little further
        alpha *= self.optimism
    else:
        alpha = self.initial_step_size / norm_d
    alpha = float(alpha)

    # Make the chosen step and compute the cost there.
    newx = manifold.retraction(x, alpha * d)
    newf = objective(newx)
    step_count = 1

    # Backtrack while the Armijo criterion is not satisfied
    while (
        newf > f0 + self.sufficient_decrease * alpha * df0
        and step_count <= self.max_iterations
    ):

        # Reduce the step size
        alpha = self.contraction_factor * alpha

        # and look closer down the line
        newx = manifold.retraction(x, alpha * d)
        newf = objective(newx)

        step_count = step_count + 1

    # If we got here without obtaining a decrease, we reject the step.
    if newf > f0:
        alpha = 0
        newx = x

    step_size = alpha * norm_d

    self._oldf0 = f0

    return step_size, newx