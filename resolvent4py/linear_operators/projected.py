from .linear_operator import LinearOperator
from ..petsc4py_helper_functions import compute_dense_inverse

class ProjectedLinearOperator(LinearOperator):
    r"""
        Class for a linear operator of the form :math:`L = PAP`,
        where :math:`A` is an instance of any subclass of the 
        `resolvent4py.linear_operators.linear_operator.LinearOperator` 
        class, and :math:`P` is a projection operator of the form

        .. math::

            P = \begin{cases} 
            \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^* & \text{if }\ 
                \text{complement} = \text{False} \\
            I - \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^*  & \text{if }\ 
                \text{complement} = \text{True}
            \end{cases}

        If the :code:`A.solve()` method is enabled, then the :code:`solve()` 
        method for this class returns

        .. math::

            y = P \circ f \circ Px,
        
        where :math:`f(z)` denotes the output of :code:`A.solve(z)`.


        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param A: instance of a subclass of the
            `resolvent4py.linear_operator.LinearOperator` class
        :param Phi: dense PETSc matrix
        :type Phi: PETSc.Mat.Type.DENSE
        :param Psi: dense PETSc matrix
        :type Psi: PETSc.Mat.Type.DENSE
        :param complement: [optional] specifies which projection operator
        :type complement: Bool
        :param nblocks: [optional] number of blocks (if the linear operator \
            has block structure)
        :type nblocks: int
    """
    
    def __init__(self, comm, A, Phi, Psi, complement=False, nblocks=None):

        super().__init__(comm, 'ProjectedLinearOperator', \
                         A.get_dimensions(), nblocks)
        self.A = A
        self.Phi = Phi.copy()
        self.Psi = Psi.copy()
        self.complement = complement
        self.enforce_biorthogonality()
        self.real = self.check_if_real_valued()
        self.block_cc = self.check_if_complex_conjugate_structure() if \
            self.get_nblocks() != None else None
        

    def enforce_biorthogonality(self):
        r"""
            Enforce biorthogonality
            :math:`\Phi\leftarrow \Phi\left(\Psi^*\Phi\right)^{-1}`.

            :rtype: None
        """
        self.Psi.conjugate()
        self.Phi = self.Phi.matMult(compute_dense_inverse(\
            self.get_comm(),self.Psi.transposeMatMult(self.Phi)))
        self.Psi.conjugate()
    
    def apply_projection(self, x):
        r"""            
            :param x: a PETSc vector
            :type x: PETSc.Vec

            :return: :math:`P x`
            :rtype: PETSc.Vec
        """
        z, y = self.Phi.createVecs()
        if self.complement:
            self.Psi.multHermitian(x,z)
            self.Phi.mult(z,y)
            y.axpy(-1.0,x)
            y.scale(-1.0)
        else:
            self.Psi.multHermitian(x,z)
            self.Phi.mult(z,y)
        return y
    
    def apply_projection_hermitian_transpose(self, x):
        r"""            
            :param x: a PETSc vector
            :type x: PETSc.Vec

            :return: :math:`P^* x`
            :rtype: PETSc.Vec
        """
        z, y = self.Phi.createVecs()
        if self.complement:
            self.Phi.multHermitian(x,z)
            self.Psi.mult(z,y)
            y.axpy(-1.0,x)
            y.scale(-1.0)
        else:
            self.Phi.multHermitian(x,z)
            self.Psi.mult(z,y)
        return y
    
    def apply(self, x):
        return self.apply_projection(self.A.apply(self.apply_projection(x)))

    def apply_hermitian_transpose(self, x):
        return self.apply_projection_hermitian_transpose(\
                self.A.apply_hermitian_transpose(\
                self.apply_projection_hermitian_transpose(x)))
    
    def solve(self, x):
        return self.apply_projection(self.A.solve(self.apply_projection(x)))
    
    def solve_hermitian_transpose(self, x):
        return self.apply_projection_hermitian_transpose(\
                self.A.solve_hermitian_transpose(\
                self.apply_projection_hermitian_transpose(x)))

    def destroy(self):
        r"""
            Destroys all attributes of the class, except for :code:`A`,
            which is itself a class with its own :code:`destroy()` method.
        """
        self.Phi.destroy()
        self.Psi.destroy()