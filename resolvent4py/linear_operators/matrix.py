from .linear_operator import LinearOperator

class MatrixLinearOperator(LinearOperator):
    r"""
        Class for a linear operator :math:`L = A`, where :math:`A` is a matrix.
        In general, :code:`A` can be any matrix (rectangular or square, 
        invertible or non-invertible). If a :code:`ksp` is provided, however, 
        :code:`A` must be invertible.

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param A: a PETSc matrix
        :type A: `MatType`_
        :param ksp: [optional] a PETSc KSP object to enable the :code:`solve()`
            and :code:solve_hermitian_transpose()` methods. If a KSP is 
            provided, its type and the type of A must be compatible.
        :type ksp: `KSP`_
    """

    def __init__(self, comm, A, ksp=None):

        super().__init__(comm, 'MatrixLinearOperator', A.getSizes())
        self.A = A
        self.ksp = ksp
        
    def apply(self, x):
        y = self.create_left_vector()
        self.A.mult(x,y)
        return y
    
    def apply_hermitian_transpose(self, x):
        y = self.create_right_vector()
        self.A.multHermitian(x,y)
        return y
    
    def solve(self, x):
        if self.ksp != None:
            y = self.create_left_vector()
            self.ksp.solve(x,y)
            return y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve(): "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve() method."
            )
        
    def solve_hermitian_transpose(self, x):
        if self.ksp != None:
            y = self.create_right_vector()
            x.conjugate()
            self.ksp.solveTranspose(x,y)
            x.conjugate()
            y.conjugate()
            return y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve_hermitian_transpose(). "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve_hermitian_transpose() method."
            )
        
    def destroy(self):
        self.A.destroy()
        self.ksp.destroy() if self.ksp != None else None
