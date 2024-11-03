from .linear_operator import LinearOperator
from ..linalg import mat_solve_hermitian_transpose
from petsc4py import PETSc
from slepc4py import SLEPc

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
            and :code:`solve_hermitian_transpose()` methods. If a :code:`KSP` is 
            provided, its type and the type of :code:`A` must be compatible.
        :type ksp: `KSP`_
        :param nblocks: [optional] number of blocks (if the linear operator \
            has block structure). This must be an odd number.
        :type nblocks: int
    """
    def __init__(self, comm, A, ksp=None, nblocks=None):
        self.A = A
        self.ksp = ksp
        super().__init__(comm, 'MatrixLinearOperator', A.getSizes(), nblocks)
    
    def apply(self, x, y=None):
        y = self.create_left_vector() if y == None else y
        self.A.mult(x,y)
        return y
    
    def apply_mat(self, X, Y=None):
        Xm = X.getMat()
        if Y != None:
            Ym = Y.getMat()
            Ym = self.A.matMult(Xm, Ym) 
            Y.restoreMat(Ym)
        else:
            Ym = Xm.duplicate()
            Ym = self.A.matMult(Xm, Ym) 
            Y = SLEPc.BV().createFromMat(Ym)
            Y.setType('mat')
            Ym.destroy()
        X.restoreMat(Xm)
        return Y

    def apply_hermitian_transpose(self, x, y=None):
        y = self.create_right_vector() if y == None else y
        self.A.multHermitian(x,y)
        return y
    
    def apply_hermitian_transpose_mat(self, X, Y=None):
        self.A.hermitianTranspose()
        Xm = X.getMat()
        if Y != None:
            Ym = Y.getMat()
            Ym = self.A.matMult(Xm, Ym) 
            Y.restoreMat(Ym)
        else:
            Ym = Xm.duplicate()
            Ym = self.A.matMult(Xm, Ym) 
            Y = SLEPc.BV().createFromMat(Ym)
            Y.setType('mat')
            Ym.destroy()
        X.restoreMat(Xm)
        self.A.hermitianTranspose()
        return Y
    
    def solve(self, x, y=None):
        if self.ksp != None:
            y = self.create_left_vector() if y == None else y
            self.ksp.solve(x,y)
            return y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve(): "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve() method."
            )

    def solve_mat(self, X, Y=None):
        if self.ksp != None:
            Xm = X.getMat()
            if Y != None:
                Ym = Y.getMat()
                self.ksp.matSolve(Xm, Ym) 
                Y.restoreMat(Ym)
            else:
                Ym = Xm.duplicate()
                self.ksp.matSolve(Xm, Ym) 
                Y = SLEPc.BV().createFromMat(Ym)
                Y.setType('mat')
                Ym.destroy()
            X.restoreMat(Xm)
            return Y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve(): "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve() method."
            )

    def solve_hermitian_transpose(self, x, y=None):
        if self.ksp != None:
            y = self.create_right_vector() if y == None else y
            x.conjugate()
            self.ksp.solveTranspose(x, y)
            x.conjugate()
            y.conjugate()
            return y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve_hermitian_transpose(). "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve_hermitian_transpose() method."
            )
        
    def solve_hermitian_transpose_mat(self, X, Y=None):
        if self.ksp != None:
            Xm = X.getMat()
            if Y != None:
                Ym = Y.getMat()
                Ym = mat_solve_hermitian_transpose(self.ksp, Xm, Ym)
                Y.restoreMat(Ym)
            else:
                Ym = Xm.duplicate()
                Ym = mat_solve_hermitian_transpose(self.ksp, Xm, Ym)
                Y = SLEPc.BV().createFromMat(Ym)
                Y.setType('mat')
                Ym.destroy()
            X.restoreMat(Xm)
            return Y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve(): "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve() method."
            )
        
    def destroy(self):
        self.A.destroy()
        self.ksp.destroy()