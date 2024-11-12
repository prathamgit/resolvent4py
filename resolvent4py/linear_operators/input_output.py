from .linear_operator import LinearOperator
from ..linalg import create_AIJ_identity

class InputOutputLinearOperator(LinearOperator):
    r"""
        Class for a linear operator 

        .. math::

            L = \underbrace{W^{1/2}_x C^*}_{\tilde{C}^*\
                \underbrace{\left(i\omega I - A\right)^{-1}}_{R_{\omega}^{-1}}\
                \underbrace{B W^{-1/2}_f}_{\tilde{B}}

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param Rom_inv: an instance of the linear operator class with a valid
            :code:`ksp` structure
        :param Btil: [optional] an instance of the linear operator class. 
            Equal to the identity operator by default
        :param Ctil: [optional] an instance of the linear operator class. 
            Equal to the identity operator by default
    """
    def __init__(self, comm, Rom_inv, Btil=None, Ctil=None):

        self.Rom_inv = Rom_inv
        sizesR = self.Rom_inv.get_dimensions()
        self.Btil = Btil if Btil != None else create_AIJ_identity(comm, sizesR)
        self.Ctil = Ctil if Ctil != None else create_AIJ_identity(comm, sizesR)
        sizes = (self.Ctil.get_dimensions()[-1], self.Btil.get_dimensions()[-1])
        nblocks = None
        if self.Rom_inv._nblocks == self.Btil._nblocks and \
            self.Btil._nblocks == self.Ctil._nblocks:
            nblocks = self.Ctil._nblocks
        super().__init__(comm, 'InputOutputLinearOperator', sizes, nblocks)
    
    
    def apply(self, x, y=None):
        y = self.create_left_vector() if y == None else y
        z1 = self.Btil.apply(x)
        z2 = self.Rom_inv.solve(z1)
        self.Ctil.apply_hermitian_transpose(z2, y)
        z1.destroy()
        z2.destroy()
        return y
    
    def apply_mat(self, X, Y=None):
        Z1 = self.Btil.apply_mat(X)
        Z2 = self.Rom_inv.solve_mat(Z1)
        Y = self.Ctil.apply_hermitian_transpose_mat(Z2) if Y == None \
            else self.Ctil.apply_hermitian_transpose_mat(Z2, Y)
        Z1.destroy()
        Z2.destroy()
        return Y

    def apply_hermitian_transpose(self, x, y=None):
        y = self.create_right_vector() if y == None else y
        z1 = self.Ctil.apply(x)
        z2 = self.Rom_inv.solve_hermitian_transpose(z1)
        self.Btil.apply_hermitian_transpose(z2, y)
        z1.destroy()
        z2.destroy()
        return y
    
    def apply_hermitian_transpose_mat(self, X, Y=None):
        Z1 = self.Ctil.apply_mat(X)
        Z2 = self.Rom_inv.solve_hermitian_transpose_mat(Z1)
        Y = self.Btil.apply_hermitian_transpose_mat(Z2) if Y == None \
            else self.Btil.apply_hermitian_transpose_mat(Z2, Y)
        Z1.destroy()
        Z2.destroy()
        return Y
    
    def destroy(self):
        pass
