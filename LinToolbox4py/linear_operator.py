from petsc4py import PETSc
from mpi4py import MPI
import scipy as sp

from .petsc4py_helper_functions import compute_dense_inverse

class LinearOperator:
    r"""
        This class creates a linear operator of the form

        .. math::
            L = \mathbb{P}\left(A + U K V^*\right)\mathbb{P}

        where :math:`U \in \mathbb{C}^{N\times m}`, :math:`V\in \mathbb{C}^{N\times q}`, :math:`K \in \mathbb{C}^{m\times q}`
        and, given the integer :math:`s\in\{0,1\}`, :math:`\mathbb{P}` is a projection operator defined as follows
        
        .. math::
            \mathbb{P} = \begin{cases} 
            I & \text{default} \\
            \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^* & \text{if } s = 0 \\
            I - \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^*  & \text{if } s = 1
            \end{cases}

        :param A: a distributed sparse matrix of size :math:`N\times N`
        :type A: PETSc.Mat.Type.MPIAIJ
        :param W: [optional] a 3-tuple of distributed matrices (e.g., W = :math:`\left(U,K,V\right)`)
        :type W: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE
        :param P: [optional] a 3-tuple of 2 rank-:math:`r` distributed matrices of size :math:`N{\times}r` and an integer :math:`s{\in\{0,1\}}` (e.g., P = :math:`\left(\Phi,\Psi,0\right)`)
        :type P: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,int
    """
    def __init__(
        self,
        comm,
        A,
        W=None,
        P=None
    ):

        self.comm = comm
        self.A = A

        if W != None:
            self.low_rank_factors = True
            self.U, self.K, self.V = W
        else:
            self.low_rank_factors = False

        if P != None:
            self.projection = True
            self.Phi, self.Psi, self.s_proj = P

            # Enforce biorthogonality
            self.Psi.conjugate()
            self.Phi = self.Phi.matMult(compute_dense_inverse(\
                self.comm,self.Psi.transposeMatMult(self.Phi)))
            self.Psi.conjugate()

        else:
            self.projection = False

    def apply_projection(
        self,
        x,
        mode='direct'
    ):
        r"""
            Compute :math:`y = \mathbb{P}x` if mode = 'direct' or :math:`y = \mathbb{P}^*x` if mode = 'adjoint'
            
            :param x: a distributed vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a distributed vector of size :math:`N`
            :rtype: PETSc.Vec
        """
        
        if self.projection == False:
            y = x.copy()
        else:
            z, y = self.Phi.createVecs()
            if mode == 'direct':
                self.Psi.multHermitian(x,z)
                self.Phi.mult(z,y)
            elif mode == 'adjoint':
                self.Phi.multHermitian(x,z)
                self.Psi.mult(z,y)
            if self.s_proj == 1:
                y.aypx(-1.0,x)
            z.destroy()
        
        return y
    
    def apply_low_rank_factors(
        self,
        x,
        mode='direct'
    ):
        r"""
            Compute :math:`y = U K V^*x` if mode = 'direct' or :math:`y = V K^* U^*x` if mode = 'adjoint'
            
            :param x: a distributed vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a distributed vector of size :math:`N`
            :rtype: PETSc.Vec
        """
        
        if self.low_rank_factors == False:
            y = x.copy()
            y.scale(0.0)
        else:
            if mode == 'direct':
                z, y = self.V.createVecs()
                Kz = self.K.createVecLeft()
                self.V.multHermitian(x,z)
                self.K.mult(z,Kz)
                self.U.mult(Kz,y)
            elif mode == 'adjoint':
                z, y = self.U.createVecs()
                Kz = self.K.createVecRight()
                self.U.multHermitian(x,z)
                self.K.multHermitian(z,Kz)
                self.V.mult(Kz,y)
            z.destroy()
            Kz.destroy()

        return y

    def apply(
        self,
        x,
        mode='direct'
    ):
        r"""
            Compute :math:`y = Lx` if mode = 'direct' or :math:`y = L^*x` if mode = 'adjoint', where

            .. math::
                L = \mathbb{P}\left(A - U V^*\right)\mathbb{P}

            :param x: a distributed vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a distributed vector of size :math:`N`
            :rtype: PETSc.Vec
        """

        z = x.duplicate()
        Px = self.apply_projection(x,mode)
        if mode == 'direct':
            self.A.mult(Px,z)
        elif mode == 'adjoint':
            self.A.multHermitian(Px,z)
        z.axpy(1.0,self.apply_low_rank_factors(Px,mode))
        y = self.apply_projection(z,mode)
        z.destroy()
        
        return y