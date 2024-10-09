from petsc4py import PETSc
from mpi4py import MPI
import scipy as sp

from .petsc4py_helper_functions import compute_dense_inverse
from .petsc4py_helper_functions import mat_solve_hermitian_transpose

class LinearOperator:
    r"""
        This class creates a linear operator of the form

        .. math::
            L = \mathbb{P}M\mathbb{P},\quad M = \left(A + 
            \textcolor{orange}{U K V^*}\right)

        where :math:`U \in \mathbb{C}^{N\times m}`, 
        :math:`V\in \mathbb{C}^{N\times q}`, 
        :math:`K \in \mathbb{C}^{m\times q}`
        and, given the integer :math:`s\in\{0,1\}`, 
        :math:`\mathbb{P}` is a projection operator defined as follows
        
        .. math::
            \mathbb{P} = \begin{cases} 
            I & \text{default} \\
            \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^* & \text{if } s = 0 \\
            I - \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^*  & \text{if } s = 1
            \end{cases}

        Perhaps the most useful methods here are :code:`apply(x)`, 
        which returns :math:`y = Lx`, and :code:`solve(x)`, 
        which returns :math:`y = \mathbb{P}M^{-1}\mathbb{P} x`,
        where :math:`M^{-1}` is computed using the Woodbury matrix identity

        .. math::

            M^{-1} = A^{-1} - X D Y^*\,\,\text{with}\,\, \textcolor{blue}{X}
            = A^{-1}U,\, \textcolor{blue}{D} = 
            K\left(I + V^* A^{-1}U K\right)^{-1},\, 
            \textcolor{blue}{Y} = A^{-*}V.

        
        :param A: a sparse matrix of size :math:`N\times N`
        :type A: PETSc.Mat.Type.MPIAIJ

        :param P: a tuple of 2 rank-:math:`r` distributed matrices of size 
            :math:`N{\times}r` and an integer :math:`s{\in\{0,1\}}` 
            (e.g., P = :math:`\left(\Phi,\Psi,0\right)`).
            If :code:`None`, :math:`L = M`.
        :type P: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,int

        :param F: a tuple of distributed matrices (e.g., 
            :math:`\left(\textcolor{orange}{U,K,V}\right)`). 
            If :code:`F == None`, :math:`M = A`.
        :type F: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE
        
        :param Finv: a tuple of distributed matrices 
            (e.g., :math:`\left(\textcolor{blue}{X,D,Y}\right)`). 
            If :code:`Finv == None`, :code:`F != None` and 
            :code:`create_solver == True`, the factors :math:`X,\,D`, 
            and :math:`Y` are computed exactly as defined in the equation above.
            However, there are instances when :math:`X,\,D`, 
            and :math:`Y` are known a-priori, and they can be passed directly
            as an argument.
        :type Finv: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,
            PETSc.Mat.Type.DENSE

        :param create_solver: :code:`True` enables the use of the 
            :code:`solve()` method
        :type create_solver: bool

        :param solver_type: one of PETSc available `KSPs`_.
        :type solver_type: str
    """
    def __init__(
        self,
        comm,
        A,
        P=None,
        F=None,
        Finv=None,
        create_solver=True,
        solver_type='preonly'
    ):

        # ------ Initialize flags ------------------
        self.projection_exists = False
        self.low_rank_factors_exist = False
        self.low_rank_factors_inverse_exist = False
        self.ksp_exists = False
        # -------------------------------------------
        
        # ------ Populate attributes ----------------
        self.comm = comm
        self.A = A
        self.set_ksp(solver_type) if create_solver else self.destroy_ksp()
        self.set_projection(P) if P != None else self.destroy_projection()
        self.set_low_rank_factors(F,Finv) if F != None else \
            self.destroy_low_rank_factors()
        # -------------------------------------------


    def set_projection(
        self,
        P
    ):
        r"""
            Set attributes :code:`Phi` and :code:`Psi` from the tuple 
            :code:`P`, and enforce biorthogonality (i.e., 
            :math:`\Phi\leftarrow \Phi\left(\Psi^*\Phi\right)^{-1}`).

            :param P: a tuple of 2 rank-:math:`r` distributed matrices of size 
                :math:`N{\times}r` and an integer :math:`s{\in\{0,1\}}` 
                (e.g., P = :math:`\left(\Phi,\Psi,0\right)`)
            :type P: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,int
        """
        self.destroy_projection()
        self.projection_exists = True
        self.Phi, self.Psi, self.s_proj = P
        self.Psi.conjugate()
        self.Phi = self.Phi.matMult(compute_dense_inverse(\
            self.comm,self.Psi.transposeMatMult(self.Phi)))
        self.Psi.conjugate()
            

    def destroy_projection(self):
        r"""
            Destroy attributes :code:`Phi` and :code:`Psi`.
        """
        if self.projection_exists:
            self.Phi.destroy()
            self.Psi.destroy()
        
        self.Phi, self.Psi, self.s_proj = None, None, None
        self.projection_exists = False

    def set_low_rank_factors(
        self,
        F,
        Finv=None
    ):
        r"""
            Set attributes :code:`U`, :code:`K` and :code:`V` and attributes 
            :code:`X`, :code:`D` and :code:`Y`.

            :param F: a tuple of distributed matrices (e.g., :math:`(U,K,V)`)
            :type F: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,
                PETSc.Mat.Type.DENSE
            
            :param Finv: a tuple of distributed matrices (e.g., :math:`(X,D,Y)`)
            :type Finv: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,
                PETSc.Mat.Type.DENSE
        """
        self.destroy_low_rank_factors()
        self.low_rank_factors_exist = True
        self.U, self.K, self.V = F

        if self.ksp_exists:
            self.low_rank_factors_inverse_exist = True
            if Finv == None:
                self.X = self.U.duplicate()
                self.ksp.matSolve(self.U,self.X)
                self.Y = mat_solve_hermitian_transpose(self.ksp,self.V)

                self.V.hermitianTranspose()
                M = self.V.matMult(self.X).matMult(self.K)
                Id = PETSc.Mat().createConstantDiagonal(M.getSizes(),1.0,\
                                                        comm=self.comm)
                M.axpy(1.0,Id,structure=PETSc.Mat.Structure.SAME)
                self.D = self.K.matMult(compute_dense_inverse(self.comm,M))
                self.V.hermitianTranspose()
            else:
                self.X, self.D, self.Y = Finv


    def destroy_low_rank_factors(self):
        r"""
            Destroy attributes :code:`U`, :code:`K`, :code:`V`,
            :code:`X`, :code:`Y` and :code:`D`.
        """

        if self.low_rank_factors_exist:
            self.U.destroy()
            self.V.destroy()
            self.K.destroy()

        self.U, self.V, self.K = None, None, None
        self.low_rank_factors_exist = False

        if self.low_rank_factors_inverse_exist:
            self.X.destroy()
            self.D.destroy()
            self.Y.destroy()

        self.X, self.D, self.Y = None, None, None
        self.low_rank_factors_inverse_exist = False


    def set_ksp(
        self,
        solver_type
    ):
        r"""
            Create the attribute :code:`ksp` that holds the necessary PETSc 
            functionality to solve linear systems.
            
            :param solver_type: one of PETSc available `KSPs`_.
            :type solver_type: str
        """
        
        self.destroy_ksp()
        self.ksp_exists = True
        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType(solver_type)

        if solver_type == 'preonly':
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')
            pc.setUp()

        self.ksp.setUp()

    def destroy_ksp(self):
        r"""
            Destroy the :code:`ksp` attribute.
        """

        self.ksp.destroy() if self.ksp_exists else None
        self.ksp = None
        self.ksp_exists = False

    
    def apply_projection(
        self,
        x,
        mode='direct'
    ):
        r"""
            Compute 

            .. math::
                y = \begin{cases}
                \mathbb{P}x & \text{if }  \text{mode} = \text{direct}\\
                \mathbb{P}^*x  & \text{if } \text{mode} = \text{adjoint}.
                \end{cases}

            
            :param x: a vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a vector of size :math:`N`
            :rtype: PETSc.Vec
        """
        
        if self.projection_exists == False:
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
            Compute 

            .. math::
                y = \begin{cases}
                U K V^* x & \text{if }  \text{mode} = \text{direct}\\
                V K^* U^*x  & \text{if } \text{mode} = \text{adjoint}.
                \end{cases}
            
            :param x: a vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a vector of size :math:`N`
            :rtype: PETSc.Vec
        """
        
        if self.low_rank_factors_exist == False:
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
            Compute 

            .. math::
                y = \begin{cases} 
                Lx & \text{if }  \text{mode} = \text{direct}\\
                L^*x  & \text{if } \text{mode} = \text{adjoint}.
                \end{cases}

            :param x: a vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a vector of size :math:`N`
            :rtype: PETSc.Vec
        """

        z = x.duplicate()
        Px = self.apply_projection(x,mode)
        self.A.mult(Px,z) if mode == 'direct' else self.A.multHermitian(Px,z)
        z.axpy(1.0,self.apply_low_rank_factors(Px,mode))
        y = self.apply_projection(z,mode)
        z.destroy()
        
        return y

    def apply_low_rank_factors_inverse(
        self,
        x,
        mode='direct'
    ):
        r"""
            Compute 

            .. math::
                y = \begin{cases}
                X D Y^* x & \text{if }  \text{mode} = \text{direct}\\
                Y D^* X^*x  & \text{if } \text{mode} = \text{adjoint}.
                \end{cases}
            
            :param x: a vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a vector of size :math:`N`
            :rtype: PETSc.Vec
        """
        if self.low_rank_factors_exist == True:
            y = x.duplicate()
            if mode == 'direct':
                z, Dz = self.D.createVecs()
                self.Y.multHermitian(x,z)
                self.D.mult(z,Dz)
                self.X.mult(Dz,y)
            elif mode == 'adjoint':
                Dz, z = self.D.createVecs()
                self.X.multHermitian(x,z)
                self.D.multHermitian(z,Dz)
                self.Y.mult(Dz,y)
        else:
            y = x.copy()
            y.set(0.0)

        return y

    def solve(
        self,
        x,
        mode='direct'
    ):
        r"""
            Compute 

            .. math::
                y = \begin{cases}
                \mathbb{P} M^{-1}\mathbb{P} x & \text{if }  
                \text{mode} = \text{`direct'}\\
                \mathbb{P}^* M^{-*}\mathbb{P}^*x  & \text{if } 
                \text{mode} = \text{`adjoint'}.
                \end{cases}

            :param x: a vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a vector of size :math:`N`
            :rtype: PETSc.Vec
        """

        Px = self.apply_projection(x,mode)
        z = Px.duplicate()
        if mode == 'direct':
            self.ksp.solve(Px,z)
        elif mode == 'adjoint':
            Px.conjugate()
            self.ksp.solveTranspose(Px,z)
            Px.conjugate()
            z.conjugate()
        z.axpy(-1.0,self.apply_low_rank_factors_inverse(Px,mode))
        y = self.apply_projection(z,mode)

        return y