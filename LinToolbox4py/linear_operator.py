from petsc4py import PETSc
from mpi4py import MPI
import scipy as sp
import numpy as np

from .petsc4py_helper_functions import compute_dense_inverse
from .petsc4py_helper_functions import mat_solve_hermitian_transpose

class LinearOperator:
    r"""
        This class creates a linear operator of the form

        .. math::
            L = \mathbb{P}M\mathbb{P},\quad M = A + 
            \textcolor{orange}{B K C^*}

        where :math:`A \in \mathbb{C}^{N\times N}`,
        :math:`B \in \mathbb{C}^{N\times m}`, 
        :math:`K \in \mathbb{C}^{m\times q}`,
        :math:`C\in \mathbb{C}^{N\times q}`
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

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or 
            :code:`MPI.COMM_SELF`)
            
        :param A: a sparse matrix of size :math:`N\times N`
        :type A: PETSc.Mat.Type.MPIAIJ

        :param P: a tuple of 2 rank-:math:`r` distributed matrices of size 
            :math:`N{\times}r` and an integer :math:`s{\in\{0,1\}}` 
            (e.g., P = :math:`\left(\Phi,\Psi,0\right)`).
            If :code:`None`, :math:`L = M`.
        :type P: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,int

        :param F: a tuple of distributed matrices (e.g., 
            :math:`\left(\textcolor{orange}{B,K,C}\right)`). 
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
            Set attributes :code:`Phi` and :code:`Psi` and enforce 
            biorthogonality (i.e., 
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
            Set attributes :code:`B`, :code:`K` and :code:`C` and attributes 
            :code:`X`, :code:`D` and :code:`Y`.

            :param F: a tuple of distributed matrices (e.g., :math:`(B,K,C)`)
            :type F: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,
                PETSc.Mat.Type.DENSE
            
            :param Finv: a tuple of distributed matrices (e.g., :math:`(X,D,Y)`)
            :type Finv: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,
                PETSc.Mat.Type.DENSE
        """
        self.destroy_low_rank_factors()
        self.low_rank_factors_exist = True
        self.B, self.K, self.C = F

        if self.ksp_exists:
            self.low_rank_factors_inverse_exist = True
            if Finv == None:
                self.X = self.B.duplicate()
                self.ksp.matSolve(self.B,self.X)
                self.Y = mat_solve_hermitian_transpose(self.ksp,self.C)

                self.C.hermitianTranspose()
                M = self.C.matMult(self.X).matMult(self.K)
                Id = PETSc.Mat().createConstantDiagonal(M.getSizes(),1.0,\
                                                        comm=self.comm)
                M.axpy(1.0,Id,structure=PETSc.Mat.Structure.SAME)
                self.D = self.K.matMult(compute_dense_inverse(self.comm,M))
                self.C.hermitianTranspose()
            else:
                self.X, self.D, self.Y = Finv


    def destroy_low_rank_factors(self):
        r"""
            Destroy attributes :code:`B`, :code:`K`, :code:`C`,
            :code:`X`, :code:`Y` and :code:`D`.
        """

        if self.low_rank_factors_exist:
            self.B.destroy()
            self.C.destroy()
            self.K.destroy()

        self.B, self.C, self.K = None, None, None
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
                B K C^* x & \text{if }  \text{mode} = \text{direct}\\
                C K^* B^*x  & \text{if } \text{mode} = \text{adjoint}.
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
                z, y = self.C.createVecs()
                Kz = self.K.createVecLeft()
                self.C.multHermitian(x,z)
                self.K.mult(z,Kz)
                self.B.mult(Kz,y)
            elif mode == 'adjoint':
                z, y = self.B.createVecs()
                Kz = self.K.createVecRight()
                self.B.multHermitian(x,z)
                self.K.multHermitian(z,Kz)
                self.C.mult(Kz,y)
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
    

    def destroy(self):
        r"""
            Destroy an instance of the class
        """
        self.destroy_ksp()
        self.destroy_low_rank_factors()
        self.destroy_projection()
        self.A.destroy()



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class LowRankLinearOperator:
    r"""
        This class creates a linear operator of the form

        .. math::
            L = \mathbb{P}M\mathbb{P},\quad M = 
            \underbrace{\textcolor{red}{U \Sigma V^*}}_{A} + 
            \textcolor{orange}{B K C^*}

        where :math:`U \in \mathbb{C}^{N\times r}`, 
        :math:`\Sigma \in \mathbb{C}^{r\times p}`,
        :math:`V\in \mathbb{C}^{N\times p}`,
        :math:`B \in \mathbb{C}^{N\times m}`, 
        :math:`K \in \mathbb{C}^{m\times q}`,
        :math:`C\in \mathbb{C}^{N\times q}`
        and, given the integer :math:`s\in\{0,1\}`, 
        :math:`\mathbb{P}` is a projection operator defined as follows
        
        .. math::
            \mathbb{P} = \begin{cases} 
            I & \text{default} \\
            \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^* & \text{if } s = 0 \\
            I - \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^*  & \text{if } s = 1
            \end{cases}

        This class is similar to the 
        :meth:`LinToolbox4py.linear_operator.LinearOperator` class, except
        that :code:`A` is a low-rank matrix (hence the name of the class) and
        the :code:`solve()` method is not provided.

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or 
            :code:`MPI.COMM_SELF`)

        :param A: a tuple of distributed matrices (e.g., 
            :math:`\left(\textcolor{red}{U,\Sigma,V}\right)`). 
        :type A: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE

        :param P: a tuple of 2 rank-:math:`r` distributed matrices of size 
            :math:`N{\times}r` and an integer :math:`s{\in\{0,1\}}` 
            (e.g., P = :math:`\left(\Phi,\Psi,0\right)`).
            If :code:`None`, :math:`L = M`.
        :type P: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,int

        :param F: a tuple of distributed matrices (e.g., 
            :math:`\left(\textcolor{orange}{B,K,C}\right)`). 
            If :code:`F == None`, :math:`M = A`.
        :type F: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE
    """
    def __init__(
        self,
        comm,
        A,
        P=None,
        F=None
    ):

        # ------ Initialize flags ------------------
        self.projection_exists = False
        self.low_rank_factors_exist = False
        # -------------------------------------------
        
        # ------ Populate attributes ----------------
        self.comm = comm
        self.set_low_rank_factors(A,['U','Sig','V'])
        self.set_projection(P) if P != None else self.destroy_projection()
        self.set_low_rank_factors(F,['B','K','C']) if F != None else None
        # -------------------------------------------


    def set_projection(
        self,
        P
    ):
        r"""
            Set attributes :code:`Phi` and :code:`Psi` and enforce 
            biorthogonality (i.e., 
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

    def set_low_rank_factors(
        self,
        F,
        attr_names
    ):
        r"""
            Set attributes corresponding to the low-rank factors in 
            :code:`F` and :code:`attr_names`.

            :param F: a tuple of distributed matrices (e.g., :math:`(B,K,C)`)
            :type F: PETSc.Mat.Type.DENSE,PETSc.Mat.Type.DENSE,
                PETSc.Mat.Type.DENSE
            
            :param attr_names: a tuple of attribute names. Must be permutations
                of :code:`('B','K','C')` or :code:`('U','Sig','V')`.
            :type attr_names: str,str,str
        """
        attr_names_sorted = np.sort(attr_names).tolist()
        check1 = attr_names_sorted == np.sort(['B','K','C']).tolist()
        check2 = attr_names_sorted == np.sort(['U','Sig','V']).tolist()
        if check1 == False and check2 == False:
            raise Exception ("The argument attr_names must be ['B','K','C']\
                             or ['U','Sig','V']")
        else:
            for (j,f) in enumerate(F):
                setattr(self,attr_names[j],f)


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
        F,
        x,
        mode='direct'
    ):
        r"""
            Compute

            .. math::
                y = \begin{cases}
                F_0 F_1 F_2^* x & \text{if }  \text{mode} = \text{direct}\\
                F_2 F_1^* F_0^*x  & \text{if } \text{mode} = \text{adjoint}.
                \end{cases}

            where :math:`F_j` is the :math:`j`th component of the tuple 
            :code:`F`.
            
            :param x: a vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str
            
            :return: a vector of size :math:`N`
            :rtype: PETSc.Vec
        """
        
        if mode == 'direct':
            z, y = F[2].createVecs()
            Kz = F[1].createVecLeft()
            self.F[2].multHermitian(x,z)
            self.F[1].mult(z,Kz)
            self.F[0].mult(Kz,y)
        elif mode == 'adjoint':
            z, y = self.F[0].createVecs()
            Kz = self.F[1].createVecRight()
            self.F[0].multHermitian(x,z)
            self.F[1].multHermitian(z,Kz)
            self.F[2].mult(Kz,y)
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
        z = self.apply_low_rank_factors((self.U,self.Sigma,self.V),Px,mode)
        z.axpy(1.0,self.apply_low_rank_factors(\
            (self.U,self.Sigma,self.V),Px,mode)) if self.U != None \
            else None
        y = self.apply_projection(z,mode)
        z.destroy()
        
        return y


    def destroy(self):
        r"""
            Destroy an instance of the class
        """
        objects = [self.B,self.K,self.C,self.U,self.Sig,\
                   self.V,self.Phi,self.Psi]
        for obj in objects:
            if obj != None:
                obj.destroy()
