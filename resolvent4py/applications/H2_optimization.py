from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np
import scipy as sp
import pymanopt

from ..io_functions import read_dense_matrix
from ..io_functions import read_coo_matrix
from ..io_functions import read_vector
from ..petsc4py_helper_functions import compute_local_size
from ..petsc4py_helper_functions import compute_dense_inverse
from ..linear_operators import MatrixLinearOperator
from ..linear_operators import LowRankLinearOperator
from ..linear_operators import LowRankUpdatedLinearOperator
from ..solvers_and_preconditioners_functions import create_mumps_solver
from ..applications import eigendecomposition

class optimization_objects:

    def __init__(self, comm, fnames_jacobian, jacobian_sizes, path_factors, \
                factor_sizes, fname_frequencies, fname_weights, \
                stability_params, compute_B, compute_C):
        
        self.comm = comm
        self.jacobian_sizes = jacobian_sizes
        A = read_coo_matrix(self.comm, fnames_jacobian, jacobian_sizes)
        A.scale(-1.0)
        ksp = create_mumps_solver(self.comm, A)
        self.A = MatrixLinearOperator(self.comm, A, ksp)
        self.freqs = np.load(fname_frequencies)
        self.weights = np.load(fname_weights)
        self.factor_sizes = factor_sizes
        self.load_factors(path_factors, factor_sizes)
        self.compute_B = compute_B
        self.compute_C = compute_C
        self.stab_reg = stability_params
        
    def load_factors(self, path_factors, factor_sizes):
        
        self.U, self.S, self.V, self.low_rank_ops = [], [], [], []
        for i in range (len(self.freqs)):
            fname_U = path_factors + 'omega_%1.5f/U.dat'%self.freqs[i]
            fname_S = path_factors + 'omega_%1.5f/S.dat'%self.freqs[i]
            fname_V = path_factors + 'omega_%1.5f/V.dat'%self.freqs[i]

            self.U.append(read_dense_matrix(self.comm, fname_U, factor_sizes))
            self.V.append(read_dense_matrix(self.comm, fname_V, factor_sizes))
            self.S.append(read_dense_matrix(self.comm, fname_S, \
                                        (factor_sizes[-1],factor_sizes[-1])))
            self.low_rank_ops.append(\
                LowRankLinearOperator(self.comm, self.U[-1], \
                                      self.S[-1], self.V[-1]))

    def evaluate_exponential(self,val):
        alpha, beta = self.stab_reg[:2]
        try:    
            value = np.exp(alpha*(val.real + beta)).real
        except: 
            value = 1e12
        return value
    
    def evaluate_gradient_exponential(self,val):
        alpha = self.stab_reg[0]
        return alpha*self.evaluate_exponential(val)


def create_objective_and_gradient(manifold,opt_obj):
    r"""
        :param manifold: one of the pymanopt manifolds
        :param opt_obj: instance of the optimization objects class
    """
    
    euclidean_hessian = None
    euclidean_gradient = None

    @pymanopt.function.numpy(manifold)
    def cost(*params):
        r"""
            Evaluate the cost function
            params = [p, K, s]
        """
        rank = opt_obj.comm.Get_rank()
        size = opt_obj.comm.Get_size()

        p, K, s = params
        B = opt_obj.compute_B(p)
        C = opt_obj.compute_C(s)

        na = B.getSizes()[-1][-1]  
        ns = C.getSizes()[-1][-1]
        naloc = compute_local_size(na)
        nsloc = compute_local_size(ns)
        nalocs = np.asarray(opt_obj.comm.allgather(naloc))
        disps = np.concatenate(([0], np.cumsum(nalocs[:-1])))
        r0 = disps[rank]
        r1 = disps[rank + 1] if rank < size - 1 else K.shape[0]
        Kd = PETSc.Mat().createDense(((naloc, na), (nsloc, ns)), \
                                     array=K[r0:r1,], comm=opt_obj.comm)

        # H2 component of the cost function
        J = 0
        for i in range (len(opt_obj.freqs)):

            RB = opt_obj.low_rank_ops[i].apply_mat(B)
            RHTC = opt_obj.low_rank_ops[i].apply_hermitian_transpose_mat(C)
            C.hermitianTranspose()
            CHTRB = C.matMult(RB)
            C.hermitianTranspose()
            
            M = CHTRB.matMult(Kd)
            Id = PETSc.Mat().createConstantDiagonal(M.getSizes(),1.0,\
                                                    comm=opt_obj.comm)
            M.axpy(1.0,Id,structure=PETSc.Mat.Structure.SAME)
            KLinv = Kd.matMult(compute_dense_inverse(opt_obj.comm,M))
            
            # Tr(RR^*)
            value = np.sum(opt_obj.S[i].getDiagonal().getArray().real**2)
            J += opt_obj.weights[i]*opt_obj.comm.allreduce(value, op=MPI.SUM)

            # -Tr(MR^*) - Tr(RM^*)
            RHTC.hermitianTranspose()
            F1 = RHTC.matMult(opt_obj.V[i].matMult(opt_obj.S[i]))
            RHTC.hermitianTranspose()
            opt_obj.U[i].hermitianTranspose()
            F2 = opt_obj.U[i].matMult(RB.matMult(KLinv))
            opt_obj.U[i].hermitianTranspose()
            J += -2*opt_obj.weights[i]*F1.matMult(F2).getDiagonal().sum().real
            
            # Tr(MM^*)
            M = RB.matMult(KLinv)
            RB.hermitianTranspose()
            F3 = RB.matMult(M)
            RB.hermitianTranspose()
            M.destroy()
            KLinv.hermitianTranspose()
            M = RHTC.matMult(KLinv)
            KLinv.hermitianTranspose()
            RHTC.hermitianTranspose()
            F4 = RHTC.matMult(M)
            RHTC.hermitianTranspose()
            M.destroy()
            J += opt_obj.weights[i]*F3.matMult(F4).getDiagonal().sum().real
        
        # Stability-promoting component of the cost function
        lin_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.A, B, Kd, C)
        D_, _ = eigendecomposition(lin_op, lin_op.solve, 300, 20)
        D = -1./D_.getDiagonal().getArray()
        Jd = None
        if rank == 0:
            Jd = np.sum([opt_obj.evaluate_exponential(D[k]) \
                         for k in range (len(D))])
        J += opt_obj.comm.bcast(Jd, root=0)
        return J
    
    
    # @pymanopt.function.numpy(manifold)
    # def euclidean_gradient(*params):
    #     """
    #         Evaluate the euclidean gradient of the cost function with respect to the parameters
    #     """
        
    #     p, K, s = params
    #     na, ns, N = len(p), len(s), opt_obj.RB[0].shape[0]
        
    #     RB = np.zeros((N,na),dtype=np.complex128)
    #     CR = np.zeros((ns,N),dtype=np.complex128)
    #     CRB_ = np.zeros((opt_obj.fom.C.shape[0],na),dtype=np.complex128)
    #     CRB = np.zeros((ns,na),dtype=np.complex128)
        
    #     B = opt_obj.interpolate(opt_obj.fom.actu_locs,opt_obj.fom.B,p)
    #     C = opt_obj.interpolate(opt_obj.fom.actu_locs,opt_obj.fom.C.T,s).T
    #     dBdp = opt_obj.compute_interp_derivative(opt_obj.fom.actu_locs,opt_obj.fom.B,p,1e-6)
    #     dBdp = lift_into_third_order_tensor(dBdp,p)
    #     dCds = opt_obj.compute_interp_derivative(opt_obj.fom.sens_locs,opt_obj.fom.C.T,s,1e-6).T
    #     dCds = lift_into_third_order_tensor(dCds,s)
        
    #     grad_p = np.zeros(len(p),dtype=np.complex128)
    #     grad_s = np.zeros(len(s),dtype=np.complex128)
    #     grad_K = np.zeros(K.shape,dtype=np.complex128)
        
    #     # H2 component of the gradient
    #     Idna, Idns = np.eye(K.shape[0]), np.eye(K.shape[-1])
        
    #     for i in range (opt_obj.n_freqs):
            
    #         RB[:,:] = opt_obj.interpolate(opt_obj.fom.actu_locs,opt_obj.RB[i],p)
    #         CRB_[:,:] = opt_obj.interpolate(opt_obj.fom.actu_locs,opt_obj.CRB[i],p)
    #         CR[:,:] = opt_obj.interpolate(opt_obj.fom.sens_locs,opt_obj.CR[i].T,s).T
    #         CRB[:,:] = opt_obj.interpolate(opt_obj.fom.sens_locs,CRB_.T,s).T
            
            
    #         Linv = sp.linalg.inv(Idns + CRB@K)
    #         KLinv = K@Linv
            
    #         wi = opt_obj.weights[i]
    #         Linv = sp.linalg.inv(Idns + CRB@K)
    #         KLinv = K@Linv
            
    #         # Gradient with respect to actuator parameters
    #         G = -opt_obj.V[i]@(opt_obj.S[i]**2)@(opt_obj.V[i].conj().T@CR.conj().T@KLinv.conj().T)
    #         G += opt_obj.V[i]@opt_obj.S[i]@(opt_obj.U[i].conj().T@RB)@KLinv@CR@CR.conj().T@KLinv.conj().T
    #         grad_p += 2*wi*np.einsum('ijk,ij',dBdp.conj(),G - CR.conj().T@KLinv.conj().T@(B.conj().T@G)).reshape(-1)
            
    #         # Gradient with respect to K
    #         G = -RB.conj().T@opt_obj.U[i]@opt_obj.S[i]@(opt_obj.V[i].conj().T@CR.conj().T@Linv.conj().T)
    #         G += RB.conj().T@RB@KLinv@CR@CR.conj().T@Linv.conj().T
    #         Mat = Idna - CRB.conj().T@KLinv.conj().T
    #         grad_K += 2*wi*Mat@G
            
    #         # Gradient with respect to sensor parameters
    #         G = -(KLinv.conj().T@RB.conj().T@opt_obj.U[i])@(opt_obj.S[i]**2)@opt_obj.U[i].conj().T
    #         G += (KLinv.conj().T@RB.conj().T@RB@KLinv@CR@opt_obj.V[i]@opt_obj.S[i])@opt_obj.U[i].conj().T
    #         grad_s += 2*wi*np.einsum('ij,ijk',G - (G@C.conj().T)@KLinv.conj().T@RB.conj().T,dCds.conj()).reshape(-1)
            
        
    #     # Stability-promoting penalty
    #     V, D, W = opt_obj.fom.compute_spectrum(B,K,C,opt_obj.stab_reg[-1])
    #     M = np.diag([opt_obj.evaluate_gradient_exponential(D[k]) for k in range (len(D))])
    #     grad_p += -np.einsum('ijk,ij',dBdp.conj(),W@M@(V.conj().T@C.conj().T@K.conj().T))
    #     grad_K += -B.conj().T@W@M@(V.conj().T@C.conj().T)
    #     grad_s += -np.einsum('ij,ijk',(K.conj().T@B.conj().T@W)@M@V.conj().T,dCds.conj())
        
        
    #     return grad_p.real, grad_K.real, grad_s.real
    
    
    return cost, euclidean_gradient, euclidean_hessian
