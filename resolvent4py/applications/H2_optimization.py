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


def compute_double_contraction(comm, Mat1, Mat2):
    Mat1_array = Mat1.getDenseArray()
    Mat2_array = Mat2.getDenseArray()
    value = comm.allreduce(np.sum(Mat1_array*Mat2_array), op=MPI.SUM)
    return value

def compute_trace(comm, L1, L2, L1_hermitian_transpose=False, \
                  L2_hermitian_transpose=False):
    r"""
        Compute the trace of the product of two low-rank operators :math:`L_1` 
        and :math:`L_2` (see 
        :meth:`resolvent4py.linear_operators.LowRankLinearOperator`).

        :param comm: MPI communicator
        :param L1: low-rank linear operator
        :param L2: low-rank linear operator
        :param L1_hermitian_transpose: [optional] :code:`True` or :code:`False`
        :type L1_hermitian_transpose: bool
        :param L2_hermitian_transpose: [optional] :code:`True` or :code:`False`
        :type L2_hermitian_transpose: bool

        :rtype: PETSc scalar
    """
    L1_action = L1.apply_mat if L1_hermitian_transpose == False else \
        L1.apply_hermitian_transpose_mat
    if L2_hermitian_transpose == False:
        L2.V.hermitianTranspose()
        F = L2.V.matMult(L1_action(L2.U.matMult(L2.Sigma)))
        L2.V.hermitianTranspose()
    else:
        L2.U.hermitianTranspose()
        L2.Sigma.hermitianTranspose()
        F = L2.U.matMult(L1_action(L2.V.matMult(L2.Sigma)))
        L2.Sigma.hermitianTranspose()
        L2.U.hermitianTranspose()
    trace = comm.allreduce(np.sum(F.getDiagonal().getArray()), op=MPI.SUM)
    return trace

def assemble_woodbury_low_rank_operator(comm, R, B, C, Kd):
    r"""
        Assemble low-rank operator representation of 

        .. math::
            
            M = RBK\left(I + C^*RB K\right)^{-1}C^*R
    """
    RB = R.apply_mat(B)
    RHTC = R.apply_hermitian_transpose_mat(C)
    C.hermitianTranspose()
    CHTRB = C.matMult(RB)
    C.hermitianTranspose()
    M = CHTRB.matMult(Kd)
    Id = PETSc.Mat().createConstantDiagonal(M.getSizes(), 1.0, comm=comm)
    Id.convert(PETSc.Mat.Type.DENSE)
    M.axpy(1.0,Id,structure=PETSc.Mat.Structure.SAME)
    Linv = compute_dense_inverse(comm,M)
    KLinv = Kd.matMult(Linv)
    M = LowRankLinearOperator(comm, RB, KLinv, RHTC)
    return M, Linv, KLinv

class optimization_objects:

    def __init__(self, comm, fnames_jacobian, jacobian_sizes, path_factors, \
                factor_sizes, fname_frequencies, fname_weights, \
                stability_params, compute_B, compute_C, compute_dB, compute_dC):
        
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
        self.compute_dB = compute_dB
        self.compute_dC = compute_dC
        self.stab_reg = stability_params
        
    def load_factors(self, path_factors, factor_sizes):
        
        self.U, self.S, self.V, self.R_ops = [], [], [], []
        for i in range (len(self.freqs)):
            fname_U = path_factors + 'omega_%1.5f/U.dat'%self.freqs[i]
            fname_S = path_factors + 'omega_%1.5f/S.dat'%self.freqs[i]
            fname_V = path_factors + 'omega_%1.5f/V.dat'%self.freqs[i]

            self.U.append(read_dense_matrix(self.comm, fname_U, factor_sizes))
            self.V.append(read_dense_matrix(self.comm, fname_V, factor_sizes))
            self.S.append(read_dense_matrix(self.comm, fname_S, \
                                        (factor_sizes[-1],factor_sizes[-1])))
            self.R_ops.append(\
                LowRankLinearOperator(self.comm, self.U[-1], \
                                      self.S[-1], self.V[-1]))

    def evaluate_exponential(self,val):
        alpha, beta = self.stab_reg[:2]
        try:    value = np.exp(alpha*(val.real + beta)).real
        except: value = 1e12
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
            Ji = 0
            M, _, _ = assemble_woodbury_low_rank_operator(opt_obj.comm, \
                                                    opt_obj.R_ops[i], B, C, Kd)
            # Tr(RR^*)
            Ji += compute_trace(opt_obj.comm, opt_obj.R_ops[i], \
                                opt_obj.R_ops[i], False, True)
            # -Tr(MR^*) - Tr(RM^*)
            Ji += -2.0*compute_trace(opt_obj.comm, opt_obj.R_ops[i], \
                                     M, False, True)
            # Tr(MM^*)
            Ji += compute_trace(opt_obj.comm, M, M, False, True)
            Ji *= opt_obj.weights[i]
            J += Ji
            M.destroy()
        
        # Stability-promoting component of the cost function
        lin_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.A, B, Kd, C)
        D_, _ = eigendecomposition(lin_op, lin_op.solve, 300, 20)
        D = -1./D_.getDiagonal().getArray()
        Jd = None
        if rank == 0:
            Jd = np.sum([opt_obj.evaluate_exponential(D[k]) \
                         for k in range (len(D))])
        J += opt_obj.comm.bcast(Jd, root=0)
        D_.destroy()
        B.destroy()
        C.destroy()
        return J
    
    
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):
        """
            Evaluate the euclidean gradient of the cost function with respect to the parameters
        """

        rank = opt_obj.comm.Get_rank()
        size = opt_obj.comm.Get_size()

        p, K, s = params
        grad_p = np.zeros(len(p))
        grad_s = np.zeros(len(s))
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
        grad_K = Kd.duplicate()
        
        for i in range (len(opt_obj.freqs)):

            wi = opt_obj.weights[i]

            M, Linv, KLinv = assemble_woodbury_low_rank_operator(opt_obj.comm, \
                                                    opt_obj.R_ops[i], B, C, Kd)
            
            # Gradient with respect to K
            Linv.hermitianTranspose()
            M.V.hermitianTranspose()
            F1 = V.matMult(Linv)
            M.V.hermitianTranspose()
            Linv.hermitianTranspose()
            F2 = M.apply_mat(F1)
            F2.axpy(-1.0,opt_obj.R_ops[i].apply_mat(F1))
            M.U.hermitianTranspose()
            grad_K_i = M.U.matMult(F2)
            M.U.hermitianTranspose()
            M.U.hermitianTranspose()
            KLinv.hermitianTranspose()
            grad_K_i.axpy(-1.0,M.U.matMult(C.matMult(KLinv)))
            KLinv.hermitianTranspose()
            M.U.hermitianTranspose()
            F1.destroy()
            F2.destroy()
            grad_K.axpy(2.0*wi,grad_K_i)
            grad_K_i.destroy()

            # Gradient with respect to p
            KLinv.hermitianTranspose()
            M.V.hermitianTranspose()
            F1 = V.matMult(Linv)
            M.V.hermitianTranspose()
            KLinv.hermitianTranspose()
            F2 = M.apply_mat(F1)
            F2.axpy(-1.0,opt_obj.R_ops[i].apply_mat(F1))
            grad_B_i = opt_obj.R_ops[i].apply_hermitian_transpose_mat(F2)
            B.hermitianTranspose()
            KLinv.hermitianTranspose()
            grad_B_i.axpy(-1.0,M.V.matMult(KLinv.matMult(B)))
            B.hermitianTranspose()
            KLinv.hermitianTranspose()
            F1.destroy()
            F2.destroy()
            for j in range (len(p)):
                pj = p.copy()
                pj[j] += 1e-3
                dB = opt_obj.compute_dB(p, pj)
                dB.conjugate()
                grad_p[j] += 2*wi*compute_double_contraction(opt_obj.comm, \
                                                             dB, grad_B_i).real
                dB.destroy()
            

            # Gradient with respect to s
            F1 = M.U.matMult(KLinv)
            F2 = M.apply_hermitian_transpose_mat(F1)
            F2.axpy(-1.0,opt_obj.R_ops[i].apply_hermitian_transpose_mat(F1))
            grad_C_i = opt_obj.R_ops[i].apply_mat(F2)
            C.hermitianTranspose()
            grad_C_i.axpy(-1.0,M.U.matMult(KLinv.matMult(C)))
            C.hermitianTranspose()
            F1.destroy()
            F2.destroy()
            for j in range (len(s)):
                sj = s.copy()
                sj[j] += 1e-3
                dC = opt_obj.compute_dC(s, sj)
                dC.conjugate()
                grad_s[j] += 2*wi*compute_double_contraction(opt_obj.comm, \
                                                             dC, grad_C_i).real
                dC.destroy()

        # Stability-promoting penalty
        V, D, W = opt_obj.fom.compute_spectrum(B,K,C,opt_obj.stab_reg[-1])
        M = np.diag([opt_obj.evaluate_gradient_exponential(D[k]) for k in range (len(D))])
        grad_p += -np.einsum('ijk,ij',dBdp.conj(),W@M@(V.conj().T@C.conj().T@K.conj().T))
        grad_K += -B.conj().T@W@M@(V.conj().T@C.conj().T)
        grad_s += -np.einsum('ij,ijk',(K.conj().T@B.conj().T@W)@M@V.conj().T,dCds.conj())
        
        
        return grad_p.real, grad_K.real, grad_s.real
    
    
    return cost, euclidean_gradient, euclidean_hessian
