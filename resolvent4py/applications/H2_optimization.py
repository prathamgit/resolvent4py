from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import time as tlib
import psutil

import os
import numpy as np
import scipy as sp
import pymanopt
from functools import partial

from ..io_functions import read_bv
from ..io_functions import read_coo_matrix
from ..linear_operators import MatrixLinearOperator
from ..linear_operators import LowRankLinearOperator
from ..linear_operators import LowRankUpdatedLinearOperator
from ..solvers_and_preconditioners_functions import create_mumps_solver
from ..applications import eig
from ..applications import right_and_left_eig
from ..comms import sequential_to_distributed_matrix
from ..comms import distributed_to_sequential_matrix
from ..linalg import compute_local_size
from ..linalg import hermitian_transpose
from ..miscellaneous import petscprint
from ..miscellaneous import create_dense_matrix
from ..linalg import bv_add
from ..linalg import bv_conj

def _compute_bv_contraction(comm, L1, L2):
    r"""
        Compute :math:`\sum_{i,j} M_{i,j}P_{i,j}`, where :math:`M = L_1` 
        and :math:`P = L_2` 
    """
    L1m = L1.getMat()
    L2m = L2.getMat()
    L1ma = L1m.getDenseArray()
    L2ma = L2m.getDenseArray()
    value = comm.allreduce(np.sum(L1ma*L2ma), op=MPI.SUM)
    L1.restoreMat(L1m)
    L2.restoreMat(L2m)
    return value

def _compute_trace_product(L1, L2, L2_hermitian_transpose=False):
    r"""
        If :code:`L2_hermitian_transpose==False`, compute
        :math:`\text{Tr}(L_1 L_2)`, else :math:`\text{Tr}(L_1 L_2^*)`.
    """
    if L2_hermitian_transpose == False:
        F1 = L2.U.dot(L1.V)
        F2 = L1.U.dot(L2.V)
        M = L1.Sigma@F1.getDenseArray()@L2.Sigma@F2.getDenseArray()
    else:
        F1 = L2.V.dot(L1.V)
        F2 = L1.U.dot(L2.U)
        M = L1.Sigma@F1.getDenseArray()@L2.Sigma.conj().T@F2.getDenseArray()
    F1.destroy()
    F2.destroy()
    return np.trace(M)


def _compute_woodbury_update(comm, R, B, K, C, RB, RHTC):
    r"""
        Assemble low-rank operator representation of 

        .. math::
            
            M = RBK\underbrace{\left(I + C^*RB K\right)^{-1}}_{L^{-1}}C^*R
    """
    RB = R.apply_mat(B, RB)
    RHTC = R.apply_hermitian_transpose_mat(C, RHTC)
    CHTRB = RB.dot(C)
    Sig = CHTRB.getDenseArray()@K
    Linv = sp.linalg.inv(np.eye(Sig.shape[0]) + Sig)
    M = LowRankLinearOperator(comm, RB, K@Linv, RHTC)
    return M, Linv

class optimization_objects:

    def __init__(self, comm, fnames_jacobian, jacobian_sizes, path_factors, \
                factor_sizes, fname_frequencies, fname_weights, \
                stability_params, compute_B, compute_C, compute_dB, compute_dC):
        
        self.comm = comm
        self.jacobian_sizes = jacobian_sizes
        A = read_coo_matrix(self.comm, fnames_jacobian, jacobian_sizes)
        A.scale(-1.0)
        ksp = create_mumps_solver(self.comm, A) if stability_params != None \
            else None
        self.A = MatrixLinearOperator(self.comm, A, ksp)
        self.freqs = np.load(fname_frequencies)
        self.weights = np.load(fname_weights)
        self.factor_sizes = factor_sizes
        self.load_factors(path_factors, factor_sizes)
        self.compute_B = compute_B
        self.compute_C = compute_C
        self.compute_dB = compute_dB
        self.compute_dC = compute_dC
        self.stab_params = stability_params
    
    def load_factors(self, path_factors, factor_sizes):
        
        self.U, self.S, self.V, self.R_ops = [], [], [], []
        for i in range (len(self.freqs)):
            fname_U = path_factors + 'omega_%1.5f/U.dat'%self.freqs[i]
            fname_S = path_factors + 'omega_%1.5f/S.npy'%self.freqs[i]
            fname_V = path_factors + 'omega_%1.5f/V.dat'%self.freqs[i]

            self.U.append(read_bv(self.comm, fname_U, factor_sizes))
            self.V.append(read_bv(self.comm, fname_V, factor_sizes))
            self.S.append(np.load(fname_S))
            self.R_ops.append(\
                LowRankLinearOperator(self.comm, self.U[-1], \
                                      self.S[-1], self.V[-1]))

    def evaluate_exponential(self, val):
        alpha, beta = self.stab_params[:2]
        try:    value = np.exp(alpha*(val.real + beta)).real
        except: value = 1e12
        value = 0.0 if value <= 1e-13 else value
        return value
    
    def evaluate_gradient_exponential(self, val):
        alpha = self.stab_params[0]
        return alpha*self.evaluate_exponential(val)
    
    def compute_dBdp(self, p, grad_B, compute_dB):
        grad_p = np.zeros_like(p)
        for j in range (len(p)):
            pjp = p.copy()
            pjm = p.copy()
            pjp[j] += 1e-5
            pjm[j] -= 1e-5
            dB = compute_dB(pjp, pjm)
            dB.scale(1./2e-5)
            bv_conj(dB)
            grad_p[j] = _compute_bv_contraction(self.comm, dB, grad_B).real
            dB.destroy()
        return grad_p
    
def cost_profile(params, opt_obj):
    r"""
        Evaluate the cost function

        :param params: a 3-tuple (actuator parameters, feedback gains, 
            sensor parameters)

        :rtype: float
    """
    comm = opt_obj.comm
    p, K, s = params

    t0 = tlib.time()

    B = opt_obj.compute_B(p)
    C = opt_obj.compute_C(s)
    RB = B.duplicate()
    RHTC = C.duplicate()

    # H2 component of the cost function
    J = 0
    for i in range (len(opt_obj.freqs)):
        w = opt_obj.weights[i]
        R = opt_obj.R_ops[i]
        M, _ = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)

        J += w*np.sum(np.diag(R.Sigma)**2)
        J -= 2.0*w*_compute_trace_product(R, M, True)
        J += w*_compute_trace_product(M, M, True)
        
        # process = psutil.Process(os.getpid())
        # value = process.memory_info().rss/(1024 * 1024)
        # value = sum(comm.allgather(value))
        # if opt_obj.comm.Get_rank() == 0:
        #     print(f"Iteration {i} usage {value} MB")
    
    # Stability-promoting component of the cost function
    if opt_obj.stab_params != None:
        cl_op = LowRankUpdatedLinearOperator(comm, opt_obj.A, B, K, C)
        krylov_dim, n_evals = opt_obj.stab_params[2], opt_obj.stab_params[3]
        D, E = eig(cl_op, cl_op.solve, krylov_dim, n_evals, lambda x: -1./x)
        D = np.diag(D)
        J += np.sum([opt_obj.evaluate_exponential(val) for val in D])
        E.destroy()
        cl_op.destroy_woodbury_operator()
        cl_op.destroy_intermediate_vectors()
            
    bvs = [B, C, RB, RHTC]
    for bv in bvs: bv.destroy()

    t1 = tlib.time()
    petscprint(opt_obj.comm, "Execution time = %1.5f [sec]"%(t1 - t0))
    return J.real

def euclidean_gradient_profile(params, opt_obj):
    r"""
        Evaluate the gradient of the cost function

        :param params: a 3-tuple (actuator parameters p, feedback gains K, 
            sensor parameters s)

        :return: (:math:`\nabla_p J`, :math:`\nabla_K J`, 
            :math:`\nabla_s J`)
    """
    comm = opt_obj.comm
    p, K, s = params

    t0 = tlib.time()

    B = opt_obj.compute_B(p)
    C = opt_obj.compute_C(s)
    RB = B.duplicate()
    RHTC = C.duplicate()
    
    grad_K = np.zeros(K.shape, dtype=np.complex128)
    grad_p = np.zeros_like(p)
    grad_s = np.zeros_like(s)

    # Allocate memory for efficiency (grad_K)
    F1K = SLEPc.BV().create(comm=opt_obj.comm)
    F1K.setSizes(opt_obj.R_ops[0]._dimensions[0], C.getSizes()[-1])
    F1K.setType('mat')
    F2K = F1K.duplicate()
    F3K = F1K.duplicate()

    # Allocate memory for efficiency (grad_p)
    F1B = SLEPc.BV().create(comm=opt_obj.comm)
    F1B.setSizes(opt_obj.R_ops[0]._dimensions[0], B.getSizes()[-1])
    F1B.setType('mat')
    F2B = F1B.duplicate()
    F3B = F1B.duplicate()
    grad_B_i = F1B.duplicate()
    Q_grad_B_i = F1B.duplicate()

    # Allocate memory for efficiency (grad_s)
    F1C = SLEPc.BV().create(comm=opt_obj.comm)
    F1C.setSizes(opt_obj.R_ops[0]._dimensions[-1], C.getSizes()[-1])
    F1C.setType('mat')
    F2C = F1C.duplicate()
    F3C = F1C.duplicate()
    grad_C_i = F1C.duplicate()
    Q_grad_C_i = F1C.duplicate()

    for i in range (len(opt_obj.freqs)):
        
        w = opt_obj.weights[i]
        R = opt_obj.R_ops[i]
        M, Linv = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)

        # Grad with respect to K
        LiCT = Linv.conj().T
        LiCTMat = PETSc.Mat().createDense(LiCT.shape, None, LiCT, MPI.COMM_SELF)
        F1K.mult(1.0, 0.0, M.V, LiCTMat)
        F2K = R.apply_mat(F1K, F2K)
        F3K = M.apply_mat(F1K, F3K)
        bv_add(-1.0, F3K, F2K)
        grad_K_i_mat = F3K.dot(M.U)
        F4K = C.dot(M.U)
        grad_K_i = grad_K_i_mat.getDenseArray()
        grad_K_i -= F4K.getDenseArray()@M.Sigma.conj().T@grad_K_i
        grad_K += 2.0*w*grad_K_i
        mats = [LiCTMat, grad_K_i_mat, F4K]
        for mat in mats: mat.destroy()

        # Grad with respect to p
        S = M.Sigma.conj().T
        SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
        F1B.mult(1.0, 0.0, M.V, SMat)
        F2B = R.apply_mat(F1B, F2B)
        F3B = M.apply_mat(F1B, F3B)
        bv_add(-1.0, F3B, F2B)
        grad_B_i = R.apply_hermitian_transpose_mat(F3B, grad_B_i)
        Q = LowRankLinearOperator(comm, M.V, M.Sigma.conj().T, B)
        Q_grad_B_i = Q.apply_mat(grad_B_i, Q_grad_B_i)
        bv_add(-1.0, grad_B_i, Q_grad_B_i)
        grad_p += 2.0*w*opt_obj.compute_dBdp(p, grad_B_i, opt_obj.compute_dB)
        SMat.destroy()

        # Grad with respect to s
        S = M.Sigma.copy()
        SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
        F1C.mult(1.0, 0.0, M.U, SMat)
        F2C = R.apply_hermitian_transpose_mat(F1C, F2C)
        F3C = M.apply_hermitian_transpose_mat(F1C, F3C)
        bv_add(-1.0, F3C, F2C)
        grad_C_i = R.apply_mat(F3C, grad_C_i)
        Q = LowRankLinearOperator(comm, M.U, M.Sigma, C)
        Q_grad_C_i = Q.apply_mat(grad_C_i, Q_grad_C_i)
        bv_add(-1.0, grad_C_i, Q_grad_C_i)
        grad_s += 2.0*w*opt_obj.compute_dBdp(s, grad_C_i, opt_obj.compute_dC)
        SMat.destroy()

        # process = psutil.Process(os.getpid())
        # value = process.memory_info().rss/(1024 * 1024)
        # value = sum(comm.allgather(value))
        # if opt_obj.comm.Get_rank() == 0:
        #     print(f"Iteration {i} usage {value} MB")

    # Stability-promoting penalty
    if opt_obj.stab_params != None:
        cl_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.A, B, K, C)
        kry_dim, n_evals = opt_obj.stab_params[2], opt_obj.stab_params[3]
        V, D, W = right_and_left_eig(cl_op, cl_op.solve, kry_dim, n_evals, \
                                                    lambda x: -1./x)
        cl_op.destroy_woodbury_operator()
        cl_op.destroy_intermediate_vectors()

        D = np.diag(D)
        MD = np.diag([opt_obj.evaluate_gradient_exponential(v) for v in D])
        BHTW = W.dot(B)
        VHTC = C.dot(V)

        # Grad with respect to K
        grad_K -= BHTW.getDenseArray()@MD@VHTC.getDenseArray()
        # Grad with respect to p
        M_data = MD@VHTC.getDenseArray()@K.conj().T
        M = PETSc.Mat().createDense(M_data.shape, None, M_data, MPI.COMM_SELF)
        grad_B_i.mult(1.0, 0.0, W, M)
        grad_p -= opt_obj.compute_dBdp(p, grad_B_i, opt_obj.compute_dB)
        M.destroy()
        # Grad with respect to s
        M_data = MD@BHTW.getDenseArray().conj().T@K
        M = PETSc.Mat().createDense(M_data.shape, None, M_data, MPI.COMM_SELF)
        grad_C_i.mult(1.0, 0.0, V, M)
        grad_s -= opt_obj.compute_dBdp(s, grad_C_i, opt_obj.compute_dC)
        M.destroy()

        objects = [V, W, BHTW, VHTC]
        for obj in objects: obj.destroy()

    objects = [F1K, F2K, F3K,F1B, F2B, F3B, F1C, F2C, F3C, grad_B_i, \
               Q_grad_B_i, grad_C_i, Q_grad_C_i]
    for obj in objects: obj.destroy()

    t1 = tlib.time()
    petscprint(opt_obj.comm, "Execution time = %1.5f [sec]"%(t1 - t0))
    
    return grad_p, grad_K.real, grad_s

def create_objective_and_gradient(manifold,opt_obj):
    r"""
        Create functions to evaluate the cost function and the gradient

        :param manifold: one of the pymanopt manifolds
        :param opt_obj: instance of the optimization objects class

        :return: (cost function, gradient, hessian=:code:`None`)
        :rtype: (Callable, Callable, None)
    """
    
    euclidean_hessian = None

    @pymanopt.function.numpy(manifold)
    def cost(*params):
        r"""
            Evaluate the cost function

            :param params: a 3-tuple (actuator parameters, feedback gains, 
                sensor parameters)

            :rtype: float
        """
        comm = opt_obj.comm
        p, K, s = params

        t0 = tlib.time()

        B = opt_obj.compute_B(p)
        C = opt_obj.compute_C(s)
        RB = B.duplicate()
        RHTC = C.duplicate()

        # H2 component of the cost function
        J = 0
        for i in range (len(opt_obj.freqs)):
            w = opt_obj.weights[i]
            R = opt_obj.R_ops[i]
            M, _ = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)

            J += w*np.sum(np.diag(R.Sigma)**2)
            J -= 2.0*w*_compute_trace_product(R, M, True)
            J += w*_compute_trace_product(M, M, True)
            
            # process = psutil.Process(os.getpid())
            # value = process.memory_info().rss/(1024 * 1024)
            # value = sum(comm.allgather(value))
            # if opt_obj.comm.Get_rank() == 0:
            #     print(f"Iteration {i} usage {value} MB")
        
        # Stability-promoting component of the cost function
        if opt_obj.stab_params != None:
            cl_op = LowRankUpdatedLinearOperator(comm, opt_obj.A, B, K, C)
            krylov_dim, n_evals = opt_obj.stab_params[2], opt_obj.stab_params[3]
            D, E = eig(cl_op, cl_op.solve, krylov_dim, n_evals, lambda x: -1./x)
            D = np.diag(D)
            J += np.sum([opt_obj.evaluate_exponential(val) for val in D])
            E.destroy()
            cl_op.destroy_woodbury_operator()
            cl_op.destroy_intermediate_vectors()
                
        bvs = [B, C, RB, RHTC]
        for bv in bvs: bv.destroy()

        t1 = tlib.time()
        petscprint(opt_obj.comm, "Cost exec time = %1.5f [sec]"%(t1 - t0))
        return J.real

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):
        r"""
            Evaluate the gradient of the cost function

            :param params: a 3-tuple (actuator parameters p, feedback gains K, 
                sensor parameters s)
            
            :return: (:math:`\nabla_p J`, :math:`\nabla_K J`, 
                :math:`\nabla_s J`)
        """
        comm = opt_obj.comm
        p, K, s = params

        t0 = tlib.time()

        B = opt_obj.compute_B(p)
        C = opt_obj.compute_C(s)
        RB = B.duplicate()
        RHTC = C.duplicate()
        
        grad_K = np.zeros(K.shape, dtype=np.complex128)
        grad_p = np.zeros_like(p)
        grad_s = np.zeros_like(s)

        # Allocate memory for efficiency (grad_K)
        F1K = SLEPc.BV().create(comm=opt_obj.comm)
        F1K.setSizes(opt_obj.R_ops[0]._dimensions[0], C.getSizes()[-1])
        F1K.setType('mat')
        F2K = F1K.duplicate()
        F3K = F1K.duplicate()

        # Allocate memory for efficiency (grad_p)
        F1B = SLEPc.BV().create(comm=opt_obj.comm)
        F1B.setSizes(opt_obj.R_ops[0]._dimensions[0], B.getSizes()[-1])
        F1B.setType('mat')
        F2B = F1B.duplicate()
        F3B = F1B.duplicate()
        grad_B_i = F1B.duplicate()
        Q_grad_B_i = F1B.duplicate()

        # Allocate memory for efficiency (grad_s)
        F1C = SLEPc.BV().create(comm=opt_obj.comm)
        F1C.setSizes(opt_obj.R_ops[0]._dimensions[-1], C.getSizes()[-1])
        F1C.setType('mat')
        F2C = F1C.duplicate()
        F3C = F1C.duplicate()
        grad_C_i = F1C.duplicate()
        Q_grad_C_i = F1C.duplicate()

        for i in range (len(opt_obj.freqs)):
            
            w = opt_obj.weights[i]
            R = opt_obj.R_ops[i]
            M, Linv = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)

            # Grad with respect to K
            LiCT = Linv.conj().T
            LiCTMat = PETSc.Mat().createDense(LiCT.shape, None, LiCT, MPI.COMM_SELF)
            F1K.mult(1.0, 0.0, M.V, LiCTMat)
            F2K = R.apply_mat(F1K, F2K)
            F3K = M.apply_mat(F1K, F3K)
            bv_add(-1.0, F3K, F2K)
            grad_K_i_mat = F3K.dot(M.U)
            F4K = C.dot(M.U)
            grad_K_i = grad_K_i_mat.getDenseArray()
            grad_K_i -= F4K.getDenseArray()@M.Sigma.conj().T@grad_K_i
            grad_K += 2.0*w*grad_K_i
            mats = [LiCTMat, grad_K_i_mat, F4K]
            for mat in mats: mat.destroy()

            # Grad with respect to p
            S = M.Sigma.conj().T
            SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
            F1B.mult(1.0, 0.0, M.V, SMat)
            F2B = R.apply_mat(F1B, F2B)
            F3B = M.apply_mat(F1B, F3B)
            bv_add(-1.0, F3B, F2B)
            grad_B_i = R.apply_hermitian_transpose_mat(F3B, grad_B_i)
            Q = LowRankLinearOperator(comm, M.V, M.Sigma.conj().T, B)
            Q_grad_B_i = Q.apply_mat(grad_B_i, Q_grad_B_i)
            bv_add(-1.0, grad_B_i, Q_grad_B_i)
            grad_p += 2.0*w*opt_obj.compute_dBdp(p, grad_B_i, opt_obj.compute_dB)
            SMat.destroy()

            # Grad with respect to s
            S = M.Sigma.copy()
            SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
            F1C.mult(1.0, 0.0, M.U, SMat)
            F2C = R.apply_hermitian_transpose_mat(F1C, F2C)
            F3C = M.apply_hermitian_transpose_mat(F1C, F3C)
            bv_add(-1.0, F3C, F2C)
            grad_C_i = R.apply_mat(F3C, grad_C_i)
            Q = LowRankLinearOperator(comm, M.U, M.Sigma, C)
            Q_grad_C_i = Q.apply_mat(grad_C_i, Q_grad_C_i)
            bv_add(-1.0, grad_C_i, Q_grad_C_i)
            grad_s += 2.0*w*opt_obj.compute_dBdp(s, grad_C_i, opt_obj.compute_dC)
            SMat.destroy()

            # process = psutil.Process(os.getpid())
            # value = process.memory_info().rss/(1024 * 1024)
            # value = sum(comm.allgather(value))
            # if opt_obj.comm.Get_rank() == 0:
            #     print(f"Iteration {i} usage {value} MB")

        # Stability-promoting penalty
        if opt_obj.stab_params != None:
            cl_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.A, B, K, C)
            kry_dim, n_evals = opt_obj.stab_params[2], opt_obj.stab_params[3]
            V, D, W = right_and_left_eig(cl_op, cl_op.solve, kry_dim, n_evals, \
                                                        lambda x: -1./x)
            cl_op.destroy_woodbury_operator()
            cl_op.destroy_intermediate_vectors()

            D = np.diag(D)
            MD = np.diag([opt_obj.evaluate_gradient_exponential(v) for v in D])
            BHTW = W.dot(B)
            VHTC = C.dot(V)

            # Grad with respect to K
            grad_K -= BHTW.getDenseArray()@MD@VHTC.getDenseArray()
            # Grad with respect to p
            M_data = MD@VHTC.getDenseArray()@K.conj().T
            M = PETSc.Mat().createDense(M_data.shape, None, M_data, MPI.COMM_SELF)
            grad_B_i.mult(1.0, 0.0, W, M)
            grad_p -= opt_obj.compute_dBdp(p, grad_B_i, opt_obj.compute_dB)
            M.destroy()
            # Grad with respect to s
            M_data = MD@BHTW.getDenseArray().conj().T@K
            M = PETSc.Mat().createDense(M_data.shape, None, M_data, MPI.COMM_SELF)
            grad_C_i.mult(1.0, 0.0, V, M)
            grad_s -= opt_obj.compute_dBdp(s, grad_C_i, opt_obj.compute_dC)
            M.destroy()

            objects = [V, W, BHTW, VHTC]
            for obj in objects: obj.destroy()
        
        objects = [F1K, F2K, F3K,F1B, F2B, F3B, F1C, F2C, F3C, grad_B_i, \
                Q_grad_B_i, grad_C_i, Q_grad_C_i]
        for obj in objects: obj.destroy()

        t1 = tlib.time()
        petscprint(opt_obj.comm, "Gradient exec time = %1.5f [sec]"%(t1 - t0))
        
        return grad_p, grad_K.real, grad_s
    
    return cost, euclidean_gradient, euclidean_hessian


def test_euclidean_gradient(M, opt_obj, params, eps):
    
    xa, K, xs = params
    comm = opt_obj.comm
    rank = comm.Get_rank()
    
    cost = partial(cost_profile, opt_obj=opt_obj)
    grad = partial(euclidean_gradient_profile, opt_obj=opt_obj)
    # cost, grad, _ = create_objective_and_gradient(M,opt_obj)
    J = cost(params)
    grad_xa, grad_K, grad_xs = grad(params)
    
    petscprint(comm,"Cost = %1.15e"%J)
    
    # Check Sb gradient
    if rank == 0:
        delta = np.random.randn(*xa.shape)
        delta /= np.linalg.norm(delta)
        params_ = (xa + eps*delta, K, xs)
        # print(grad_xa)
    else:
        params_ = None
    
    params_ = comm.bcast(params_,root=0)
    dfd = (cost(params_) - J)/eps

    if rank == 0:
        dgrad = (delta.conj().T@grad_xa).real
        error = np.abs(dfd - dgrad)
        percent_error = (error/np.abs(dfd)*100).real
        
        petscprint(comm,"------ Error xa grad -------------")
        petscprint(comm,"dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        petscprint(comm,"---------------------------------")
        
    
    # Check K gradient
    if rank == 0:
        delta = np.random.randn(*K.shape)
        delta /= np.sqrt(np.trace(delta.T@delta))
        params_ = (xa, K + eps*delta, xs)   
        # print(grad_K)
    else:
        params_ = None
    
    params_ = comm.bcast(params_,root=0)
    dfd = (cost(params_) - J)/eps

    if rank == 0:
        dgrad = np.trace(delta.conj().T@grad_K).real
        error = np.abs(dfd - dgrad)
        percent_error = (error/np.abs(dfd)*100).real
        
        petscprint(comm,"------ Error K grad -------------")
        petscprint(comm,"dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        petscprint(comm,"---------------------------------")
        
    
    # Check Sc gradient
    if rank == 0:
        delta = np.random.randn(*xs.shape)
        delta /= np.linalg.norm(delta)
        params_ = (xa, K, xs + eps*delta)   
        # print(grad_xs)
    else:
        params_ = None

    params_ = comm.bcast(params_,root=0)
    dfd = (cost(params_) - J)/eps

    if rank == 0:
        dgrad = (delta.conj().T@grad_xs).real
        error = np.abs(dfd - dgrad)
        percent_error = (error/np.abs(dfd)*100).real
        
        petscprint(comm,"------ Error xs grad -------------")
        petscprint(comm,"dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        petscprint(comm,"---------------------------------")
        
    

