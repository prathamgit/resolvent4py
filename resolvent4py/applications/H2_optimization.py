import time as tlib
import psutil

import matplotlib.pyplot as plt
from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc
from .. import pymanopt

from ..io_functions import read_bv
from ..io_functions import read_coo_matrix
from ..linear_operators import MatrixLinearOperator
from ..linear_operators import LowRankLinearOperator
from ..linear_operators import LowRankUpdatedLinearOperator
from ..linear_operators import ProductLinearOperator
from ..solvers_and_preconditioners_functions import create_mumps_solver
from ..applications import eig
from ..miscellaneous import petscprint
from ..linalg import bv_add
from ..linalg import bv_conj
from ..io_functions import write_to_file


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

def _compute_trace_product_new(L1, L2, DDHT, FFHT):
    Ln = getattr(FFHT, FFHT.names[-1])
    if Ln._name == 'LowRankLinearOperator':
        F1 = L1.apply_mat(Ln.U)
        F2 = L2.apply_mat(Ln.U)
        F3 = DDHT.apply_mat(F1)
        F4 = F3.dot(F2)
        M = F4.getDenseArray()
    else:
        F1 = DDHT.apply_mat(L1.U)
        F2 = L2.apply_hermitian_transpose_mat(F1)
        F3 = FFHT.apply_mat(F2)
        F4 = F3.dot(L1.V)
        M = L1.Sigma@F4.getDenseArray()
    objs = [F1, F2, F3, F4]
    for obj in objs: obj.destroy()
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

class InputOutputMatrices:

    def __init__(self, comm, compute_B, compute_C, compute_dB, compute_dC, \
                 D, F):
        self.comm = comm
        self.compute_B = compute_B
        self.compute_C = compute_C
        self.compute_dB = compute_dB
        self.compute_dC = compute_dC
        D_actions = [D.apply_hermitian_transpose, D.apply]
        F_actions = [F.apply_hermitian_transpose, F.apply]
        self.DDHT = ProductLinearOperator(comm, [D, D], D_actions)
        self.FFHT = ProductLinearOperator(comm, [F, F], F_actions)

    def compute_dBdp(self, p, grad_B, Bp1, Bp0, compute_dB):
        eps = 1e-4
        grad_p = np.zeros_like(p)
        for j in range (len(p)):
            pjp = p.copy()
            pjm = p.copy()
            pjp[j] += eps
            pjm[j] -= eps
            Bp1 = compute_dB(pjp, pjm, Bp1, Bp0)
            Bp1.scale(1./(2*eps))
            bv_conj(Bp1)
            grad_p[j] = _compute_bv_contraction(self.comm, Bp1, grad_B).real
        return grad_p


class H2Component:

    def __init__(self, comm, path_factors, factor_sizes, frequencies, weights):
        self.comm = comm
        self.freqs = frequencies
        self.weights = weights
        self.load_factors(path_factors, factor_sizes)

    def load_factors(self, path_factors, factor_sizes):
        self.U, self.S, self.V, self.R_ops = [], [], [], []
        for i in range (len(self.freqs)):
            # t0 = tlib.time()
            fname_U = path_factors + 'omega_%1.5f/U.dat'%self.freqs[i]
            fname_S = path_factors + 'omega_%1.5f/S.npy'%self.freqs[i]
            fname_V = path_factors + 'omega_%1.5f/V.dat'%self.freqs[i]
            self.U.append(read_bv(self.comm, fname_U, factor_sizes))
            self.V.append(read_bv(self.comm, fname_V, factor_sizes))
            self.S.append(np.load(fname_S))
            self.R_ops.append(\
                LowRankLinearOperator(self.comm, self.U[-1], \
                                      self.S[-1], self.V[-1]))
            # dt = tlib.time() - t0
            # petscprint(self.comm, "Loaded freq = %d/%d (%1.3e)"%(i, len(self.freqs), dt))
            
    def evaluate_cost(self, params, IOMats):
        p, K, s = params
        B = SLEPc.BV().create(comm=self.comm)
        B.setSizes(self.R_ops[0]._dimensions[0], K.shape[0])
        B.setType('mat')
        C = SLEPc.BV().create(comm=self.comm)
        C.setSizes(self.R_ops[0]._dimensions[0], K.shape[-1])
        C.setType('mat')
        B = IOMats.compute_B(p, B)
        C = IOMats.compute_C(s, C)
        RB = B.duplicate()
        RHTC = C.duplicate()
        J = 0

        Ln = getattr(IOMats.FFHT, IOMats.FFHT.names[-1])
        Qn = getattr(IOMats.DDHT, IOMats.DDHT.names[-1])
        if Ln._name != 'LowRankLinearOperator':
            for i in range (len(self.freqs)):
                w = self.weights[i]
                R = self.R_ops[i]
                M, _ = _compute_woodbury_update(self.comm, R, B, K, C, \
                                                RB, RHTC)
                J += w*_compute_trace_product_new(R, R, IOMats.DDHT, \
                                                  IOMats.FFHT)
                J -= 2.0*w*_compute_trace_product_new(R, M, IOMats.DDHT, \
                                                    IOMats.FFHT)
                J += w*_compute_trace_product_new(M, M, IOMats.DDHT, \
                                                  IOMats.FFHT)
        else:
            F1 = Ln.U.duplicate()
            F2 = Ln.U.duplicate()
            F3 = Qn.create_left_bv(Ln.U.getSizes()[-1])
            for i in range (len(self.freqs)):
                w = self.weights[i]
                R = self.R_ops[i]
                M, _ = _compute_woodbury_update(self.comm, R, B, K, C, \
                                                RB, RHTC)
                F1 = R.apply_mat(Ln.U, F1)
                F2 = M.apply_mat(Ln.U, F2)
                bv_add(-1.0, F1, F2)
                F3 = Qn.apply_hermitian_transpose_mat(F1, F3)
                Mat = F3.dot(F3)
                J += w*np.trace(Mat.getDenseArray())
            objs = [F1, F2, F3]
            for obj in objs: obj.destroy()

        objs = [B, C, RB, RHTC]
        for obj in objs: obj.destroy()
        return J.real
    
    def evaluate_gradient(self, params, IOMats):
        comm = self.comm
        p, K, s = params

        B = SLEPc.BV().create(comm=comm)
        B.setSizes(self.R_ops[0]._dimensions[0], K.shape[0])
        B.setType('mat')
        Bp1 = B.copy()
        Bp0 = B.copy()
        C = SLEPc.BV().create(comm=comm)
        C.setSizes(self.R_ops[0]._dimensions[0], K.shape[-1])
        C.setType('mat')
        Cp1 = C.copy()
        Cp0 = C.copy()
        B = IOMats.compute_B(p, B)
        C = IOMats.compute_C(s, C)
        RB = B.duplicate()
        RHTC = C.duplicate()
        
        grad_K = np.zeros(K.shape, dtype=np.complex128)
        grad_p = np.zeros_like(p)
        grad_s = np.zeros_like(s)

        # Allocate memory for efficiency (grad_K)
        F1K = SLEPc.BV().create(comm=comm)
        F1K.setSizes(self.R_ops[0]._dimensions[0], C.getSizes()[-1])
        F1K.setType('mat')
        F2K = F1K.duplicate()
        F3K = F1K.duplicate()
        F5K = F1K.duplicate()
        F6K = F1K.duplicate()

        # Allocate memory for efficiency (grad_p)
        F1B = SLEPc.BV().create(comm=comm)
        F1B.setSizes(self.R_ops[0]._dimensions[0], B.getSizes()[-1])
        F1B.setType('mat')
        F2B = F1B.duplicate()
        F3B = F1B.duplicate()
        F5B = F1B.duplicate()
        F6B = F1B.duplicate()
        grad_B_i = F1B.duplicate()
        Q_grad_B_i = F1B.duplicate()

        # Allocate memory for efficiency (grad_s)
        F1C = SLEPc.BV().create(comm=comm)
        F1C.setSizes(self.R_ops[0]._dimensions[-1], C.getSizes()[-1])
        F1C.setType('mat')
        F2C = F1C.duplicate()
        F3C = F1C.duplicate()
        F5C = F1C.duplicate()
        F6C = F1C.duplicate()
        grad_C_i = F1C.duplicate()
        Q_grad_C_i = F1C.duplicate()

        for i in range (len(self.freqs)):
            
            w = self.weights[i]
            R = self.R_ops[i]
            M, Linv = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)

            # Grad with respect to K
            LiCT = Linv.conj().T
            LiCTMat = PETSc.Mat().createDense(LiCT.shape, None, LiCT, \
                                              MPI.COMM_SELF)
            F1K.mult(1.0, 0.0, M.V, LiCTMat)
            F5K = IOMats.FFHT.apply_mat(F1K, F5K)
            F2K = R.apply_mat(F5K, F2K)     # Used to be F1K instead of F5K
            F3K = M.apply_mat(F5K, F3K)     # Used to be F1K instead of F5K
            bv_add(-1.0, F3K, F2K)
            F6K = IOMats.DDHT.apply_mat(F3K, F6K)
            grad_K_i_mat = F6K.dot(M.U)     # Used to be F3K instead of F6K
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
            F5B = IOMats.FFHT.apply_mat(F1B, F5B)
            F2B = R.apply_mat(F5B, F2B)     # Used to be F1B instead of F5B
            F3B = M.apply_mat(F5B, F3B)     # Used to be F1B instead of F5B
            bv_add(-1.0, F3B, F2B)
            F6B = IOMats.DDHT.apply_mat(F3B, F6B)
            grad_B_i = R.apply_hermitian_transpose_mat(F6B, grad_B_i) # Used to be F3B instead of F6B
            Q = LowRankLinearOperator(comm, M.V, M.Sigma.conj().T, B)
            Q_grad_B_i = Q.apply_mat(grad_B_i, Q_grad_B_i)
            bv_add(-1.0, grad_B_i, Q_grad_B_i)
            grad_p += 2.0*w*IOMats.compute_dBdp(p, grad_B_i, Bp1, Bp0, \
                                                IOMats.compute_dB)
            SMat.destroy()

            # Grad with respect to s
            S = M.Sigma.copy()
            SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
            F1C.mult(1.0, 0.0, M.U, SMat)
            F5C = IOMats.DDHT.apply_mat(F1C, F5C)
            F2C = R.apply_hermitian_transpose_mat(F5C, F2C)     # Used to be F1C instead of F5C
            F3C = M.apply_hermitian_transpose_mat(F5C, F3C)     # Used to be F1C instead of F5C
            bv_add(-1.0, F3C, F2C)
            F6C = IOMats.FFHT.apply_mat(F3C, F6C)   # Used to be F3C instead of F6C
            grad_C_i = R.apply_mat(F6C, grad_C_i)
            Q = LowRankLinearOperator(comm, M.U, M.Sigma, C)
            Q_grad_C_i = Q.apply_mat(grad_C_i, Q_grad_C_i)
            bv_add(-1.0, grad_C_i, Q_grad_C_i)
            grad_s += 2.0*w*IOMats.compute_dBdp(s, grad_C_i, Cp1, Cp0, \
                                                IOMats.compute_dC)
            SMat.destroy()

            # process = psutil.Process(os.getpid())
            # value = process.memory_info().rss/(1024 * 1024)
            # value = sum(comm.allgather(value))
            # if opt_obj.comm.Get_rank() == 0:
            #     print(f"Iteration {i} usage {value} MB")
        objects = [F1K, F2K, F3K, F1B, F2B, F3B, F1C, F2C, F3C, grad_B_i, \
                   Q_grad_B_i, grad_C_i, Q_grad_C_i, Bp1, Bp0, Cp1, Cp0, B, C, \
                   RB, RHTC, F5K, F6K, F5B, F6B, F5C, F6C]
        for obj in objects: obj.destroy()
        return grad_p.real, grad_K.real, grad_s.real

class BodeFormula:

    def __init__(self, comm, path_factors, factor_sizes, frequencies, weights, \
                 open_loop_eigvals, sigma, alpha, eps):
        self.comm = comm
        self.freqs = frequencies
        self.weights = weights
        self.load_factors(path_factors, factor_sizes)
        self.open_loop_eigvals = open_loop_eigvals
        self.sigma = sigma
        self.alpha = alpha
        self.eps = eps

    def load_factors(self, path_factors, factor_sizes):
        self.U, self.S, self.V, self.R_ops = [], [], [], []
        for i in range (len(self.freqs)):
            # t0 = tlib.time()
            fname_U = path_factors + 'omega_%1.5f/U.dat'%self.freqs[i]
            fname_S = path_factors + 'omega_%1.5f/S.npy'%self.freqs[i]
            fname_V = path_factors + 'omega_%1.5f/V.dat'%self.freqs[i]
            self.U.append(read_bv(self.comm, fname_U, factor_sizes))
            self.V.append(read_bv(self.comm, fname_V, factor_sizes))
            self.S.append(np.load(fname_S))
            self.R_ops.append(\
                LowRankLinearOperator(self.comm, self.U[-1], \
                                      self.S[-1], self.V[-1]))
            # dt = tlib.time() - t0
            # petscprint(self.comm, "Loaded freq = %d/%d (%1.3e)"%(i, len(self.freqs), dt))

    def evaluate_cost(self, params, IOMats):
        p, K, s = params
        Id = np.eye(K.shape[-1])
        B = SLEPc.BV().create(comm=self.comm)
        B.setSizes(self.R_ops[0]._dimensions[0], K.shape[0])
        B.setType('mat')
        C = SLEPc.BV().create(comm=self.comm)
        C.setSizes(self.R_ops[0]._dimensions[0], K.shape[-1])
        C.setType('mat')
        B = IOMats.compute_B(p, B)
        C = IOMats.compute_C(s, C)
        RB = B.duplicate()
        integral = 0
        kernel = np.zeros(len(self.freqs)-1)
        f = 1./np.pi if np.min(self.freqs) > 0.0 else 1./(2*np.pi)
        for i in range (len(self.freqs) - 1):
            RB = self.R_ops[i].apply_mat(B, RB)
            CRB = RB.dot(C)
            S = sp.linalg.inv(Id + CRB.getDenseArray()@K)
            _, sv, _ = sp.linalg.svd(S)
            integral += f*self.weights[i]*np.sum(np.log(sv))
            CRB.destroy()
            kernel[i] = np.sum(np.log(sv))
        RB = self.R_ops[-1].apply_mat(B, RB)
        CRB = RB.dot(C)
        L = CRB.getDenseArray()@K
        sumRe = np.sum(self.open_loop_eigvals.real - self.sigma) - \
                0.5*np.trace(1j*self.freqs[-1]*L).real - integral
        objs = [RB, CRB, B, C]
        for obj in objs: obj.destroy()
        # if self.comm.Get_rank() == 0:
        #     plt.figure()
        #     plt.plot(self.freqs[:-1], kernel)
        #     plt.axhline(y=0.0, color='r', alpha=0.5)
        #     plt.savefig("kernel.png")
        exp_arg = self.alpha*np.abs(sumRe)
        value = sumRe**2*np.exp(exp_arg) if exp_arg < np.log(1e13) else 1e13
        return value

    def evaluate_gradient(self, params, IOMats):
        p, K, s = params
        Id = np.eye(K.shape[-1])
        B = SLEPc.BV().create(comm=self.comm)
        B.setSizes(self.R_ops[0]._dimensions[0], K.shape[0])
        B.setType('mat')
        B = IOMats.compute_B(p, B)
        grad_B_i = B.duplicate()
        Bp1 = B.duplicate()
        Bp0 = B.duplicate()
        RB = B.duplicate()

        C = SLEPc.BV().create(comm=self.comm)
        C.setSizes(self.R_ops[0]._dimensions[0], K.shape[-1])
        C.setType('mat')
        C = IOMats.compute_C(s, C)
        RHTC = C.duplicate()
        grad_C_i = C.duplicate()
        Cp1 = C.duplicate()
        Cp0 = C.duplicate()

        grad_K = np.zeros(K.shape, dtype=np.complex128)
        grad_p = np.zeros_like(p)
        grad_s = np.zeros_like(s)

        integral = 0.0
        
        f = 1./np.pi if np.min(self.freqs) > 0.0 else 1./(2*np.pi)
        for i in range (len(self.freqs) - 1):
            RHTC = self.R_ops[i].apply_hermitian_transpose_mat(C, RHTC)
            RB = self.R_ops[i].apply_mat(B, RB)
            CRB = RB.dot(C)
            CRBa = CRB.getDenseArray()
            S = sp.linalg.inv(Id + CRBa@K)
            _, sv, _ = sp.linalg.svd(S)
            integral += f*self.weights[i]*np.sum(np.log(sv))
            
            grad_K += f*self.weights[i]*CRBa.conj().T@S.conj().T
            Mat = S.conj().T@K.conj().T
            Mat_ = PETSc.Mat().createDense(Mat.shape, None, Mat, MPI.COMM_SELF)
            grad_B_i.mult(f*self.weights[i], 0.0, RHTC, Mat_)
            grad_p += IOMats.compute_dBdp(p, grad_B_i, Bp1, \
                                          Bp0, IOMats.compute_dB)
            Mat_.hermitianTranspose(None)
            grad_C_i.mult(f*self.weights[i], 0.0, RB, Mat_)
            grad_s += IOMats.compute_dBdp(s, grad_C_i, Cp1, Cp0, \
                                          IOMats.compute_dC)
            
            Mat_.destroy()
            CRB.destroy()
        
        RHTC = self.R_ops[-1].apply_hermitian_transpose_mat(C, RHTC)
        RB = self.R_ops[-1].apply_mat(B, RB)
        CRB = RB.dot(C)
        CRBa = CRB.getDenseArray()
        sumRe = np.sum(self.open_loop_eigvals.real - self.sigma) - \
                0.5*np.trace(1j*self.freqs[-1]*CRBa@K).real - integral

        grad_K -= 0.5*(1j*self.freqs[-1]*CRBa).conj().T
        Mat = K.conj().T
        Mat_ = PETSc.Mat().createDense(Mat.shape, None, Mat, MPI.COMM_SELF)
        grad_B_i.mult(0.5*np.conj(1j*self.freqs[-1]), 0.0, RHTC, Mat_)
        grad_p -= IOMats.compute_dBdp(p, grad_B_i, Bp1, Bp0, IOMats.compute_dB)
        Mat_.hermitianTranspose(None)
        grad_C_i.mult(0.5*1j*self.freqs[-1], 0.0, RB, Mat_)
        grad_s -= IOMats.compute_dBdp(s, grad_C_i, Cp1, Cp0, IOMats.compute_dC)

        exp_arg = self.alpha*np.abs(sumRe)
        if exp_arg < np.log(1e13):
            expval = np.exp(exp_arg)
            value = (2*sumRe + self.alpha*sumRe**2)*expval if sumRe > 0 else \
                (2*sumRe - self.alpha*sumRe**2)*expval
        else:
            value = 1e13
        
        grad_K *= value
        grad_s *= value
        grad_p *= value

        # scaling_exp = self.alpha*(sumRe**2 - self.eps)
        # if scaling_exp < np.log(1e13):
        #     scaling_value = 2*self.alpha*sumRe*np.exp(scaling_exp)
        #     grad_K *= scaling_value
        #     grad_s *= scaling_value
        #     grad_p *= scaling_value
        # else:
        #     grad_K *= 1e13
        #     grad_s *= 1e13
        #     grad_p *= 1e13

        objs = [B, C, Bp0, Bp1, Cp0, Cp1, grad_B_i, \
                grad_C_i, RB, RHTC, CRB, Mat_]
        for obj in objs: obj.destroy()
        return grad_p.real, grad_K.real, grad_s.real

class SensitivityCost:

    def __init__(self, comm, path_factors, factor_sizes, frequencies, weights):
        self.comm = comm
        self.freqs = frequencies
        self.weights = weights
        self.load_factors(path_factors, factor_sizes)

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
    
    def evaluate_cost(self, params, IOMats):
        p, K, s = params
        Id = np.eye(K.shape[-1])
        B = SLEPc.BV().create(comm=self.comm)
        B.setSizes(self.R_ops[0]._dimensions[0], K.shape[0])
        B.setType('mat')
        C = SLEPc.BV().create(comm=self.comm)
        C.setSizes(self.R_ops[0]._dimensions[0], K.shape[-1])
        C.setType('mat')
        B = IOMats.compute_B(p, B)
        C = IOMats.compute_C(s, C)
        RB = B.duplicate()
        J = 0
        for i in range (len(self.freqs)):
            RB = self.R_ops[i].apply_mat(B, RB)
            CRB = RB.dot(C)
            S = sp.linalg.inv(Id + CRB.getDenseArray()@K)
            _, sv, _ = sp.linalg.svd(S)
            J += self.weights[i]*sv[0]
            CRB.destroy()
        objs = [RB, CRB, B, C]
        for obj in objs: obj.destroy()
        return J

    def evaluate_gradient(self, params, IOMats):
        p, K, s = params
        Id = np.eye(K.shape[-1])
        B = SLEPc.BV().create(comm=self.comm)
        B.setSizes(self.R_ops[0]._dimensions[0], K.shape[0])
        B.setType('mat')
        B = IOMats.compute_B(p, B)
        grad_B_i = B.duplicate()
        Bp1 = B.duplicate()
        Bp0 = B.duplicate()
        RB = B.duplicate()

        C = SLEPc.BV().create(comm=self.comm)
        C.setSizes(self.R_ops[0]._dimensions[0], K.shape[-1])
        C.setType('mat')
        C = IOMats.compute_C(s, C)
        RHTC = C.duplicate()
        grad_C_i = C.duplicate()
        Cp1 = C.duplicate()
        Cp0 = C.duplicate()

        grad_K = np.zeros(K.shape, dtype=np.complex128)
        grad_p = np.zeros_like(p)
        grad_s = np.zeros_like(s)

        for i in range (len(self.freqs)):
            RHTC = self.R_ops[i].apply_hermitian_transpose_mat(C, RHTC)
            RB = self.R_ops[i].apply_mat(B, RB)
            CRB = RB.dot(C)
            CRBa = CRB.getDenseArray()
            S = sp.linalg.inv(Id + CRBa@K)
            _, sv, vh = sp.linalg.svd(S)
            v = vh.conj().T
            V = np.outer(v[:,0], v[:,0].conj())
            
            grad_K -= self.weights[i]*sv[0]*CRBa.conj().T@V@S.conj().T
            Mat = V@S.conj().T@K.conj().T
            Mat_ = PETSc.Mat().createDense(Mat.shape, None, Mat, MPI.COMM_SELF)
            grad_B_i.mult(self.weights[i]*sv[0], 0.0, RHTC, Mat_)
            grad_p -= IOMats.compute_dBdp(p, grad_B_i, Bp1, \
                                          Bp0, IOMats.compute_dB)
            Mat_.hermitianTranspose(None)
            grad_C_i.mult(self.weights[i]*sv[0], 0.0, RB, Mat_)
            grad_s -= IOMats.compute_dBdp(s, grad_C_i, Cp1, Cp0, \
                                          IOMats.compute_dC)
            
            Mat_.destroy()
            CRB.destroy()

        objs = [B, C, Bp0, Bp1, Cp0, Cp1, grad_B_i, grad_C_i, RB, RHTC, CRB]
        for obj in objs: obj.destroy()
        return grad_p.real, grad_K.real, grad_s.real

def create_objective_and_gradient_new(manifold, IOMats, WhichGrads, \
                                      H2Comp=None, StabComp=None):
    r"""
        Create functions to evaluate the cost function and the gradient

        :param manifold: one of the pymanopt manifolds
        :param opt_obj: instance of the optimization objects class

        :return: (cost function, gradient, hessian=:code:`None`)
        :rtype: (Callable, Callable, None)
    """

    if H2Comp == None and StabComp == None:
        raise ValueError (
            f"Provide at least one of H2Comp or StabComp should be provided"
        )

    euclidean_hessian = None

    @pymanopt.function.numpy(manifold)
    def cost(*params):
        JH2 = H2Comp.evaluate_cost(params, IOMats) if H2Comp != None else 0.0
        JS = StabComp.evaluate_cost(params, IOMats) if StabComp != None else 0.0
        return JH2 + JS
    
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):
        grad_pH2, grad_KH2, grad_sH2 = 0.0, 0.0, 0.0
        grad_pStb, grad_KStb, grad_sStb = 0.0, 0.0, 0.0
        if H2Comp != None:
            grad_pH2, grad_KH2, grad_sH2 = \
                H2Comp.evaluate_gradient(params, IOMats)
        if StabComp != None:
            grad_pStb, grad_KStb, grad_sStb = \
                StabComp.evaluate_gradient(params, IOMats)
        grad_p = grad_pH2 + grad_pStb
        grad_K = grad_KH2 + grad_KStb
        grad_s = grad_sH2 + grad_sStb
        return WhichGrads[0]*grad_p, WhichGrads[1]*grad_K, WhichGrads[2]*grad_s
    
    return cost, euclidean_gradient, euclidean_hessian
        
def test_euclidean_gradient(comm, cost, grad, params, eps):
    
    xa, K, xs = params
    rank = comm.Get_rank()
    J = cost(*params)
    grad_xa, grad_K, grad_xs = grad(*params)
    
    petscprint(comm, grad_xa)
    petscprint(comm, grad_K)
    petscprint(comm, grad_xs)
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
    dfd = (cost(*params_) - J)/eps

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
    dfd = (cost(*params_) - J)/eps

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
    dfd = (cost(*params_) - J)/eps

    if rank == 0:
        dgrad = (delta.conj().T@grad_xs).real
        error = np.abs(dfd - dgrad)
        percent_error = (error/np.abs(dfd)*100).real
        petscprint(comm,"------ Error xs grad -------------")
        petscprint(comm,"dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        petscprint(comm,"---------------------------------")
        
def generate_initial_gains(params, IOMats, H2Comp, K):
    r"""
        Generate initial guess for the feedback gains

        :param params: a 2-tuple (actuator parameters, sensor parameters)

    """
    comm = H2Comp.comm
    p, s = params

    B = SLEPc.BV().create(comm=comm)
    B.setSizes(H2Comp.R_ops[0]._dimensions[0], K.shape[0])
    B.setType('mat')
    C = SLEPc.BV().create(comm=comm)
    C.setSizes(H2Comp.R_ops[0]._dimensions[0], K.shape[-1])
    C.setType('mat')
    B = IOMats.compute_B(p, B)
    C = IOMats.compute_C(s, C)
    RB = B.duplicate()
    DDHTRB = B.duplicate()
    RHTC = C.duplicate()
    FFHTRHTC = C.duplicate()
    RFFHTRHTC = C.duplicate()

    # H2 component of the cost function
    M = np.zeros((len(K.reshape(-1)), len(K.reshape(-1))), dtype=np.float64)
    RHS = np.zeros(M.shape[0], dtype=np.float64)
    cc = 1 if np.sum(H2Comp.freqs < 0.0) == 0 else 0
    for i in range (len(H2Comp.freqs)):
        w = H2Comp.weights[i]
        R = H2Comp.R_ops[i]
        RB = R.apply_mat(B, RB)
        DDHTRB = IOMats.DDHT.apply_mat(RB, DDHTRB)
        Qi = DDHTRB.dot(RB)
        Qia = Qi.getDenseArray()

        RHTC = R.apply_hermitian_transpose_mat(C, RHTC)
        FFHTRHTC = IOMats.FFHT.apply_mat(RHTC, FFHTRHTC)
        Pi = FFHTRHTC.dot(RHTC)
        Pia = Pi.getDenseArray()
        Mi = np.kron(Qia, Pia.T)
        M += 2*w*Mi.real if cc == 1 else w*Mi

        RFFHTRHTC = R.apply_mat(FFHTRHTC, RFFHTRHTC)
        RHSm = RFFHTRHTC.dot(DDHTRB)
        RHSma = RHSm.getDenseArray()
        RHS += 2*w*RHSma.reshape(-1).real if cc == 1 else w*RHSma.reshape(-1)

        mats = [Pi, Qi, RHSm]
        for mat in mats: mat.destroy()
    mats = [B, C, RB, DDHTRB, RHTC, FFHTRHTC, RFFHTRHTC]
    for mat in mats: mat.destroy()
    u, s, v = sp.linalg.svd(M)
    v = v.conj().T
    idces = np.argwhere(s >= 1e-12).reshape(-1)
    Kvec = u[:,idces]@np.diag(1./s[idces])@(v[:,idces].conj().T@RHS)
    K[:,:] = Kvec.reshape(K.shape)
    return K

# -------------------------------------------------------------
# -------------------------------------------------------------
# ------------- Old stuff -------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

# class optimization_objects:

#     def __init__(self, comm, fnames_jacobian, jacobian_sizes, path_factors, \
#                 factor_sizes, fname_frequencies, fname_weights, \
#                 stability_params, compute_B, compute_C, compute_dB, compute_dC,\
#                 which_grad=None):
        
#         self.comm = comm
#         self.jacobian_sizes = jacobian_sizes
#         self.A = None
#         if stability_params != None:
#             A = read_coo_matrix(self.comm, fnames_jacobian, jacobian_sizes)
#             self.As = []
#             self.process_evals = []
#             for i in range (len(stability_params[4])):
#                 Id = PETSc.Mat().createConstantDiagonal(jacobian_sizes,1.0,comm)
#                 Id.scale(1j*stability_params[4][i])
#                 Id.convert(PETSc.Mat.Type.MPIAIJ)
#                 Id.axpy(-1.0, A)
#                 ksp = create_mumps_solver(self.comm, Id)
#                 self.As.append(MatrixLinearOperator(self.comm, Id, ksp))
#                 self.process_evals.append(lambda x: \
#                                           1j*stability_params[4][i] - 1./x)

#         self.freqs = np.load(fname_frequencies)
#         self.weights = np.load(fname_weights)
#         self.load_factors(path_factors, factor_sizes)
#         self.compute_B = compute_B
#         self.compute_C = compute_C
#         self.compute_dB = compute_dB
#         self.compute_dC = compute_dC
#         self.stab_params = stability_params
#         self.which_grad = which_grad
    
#     def load_factors(self, path_factors, factor_sizes):
        
#         self.U, self.S, self.V, self.R_ops = [], [], [], []
#         for i in range (len(self.freqs)):
#             fname_U = path_factors + 'omega_%1.5f/U.dat'%self.freqs[i]
#             fname_S = path_factors + 'omega_%1.5f/S.npy'%self.freqs[i]
#             fname_V = path_factors + 'omega_%1.5f/V.dat'%self.freqs[i]

#             self.U.append(read_bv(self.comm, fname_U, factor_sizes))
#             self.V.append(read_bv(self.comm, fname_V, factor_sizes))
#             self.S.append(np.load(fname_S))
#             self.R_ops.append(\
#                 LowRankLinearOperator(self.comm, self.U[-1], \
#                                       self.S[-1], self.V[-1]))

#     def evaluate_exponential(self, val):
#         alpha, beta = self.stab_params[:2]
#         try:    value = np.exp(alpha*(val.real + beta)).real
#         except: value = 1e12
#         value = 0.0 if value <= 1e-13 else value
#         return value
    
#     def evaluate_gradient_exponential(self, val):
#         alpha = self.stab_params[0]
#         return alpha*self.evaluate_exponential(val)
    
#     def compute_dBdp(self, p, grad_B, Bp1, Bp0, compute_dB):
#         grad_p = np.zeros_like(p)
#         for j in range (len(p)):
#             pjp = p.copy()
#             pjm = p.copy()
#             pjp[j] += 1e-7
#             pjm[j] -= 1e-7
#             Bp1 = compute_dB(pjp, pjm, Bp1, Bp0)
#             Bp1.scale(1./2e-7)
#             bv_conj(Bp1)
#             grad_p[j] = _compute_bv_contraction(self.comm, Bp1, grad_B).real
#         return grad_p

# def cost_profile(params, opt_obj):
#     r"""
#         Evaluate the cost function

#         :param params: a 3-tuple (actuator parameters, feedback gains, 
#             sensor parameters)

#         :rtype: float
#     """
#     comm = opt_obj.comm
#     p, K, s = params

#     t0 = tlib.time()

#     B = SLEPc.BV().create(comm=comm)
#     B.setSizes(opt_obj.R_ops[0]._dimensions[0], K.shape[0])
#     B.setType('mat')
#     C = SLEPc.BV().create(comm=comm)
#     C.setSizes(opt_obj.R_ops[0]._dimensions[0], K.shape[-1])
#     C.setType('mat')
#     B = opt_obj.compute_B(p, B)
#     C = opt_obj.compute_C(s, C)
#     RB = B.duplicate()
#     RHTC = C.duplicate()

#     # H2 component of the cost function
#     J = 0
#     for i in range (len(opt_obj.freqs)):
#         w = opt_obj.weights[i]
#         R = opt_obj.R_ops[i]
#         M, _ = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)
        
#         J += w*np.sum(np.diag(R.Sigma)**2)
#         J -= 2.0*w*_compute_trace_product(R, M, True)
#         J += w*_compute_trace_product(M, M, True)
        
#         # process = psutil.Process(os.getpid())
#         # value = process.memory_info().rss/(1024 * 1024)
#         # value = sum(comm.allgather(value))
#         # if opt_obj.comm.Get_rank() == 0:
#         #     print(f"Iteration {i} usage {value} MB")
    
#     # Stability-promoting component of the cost function
#     if opt_obj.stab_params != None:
#         for i in range (len(opt_obj.stab_params[4])):
#             cl_op = LowRankUpdatedLinearOperator(comm, opt_obj.As[i], B, K, C)
#             krylov_dim, n_evals = opt_obj.stab_params[2], opt_obj.stab_params[3]
#             D, E = eig(cl_op, cl_op.solve, krylov_dim, n_evals, \
#                     opt_obj.process_evals)
#             D = np.diag(D)
#             J += np.sum([opt_obj.evaluate_exponential(val) for val in D])
#             E.destroy()
#             cl_op.destroy_woodbury_operator()
#             cl_op.destroy_intermediate_vectors()
            
#     bvs = [B, C, RB, RHTC]
#     for bv in bvs: bv.destroy()

#     t1 = tlib.time()
#     petscprint(opt_obj.comm, "Execution time = %1.5f [sec]"%(t1 - t0))
#     return J.real

# def euclidean_gradient_profile(params, opt_obj):
#     r"""
#         Evaluate the gradient of the cost function

#         :param params: a 3-tuple (actuator parameters p, feedback gains K, 
#             sensor parameters s)

#         :return: (:math:`\nabla_p J`, :math:`\nabla_K J`, 
#             :math:`\nabla_s J`)
#     """
#     comm = opt_obj.comm
#     p, K, s = params

#     t0 = tlib.time()

#     B = SLEPc.BV().create(comm=comm)
#     B.setSizes(opt_obj.R_ops[0]._dimensions[0], K.shape[0])
#     B.setType('mat')
#     Bp1 = B.copy()
#     Bp0 = B.copy()
#     C = SLEPc.BV().create(comm=comm)
#     C.setSizes(opt_obj.R_ops[0]._dimensions[0], K.shape[-1])
#     C.setType('mat')
#     Cp1 = C.copy()
#     Cp0 = C.copy()
#     B = opt_obj.compute_B(p, B)
#     C = opt_obj.compute_C(s, C)
#     RB = B.duplicate()
#     RHTC = C.duplicate()
    
#     grad_K = np.zeros(K.shape, dtype=np.complex128)
#     grad_p = np.zeros_like(p)
#     grad_s = np.zeros_like(s)

#     # Allocate memory for efficiency (grad_K)
#     F1K = SLEPc.BV().create(comm=opt_obj.comm)
#     F1K.setSizes(opt_obj.R_ops[0]._dimensions[0], C.getSizes()[-1])
#     F1K.setType('mat')
#     F2K = F1K.duplicate()
#     F3K = F1K.duplicate()

#     # Allocate memory for efficiency (grad_p)
#     F1B = SLEPc.BV().create(comm=opt_obj.comm)
#     F1B.setSizes(opt_obj.R_ops[0]._dimensions[0], B.getSizes()[-1])
#     F1B.setType('mat')
#     F2B = F1B.duplicate()
#     F3B = F1B.duplicate()
#     grad_B_i = F1B.duplicate()
#     Q_grad_B_i = F1B.duplicate()

#     # Allocate memory for efficiency (grad_s)
#     F1C = SLEPc.BV().create(comm=opt_obj.comm)
#     F1C.setSizes(opt_obj.R_ops[0]._dimensions[-1], C.getSizes()[-1])
#     F1C.setType('mat')
#     F2C = F1C.duplicate()
#     F3C = F1C.duplicate()
#     grad_C_i = F1C.duplicate()
#     Q_grad_C_i = F1C.duplicate()

#     for i in range (len(opt_obj.freqs)):
        
#         w = opt_obj.weights[i]
#         R = opt_obj.R_ops[i]
#         M, Linv = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)

#         # Grad with respect to K
#         LiCT = Linv.conj().T
#         LiCTMat = PETSc.Mat().createDense(LiCT.shape, None, LiCT, MPI.COMM_SELF)
#         F1K.mult(1.0, 0.0, M.V, LiCTMat)
#         F2K = R.apply_mat(F1K, F2K)
#         F3K = M.apply_mat(F1K, F3K)
#         bv_add(-1.0, F3K, F2K)
#         grad_K_i_mat = F3K.dot(M.U)
#         F4K = C.dot(M.U)
#         grad_K_i = grad_K_i_mat.getDenseArray()
#         grad_K_i -= F4K.getDenseArray()@M.Sigma.conj().T@grad_K_i
#         grad_K += 2.0*w*grad_K_i
#         mats = [LiCTMat, grad_K_i_mat, F4K]
#         for mat in mats: mat.destroy()

#         # Grad with respect to p
#         S = M.Sigma.conj().T
#         SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
#         F1B.mult(1.0, 0.0, M.V, SMat)
#         F2B = R.apply_mat(F1B, F2B)
#         F3B = M.apply_mat(F1B, F3B)
#         bv_add(-1.0, F3B, F2B)
#         grad_B_i = R.apply_hermitian_transpose_mat(F3B, grad_B_i)
#         Q = LowRankLinearOperator(comm, M.V, M.Sigma.conj().T, B)
#         Q_grad_B_i = Q.apply_mat(grad_B_i, Q_grad_B_i)
#         bv_add(-1.0, grad_B_i, Q_grad_B_i)
#         grad_p += 2.0*w*opt_obj.compute_dBdp(p, grad_B_i, Bp1, Bp0, \
#                                              opt_obj.compute_dB)
#         SMat.destroy()

#         # Grad with respect to s
#         S = M.Sigma.copy()
#         SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
#         F1C.mult(1.0, 0.0, M.U, SMat)
#         F2C = R.apply_hermitian_transpose_mat(F1C, F2C)
#         F3C = M.apply_hermitian_transpose_mat(F1C, F3C)
#         bv_add(-1.0, F3C, F2C)
#         grad_C_i = R.apply_mat(F3C, grad_C_i)
#         Q = LowRankLinearOperator(comm, M.U, M.Sigma, C)
#         Q_grad_C_i = Q.apply_mat(grad_C_i, Q_grad_C_i)
#         bv_add(-1.0, grad_C_i, Q_grad_C_i)
#         grad_s += 2.0*w*opt_obj.compute_dBdp(s, grad_C_i, Cp1, Cp0, \
#                                              opt_obj.compute_dC)
#         SMat.destroy()

#         # process = psutil.Process(os.getpid())
#         # value = process.memory_info().rss/(1024 * 1024)
#         # value = sum(comm.allgather(value))
#         # if opt_obj.comm.Get_rank() == 0:
#         #     print(f"Iteration {i} usage {value} MB")

#     # Stability-promoting penalty
#     if opt_obj.stab_params != None:
#         for i in range (len(opt_obj.stab_params[4])):
#             cl_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.As[i], \
#                                                  B, K, C)
#             kry_dim, n_evals = opt_obj.stab_params[2], opt_obj.stab_params[3]
#             V, D, W = right_and_left_eig(cl_op, cl_op.solve, kry_dim, n_evals, \
#                                                         opt_obj.process_evals)
#             cl_op.destroy_woodbury_operator()
#             cl_op.destroy_intermediate_vectors()

#             D = np.diag(D)
#             MD = np.diag([opt_obj.evaluate_gradient_exponential(v) for v in D])
#             BHTW = W.dot(B)
#             VHTC = C.dot(V)

#             # Grad with respect to K
#             grad_K -= BHTW.getDenseArray()@MD@VHTC.getDenseArray()
#             # Grad with respect to p
#             M_data = MD@VHTC.getDenseArray()@K.conj().T
#             M = PETSc.Mat().createDense(M_data.shape, None, M_data, \
#                                         MPI.COMM_SELF)
#             grad_B_i.mult(1.0, 0.0, W, M)
#             grad_p -= opt_obj.compute_dBdp(p, grad_B_i, Bp1, Bp0,\
#                                         opt_obj.compute_dB)
#             M.destroy()
#             # Grad with respect to s
#             M_data = MD@BHTW.getDenseArray().conj().T@K
#             M = PETSc.Mat().createDense(M_data.shape, None, M_data, \
#                                         MPI.COMM_SELF)
#             grad_C_i.mult(1.0, 0.0, V, M)
#             grad_s -= opt_obj.compute_dBdp(s, grad_C_i, Cp1, Cp0,\
#                                         opt_obj.compute_dC)
#             M.destroy()

#         objects = [V, W, BHTW, VHTC]
#         for obj in objects: obj.destroy()

#     objects = [F1K, F2K, F3K,F1B, F2B, F3B, F1C, F2C, F3C, grad_B_i, \
#                Q_grad_B_i, grad_C_i, Q_grad_C_i, Bp1, Bp0, Cp1, Cp0, B, C]
#     for obj in objects: obj.destroy()

#     t1 = tlib.time()
#     petscprint(opt_obj.comm, "Execution time = %1.5f [sec]"%(t1 - t0))
    
#     return grad_p, grad_K.real, grad_s

# def create_objective_and_gradient(manifold,opt_obj):
#     r"""
#         Create functions to evaluate the cost function and the gradient

#         :param manifold: one of the pymanopt manifolds
#         :param opt_obj: instance of the optimization objects class

#         :return: (cost function, gradient, hessian=:code:`None`)
#         :rtype: (Callable, Callable, None)
#     """
    
#     euclidean_hessian = None

#     @pymanopt.function.numpy(manifold)
#     def cost(*params):
#         r"""
#             Evaluate the cost function

#             :param params: a 3-tuple (actuator parameters, feedback gains, 
#                 sensor parameters)

#             :rtype: float
#         """
#         comm = opt_obj.comm
#         p, K, s = params
#         idces = np.argwhere(p < 0.0)
#         p[idces] = 1e-5
#         idces = np.argwhere(s < 0.0)
#         s[idces] = 1e-5
        
#         # petscprint(comm, "Params cost ---------")
#         # petscprint(comm, p)
#         # petscprint(comm, K)
#         # petscprint(comm, s)
#         # petscprint(comm, "---------- ---------")

#         t0 = tlib.time()

#         B = SLEPc.BV().create(comm=comm)
#         B.setSizes(opt_obj.R_ops[0]._dimensions[0], K.shape[0])
#         B.setType('mat')
#         C = SLEPc.BV().create(comm=comm)
#         C.setSizes(opt_obj.R_ops[0]._dimensions[0], K.shape[-1])
#         C.setType('mat')
#         B = opt_obj.compute_B(p, B)
#         C = opt_obj.compute_C(s, C)
#         RB = B.duplicate()
#         RHTC = C.duplicate()

#         # H2 component of the cost function
#         J = 0
#         for i in range (len(opt_obj.freqs)):
#             w = opt_obj.weights[i]
#             R = opt_obj.R_ops[i]
#             M, _ = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)

#             J += w*np.sum(np.diag(R.Sigma)**2)
#             J -= 2.0*w*_compute_trace_product(R, M, True)
#             J += w*_compute_trace_product(M, M, True)
        
#         # Stability-promoting component of the cost function
#         if opt_obj.stab_params != None:
#             for i in range (len(opt_obj.stab_params[4])):
#                 cl_op = LowRankUpdatedLinearOperator(comm, opt_obj.As[i], \
#                                                      B, K, C)
#                 krylov_dim, n_evals = opt_obj.stab_params[2], \
#                     opt_obj.stab_params[3]
#                 D, E = eig(cl_op, cl_op.solve, krylov_dim, n_evals, \
#                         opt_obj.process_evals[i])
#                 D = np.diag(D)
#                 J += np.sum([opt_obj.evaluate_exponential(val) for val in D])
#                 E.destroy()
#                 cl_op.destroy_woodbury_operator()
#                 cl_op.destroy_intermediate_vectors()
                
#         bvs = [B, C, RB, RHTC]
#         for bv in bvs: bv.destroy()

#         t1 = tlib.time()
#         petscprint(opt_obj.comm, "Cost exec time = %1.5f [sec]"%(t1 - t0))
#         return J.real

#     @pymanopt.function.numpy(manifold)
#     def euclidean_gradient(*params):
#         r"""
#             Evaluate the gradient of the cost function

#             :param params: a 3-tuple (actuator parameters p, feedback gains K, 
#                 sensor parameters s)

#             :return: (:math:`\nabla_p J`, :math:`\nabla_K J`, 
#                 :math:`\nabla_s J`)
#         """
#         comm = opt_obj.comm
#         p, K, s = params
#         idces = np.argwhere(p < 0.0)
#         p[idces] = 1e-5
#         idces = np.argwhere(s < 0.0)
#         s[idces] = 1e-5
#         # petscprint(comm, "Params grad ---------")
#         # petscprint(comm, p)
#         # petscprint(comm, K)
#         # petscprint(comm, s)
#         # petscprint(comm, "---------- ---------")

#         t0 = tlib.time()

#         B = SLEPc.BV().create(comm=comm)
#         B.setSizes(opt_obj.R_ops[0]._dimensions[0], K.shape[0])
#         B.setType('mat')
#         Bp1 = B.copy()
#         Bp0 = B.copy()
#         C = SLEPc.BV().create(comm=comm)
#         C.setSizes(opt_obj.R_ops[0]._dimensions[0], K.shape[-1])
#         C.setType('mat')
#         Cp1 = C.copy()
#         Cp0 = C.copy()
#         B = opt_obj.compute_B(p, B)
#         C = opt_obj.compute_C(s, C)
#         RB = B.duplicate()
#         RHTC = C.duplicate()
        
#         grad_K = np.zeros(K.shape, dtype=np.complex128)
#         grad_p = np.zeros_like(p)
#         grad_s = np.zeros_like(s)

#         # Allocate memory for efficiency (grad_K)
#         F1K = SLEPc.BV().create(comm=opt_obj.comm)
#         F1K.setSizes(opt_obj.R_ops[0]._dimensions[0], C.getSizes()[-1])
#         F1K.setType('mat')
#         F2K = F1K.duplicate()
#         F3K = F1K.duplicate()

#         # Allocate memory for efficiency (grad_p)
#         F1B = SLEPc.BV().create(comm=opt_obj.comm)
#         F1B.setSizes(opt_obj.R_ops[0]._dimensions[0], B.getSizes()[-1])
#         F1B.setType('mat')
#         F2B = F1B.duplicate()
#         F3B = F1B.duplicate()
#         grad_B_i = F1B.duplicate()
#         Q_grad_B_i = F1B.duplicate()

#         # Allocate memory for efficiency (grad_s)
#         F1C = SLEPc.BV().create(comm=opt_obj.comm)
#         F1C.setSizes(opt_obj.R_ops[0]._dimensions[-1], C.getSizes()[-1])
#         F1C.setType('mat')
#         F2C = F1C.duplicate()
#         F3C = F1C.duplicate()
#         grad_C_i = F1C.duplicate()
#         Q_grad_C_i = F1C.duplicate()

#         for i in range (len(opt_obj.freqs)):            
#             w = opt_obj.weights[i]
#             R = opt_obj.R_ops[i]
#             M, Linv = _compute_woodbury_update(comm, R, B, K, C, RB, RHTC)

#             # Grad with respect to K
#             LiCT = Linv.conj().T
#             LiCTMat = PETSc.Mat().createDense(LiCT.shape, None, LiCT, \
#                                               MPI.COMM_SELF)
#             F1K.mult(1.0, 0.0, M.V, LiCTMat)
#             F2K = R.apply_mat(F1K, F2K)
#             F3K = M.apply_mat(F1K, F3K)
#             bv_add(-1.0, F3K, F2K)
#             grad_K_i_mat = F3K.dot(M.U)
#             F4K = C.dot(M.U)
#             grad_K_i = grad_K_i_mat.getDenseArray()
#             grad_K_i -= F4K.getDenseArray()@M.Sigma.conj().T@grad_K_i
#             grad_K += 2.0*w*grad_K_i
#             mats = [LiCTMat, grad_K_i_mat, F4K]
#             for mat in mats: mat.destroy()

#             # Grad with respect to p
#             S = M.Sigma.conj().T
#             SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
#             F1B.mult(1.0, 0.0, M.V, SMat)
#             F2B = R.apply_mat(F1B, F2B)
#             F3B = M.apply_mat(F1B, F3B)
#             bv_add(-1.0, F3B, F2B)
#             grad_B_i = R.apply_hermitian_transpose_mat(F3B, grad_B_i)
#             Q = LowRankLinearOperator(comm, M.V, M.Sigma.conj().T, B)
#             Q_grad_B_i = Q.apply_mat(grad_B_i, Q_grad_B_i)
#             bv_add(-1.0, grad_B_i, Q_grad_B_i)
#             grad_p += 2.0*w*opt_obj.compute_dBdp(p, grad_B_i, Bp1, Bp0, \
#                                                 opt_obj.compute_dB)
#             SMat.destroy()

#             # Grad with respect to s
#             S = M.Sigma.copy()
#             SMat = PETSc.Mat().createDense(S.shape, None, S, MPI.COMM_SELF)
#             F1C.mult(1.0, 0.0, M.U, SMat)
#             F2C = R.apply_hermitian_transpose_mat(F1C, F2C)
#             F3C = M.apply_hermitian_transpose_mat(F1C, F3C)
#             bv_add(-1.0, F3C, F2C)
#             grad_C_i = R.apply_mat(F3C, grad_C_i)
#             Q = LowRankLinearOperator(comm, M.U, M.Sigma, C)
#             Q_grad_C_i = Q.apply_mat(grad_C_i, Q_grad_C_i)
#             bv_add(-1.0, grad_C_i, Q_grad_C_i)
#             grad_s += 2.0*w*opt_obj.compute_dBdp(s, grad_C_i, Cp1, Cp0, \
#                                                 opt_obj.compute_dC)
#             SMat.destroy()

#         # Stability-promoting penalty
#         if opt_obj.stab_params != None:
#             for i in range (len(opt_obj.stab_params[4])):
#                 cl_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.As[i], \
#                                                     B, K, C)
#                 kry_dim, n_evals = opt_obj.stab_params[2],opt_obj.stab_params[3]
#                 V, D, W = right_and_left_eig(cl_op, cl_op.solve, kry_dim, \
#                                              n_evals, opt_obj.process_evals[i])
#                 cl_op.destroy_woodbury_operator()
#                 cl_op.destroy_intermediate_vectors()
#                 D = np.diag(D)
#                 MD = np.diag([opt_obj.evaluate_gradient_exponential(v) \
#                               for v in D])
#                 BHTW = W.dot(B)
#                 VHTC = C.dot(V)

#                 # Grad with respect to K
#                 grad_K -= BHTW.getDenseArray()@MD@VHTC.getDenseArray()
#                 # Grad with respect to p
#                 M_data = MD@VHTC.getDenseArray()@K.conj().T
#                 M = PETSc.Mat().createDense(M_data.shape, None, M_data, \
#                                             MPI.COMM_SELF)
#                 grad_B_i.mult(1.0, 0.0, W, M)
#                 grad_p -= opt_obj.compute_dBdp(p, grad_B_i, Bp1, Bp0,\
#                                             opt_obj.compute_dB)
#                 M.destroy()
#                 # Grad with respect to s
#                 M_data = MD@BHTW.getDenseArray().conj().T@K
#                 M = PETSc.Mat().createDense(M_data.shape, None, M_data, \
#                                             MPI.COMM_SELF)
#                 grad_C_i.mult(1.0, 0.0, V, M)
#                 grad_s -= opt_obj.compute_dBdp(s, grad_C_i, Cp1, Cp0,\
#                                             opt_obj.compute_dC)
#                 M.destroy()

#             objects = [V, W, BHTW, VHTC]
#             for obj in objects: obj.destroy()

#         objects = [F1K, F2K, F3K,F1B, F2B, F3B, F1C, F2C, F3C, grad_B_i, \
#                 Q_grad_B_i, grad_C_i, Q_grad_C_i, Bp1, Bp0, Cp1, Cp0, B, C]
#         for obj in objects: obj.destroy()

#         t1 = tlib.time()
#         petscprint(opt_obj.comm, "Grad exec time = %1.5f [sec]"%(t1 - t0))
        
#         grad_p *= opt_obj.which_grad[0]
#         grad_K *= opt_obj.which_grad[1]
#         grad_s *= opt_obj.which_grad[2]
        
#         return grad_p, grad_K.real, grad_s
    
#     return cost, euclidean_gradient, euclidean_hessian



# -------------------------------------------------
# ------- Other old stuff that does work ----------
# -------------------------------------------------

# class FOMStabilityComponent:

#     def __init__(self, comm, fnames_jacobian, jacobian_sizes, alpha, beta, \
#                  krylov_dim, n_evals, shifts):
#         self.comm = comm
#         self.alpha = alpha
#         self.beta = beta
#         self.krylov_dim = krylov_dim
#         self.n_evals = n_evals
#         A = read_coo_matrix(self.comm, fnames_jacobian, jacobian_sizes)
#         self.As = []
#         self.process_evals = []
#         for i in range (len(shifts)):
#             s = shifts[i]
#             Id = PETSc.Mat().createConstantDiagonal(jacobian_sizes,1.0,comm)
#             Id.scale(1j*s)
#             Id.convert(PETSc.Mat.Type.MPIAIJ)
#             Id.axpy(-1.0, A)
#             ksp = create_mumps_solver(self.comm, Id)
#             self.As.append(MatrixLinearOperator(self.comm, Id, ksp))
#             self.process_evals.append(lambda x: 1j*s - 1./x)

#     def evaluate_exponential(self, val):
#         try:    value = np.exp(self.alpha*(val.real + self.beta)).real
#         except: value = 1e12
#         value = 0.0 if value <= 1e-13 else value
#         return value
    
#     def evaluate_grad_exponential(self, val):
#         return self.alpha*self.evaluate_exponential(val)

#     def evaluate_cost(self, params, IOMats):
#         p, K, s = params
#         B = SLEPc.BV().create(comm=self.comm)
#         B.setSizes(self.As[0]._dimensions[0], K.shape[0])
#         B.setType('mat')
#         C = SLEPc.BV().create(comm=self.comm)
#         C.setSizes(self.As[0]._dimensions[0], K.shape[-1])
#         C.setType('mat')
#         B = IOMats.compute_B(p, B)
#         C = IOMats.compute_C(s, C) 
#         J = 0
#         for i in range (len(self.As)):
#             linop = LowRankUpdatedLinearOperator(self.comm, self.As[i], B, K, C)
#             D, E = eig(linop, linop.solve, self.krylov_dim, self.n_evals, \
#                        self.process_evals[i])
#             J += np.sum([self.evaluate_exponential(val) for val in np.diag(D)])
#             E.destroy()
#             linop.destroy_woodbury_operator()
#             linop.destroy_intermediate_vectors()
#         B.destroy()
#         C.destroy()
#         return J.real
    
#     def evaluate_gradient(self, params, IOMats):
#         p, K, s = params
#         B = SLEPc.BV().create(comm=self.comm)
#         B.setSizes(self.As[0]._dimensions[0], K.shape[0])
#         B.setType('mat')
#         Bp1 = B.copy()
#         Bp0 = B.copy()
#         C = SLEPc.BV().create(comm=self.comm)
#         C.setSizes(self.As[0]._dimensions[0], K.shape[-1])
#         C.setType('mat')
#         Cp1 = C.copy()
#         Cp0 = C.copy()
#         B = IOMats.compute_B(p, B)
#         C = IOMats.compute_C(s, C) 
#         grad_K = np.zeros(K.shape, dtype=np.complex128)
#         grad_p = np.zeros_like(p)
#         grad_s = np.zeros_like(s)
#         # Allocate memory for efficiency (grad_p)
#         grad_B = SLEPc.BV().create(comm=self.comm)
#         grad_B.setSizes(self.As[0]._dimensions[0], B.getSizes()[-1])
#         grad_B.setType('mat')
#         grad_B.scale(0.0)
#         # Allocate memory for efficiency (grad_s)
#         grad_C = SLEPc.BV().create(comm=self.comm)
#         grad_C.setSizes(self.As[0]._dimensions[-1], C.getSizes()[-1])
#         grad_C.setType('mat')
#         grad_C.scale(0.0)
#         for i in range (len(self.As)):
#             linop = LowRankUpdatedLinearOperator(self.comm, self.As[i], B, K, C)
#             V, D, W = right_and_left_eig(linop, linop.solve, self.krylov_dim, \
#                                          self.n_evals, self.process_evals[i])
#             linop.destroy_woodbury_operator()
#             linop.destroy_intermediate_vectors()
#             D = np.diag(D)
#             MD = np.diag([self.evaluate_grad_exponential(v) for v in D])
#             BHTW = W.dot(B)
#             VHTC = C.dot(V)
#             # Grad with respect to K
#             grad_K -= BHTW.getDenseArray()@MD@VHTC.getDenseArray()
#             # Grad with respect to p
#             Ma = MD@VHTC.getDenseArray()@K.conj().T
#             M = PETSc.Mat().createDense(Ma.shape, None, Ma, MPI.COMM_SELF)
#             grad_B.mult(1.0, 0.0, W, M)
#             grad_p -= IOMats.compute_dBdp(p, grad_B, Bp1, Bp0, \
#                                           IOMats.compute_dB)
#             M.destroy()
#             # Grad with respect to s
#             Ma = MD@BHTW.getDenseArray().conj().T@K
#             M = PETSc.Mat().createDense(Ma.shape, None, Ma, MPI.COMM_SELF)
#             grad_C.mult(1.0, 0.0, V, M)
#             grad_s -= IOMats.compute_dBdp(s, grad_C, Cp1, Cp0, \
#                                           IOMats.compute_dC)
#             M.destroy()
#             objects = [V, W, BHTW, VHTC]
#             for obj in objects: obj.destroy()

#         objects = [B, C, Bp0, Bp1, Cp0, Cp1, grad_B, grad_C]
#         for obj in objects: obj.destroy()
#         return grad_p, grad_K.real, grad_s

# class ROMStabilityComponent:

#     def __init__(self, comm, Ar, Phi, Psi, alpha, beta):
#         self.comm = comm
#         self.Ar = Ar
#         self.Phi = Phi
#         self.Psi = Psi
#         self.alpha = alpha
#         self.beta = beta

#     def evaluate_exponential(self, val):
#         try:    value = np.exp(self.alpha*(val.real + self.beta)).real
#         except: value = 1e12
#         value = 0.0 if value <= 1e-13 else value
#         return value
    
#     def evaluate_grad_exponential(self, val):
#         return self.alpha*self.evaluate_exponential(val)

#     def compute_closed_loop_tensor(self, B, K, C):
#         PsiHTB = B.dot(self.Psi)
#         CHTPhi = self.Phi.dot(C)
#         Arcl = self.Ar - PsiHTB.getDenseArray()@K@CHTPhi.getDenseArray()
#         PsiHTB.destroy()
#         CHTPhi.destroy()
#         return Arcl

#     def evaluate_cost(self, params, IOMats):
#         p, K, s = params
#         B = SLEPc.BV().create(comm=self.comm)
#         B.setSizes(self.Phi.getSizes()[0], K.shape[0])
#         B.setType('mat')
#         C = SLEPc.BV().create(comm=self.comm)
#         C.setSizes(self.Phi.getSizes()[0], K.shape[-1])
#         C.setType('mat')
#         B = IOMats.compute_B(p, B)
#         C = IOMats.compute_C(s, C) 
#         Arcl = self.compute_closed_loop_tensor(B, K, C)
#         D, _ = sp.linalg.eig(Arcl)
#         J = np.sum([self.evaluate_exponential(val) for val in D])
#         B.destroy()
#         C.destroy()
#         return J.real

#     def evaluate_gradient(self, params, IOMats):
#         p, K, s = params
#         B = SLEPc.BV().create(comm=self.comm)
#         B.setSizes(self.Phi.getSizes()[0], K.shape[0])
#         B.setType('mat')
#         grad_B = B.copy()
#         Bp1 = B.copy()
#         Bp0 = B.copy()
#         C = SLEPc.BV().create(comm=self.comm)
#         C.setSizes(self.Phi.getSizes()[0], K.shape[-1])
#         C.setType('mat')
#         grad_C = C.copy()
#         Cp1 = C.copy()
#         Cp0 = C.copy()
#         B = IOMats.compute_B(p, B)
#         C = IOMats.compute_C(s, C) 
#         grad_K = np.zeros(K.shape, dtype=np.complex128)
#         grad_p = np.zeros_like(p)
#         grad_s = np.zeros_like(s)
#         Arcl = self.compute_closed_loop_tensor(B, K, C)
#         D, V = sp.linalg.eig(Arcl)
#         W = sp.linalg.inv(V).conj().T
#         MD = np.diag([self.evaluate_grad_exponential(v) for v in D])
#         M = W@MD@V.conj().T
#         PhiHTC = C.dot(self.Phi)
#         BHTPsi = self.Psi.dot(B)
#         # Grad with respect to p
#         Mat = M@PhiHTC.getDenseArray()@K.conj().T
#         Matm = PETSc.Mat().createDense(Mat.shape, None, Mat, MPI.COMM_SELF)
#         grad_B.mult(1.0, 0.0, self.Psi, Matm)
#         grad_p = -IOMats.compute_dBdp(p, grad_B, Bp1, Bp0, IOMats.compute_dB)
#         Matm.destroy()
#         # Grad with respect to s
#         Mat = M.conj().T@BHTPsi.getDenseArray().conj().T@K
#         Matm = PETSc.Mat().createDense(Mat.shape, None, Mat, MPI.COMM_SELF)
#         grad_C.mult(1.0, 0.0, self.Phi, Matm)
#         grad_s = -IOMats.compute_dBdp(s, grad_C, Cp1, Cp0, IOMats.compute_dC)
#         Matm.destroy()
#         # Grad with respect to K
#         grad_K = -BHTPsi.getDenseArray()@M@PhiHTC.getDenseArray()
#         objs = [B, C, Bp0, Bp1, Cp0, Cp1, grad_B, grad_C, BHTPsi, PhiHTC]
#         for obj in objs: obj.destroy()
#         return grad_p, grad_K.real, grad_s

