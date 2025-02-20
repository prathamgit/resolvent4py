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
from ..applications import right_and_left_eig
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



class H2Component:

    def __init__(self, comm, path_SVD, SVD_sizes, frequencies, weights, \
                 RBinterps, RHTCinterps, CHTRBinterps, D, F):
        
        self.comm = comm
        self.freqs = frequencies
        self.weights = weights
        self.load_SVD(path_SVD, SVD_sizes)
        self.RBinterps = RBinterps
        self.RHTCinterps = RHTCinterps
        self.CHTRBinterps = CHTRBinterps
        D_actions = [D.apply_hermitian_transpose, D.apply]
        F_actions = [F.apply_hermitian_transpose, F.apply]
        self.DDHT = ProductLinearOperator(comm, [D, D], D_actions)
        self.FFHT = ProductLinearOperator(comm, [F, F], F_actions)

    def load_SVD(self, path_factors, factor_sizes):
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
            
    def evaluate_cost(self, params):
        p, K, s = params
        na, ns = K.shape
        RB = SLEPc.BV().create(comm=self.comm)
        RB.setSizes(self.R_ops[0]._dimensions[0], K.shape[0])
        RB.setType('mat')
        RHTC = SLEPc.BV().create(comm=self.comm)
        RHTC.setSizes(self.R_ops[0]._dimensions[0], K.shape[-1])
        RHTC.setType('mat')
        CHTRB = np.zeros((K.shape[-1], K.shape[0]), dtype=np.complex128)
        Id = np.eye(K.shape[-1])
        scals_p = np.ones(na)
        scals_s = np.ones(ns)
        scals_ps = np.ones((ns, na))
        J = 0
        for i in range (len(self.freqs)):
            w = self.weights[i]
            R = self.R_ops[i]
            RB = self.RBinterps[i][0](p, RB, scals_p)
            RHTC = self.RHTCinterps[i][0](s, RHTC, scals_s)
            CHTRB = self.CHTRBinterps[i][0](p, s, CHTRB, scals_ps)
            S = sp.linalg.inv(Id + CHTRB@K)
            M = LowRankLinearOperator(self.comm, RB, K@S, RHTC)
            J += w*_compute_trace_product_new(R, R, self.DDHT, self.FFHT)
            J -= 2*w*_compute_trace_product_new(R, M, self.DDHT, self.FFHT)
            J += w*_compute_trace_product_new(M, M, self.DDHT, self.FFHT)

        objs = [RB, RHTC]
        for obj in objs: obj.destroy()
        return J.real
    
    def evaluate_gradient(self, params):

        commself = MPI.COMM_SELF
        comm = self.comm
        p, K, s = params
        na, ns = K.shape
        num_params_actu = len(p)//na
        num_params_sens = len(s)//ns

        RB = SLEPc.BV().create(comm=self.comm)
        RB.setSizes(self.R_ops[0]._dimensions[0], na)
        RB.setType('mat')
        RHTC = SLEPc.BV().create(comm=self.comm)
        RHTC.setSizes(self.R_ops[0]._dimensions[0], ns)
        RHTC.setType('mat')
        CHTRB = np.zeros((ns, na), dtype=np.complex128)
        Id = np.eye(ns)
        Idna = np.eye(na)
        dRB = RB.duplicate()
        dRHTC = RHTC.duplicate()
        dCHTRB = np.zeros_like(CHTRB)

        F1K = SLEPc.BV().create(comm=comm)
        F1K.setSizes(self.R_ops[0]._dimensions[0], RHTC.getSizes()[-1])
        F1K.setType('mat')
        F2K = F1K.duplicate()
        F3K = F1K.duplicate()
        F4K = F1K.duplicate()
        F5K = F1K.duplicate()

        F1B = SLEPc.BV().create(comm=comm)
        F1B.setSizes(self.R_ops[0]._dimensions[0], RB.getSizes()[-1])
        F1B.setType('mat')
        F2B = F1B.duplicate()
        F3B = F1B.duplicate()
        F4B = F1B.duplicate()
        F5B = F1B.duplicate()

        F1C = SLEPc.BV().create(comm=comm)
        F1C.setSizes(self.R_ops[0]._dimensions[-1], RHTC.getSizes()[-1])
        F1C.setType('mat')
        F2C = F1C.duplicate()
        F3C = F1C.duplicate()
        F4C = F1C.duplicate()
        F5C = F1C.duplicate()
        
        grad_K = np.zeros(K.shape, dtype=np.complex128)
        grad_p = np.zeros(len(p), dtype=np.complex128)
        grad_s = np.zeros(len(s), dtype=np.complex128)

        scals_p = np.ones(na)
        scals_s = np.ones(ns)
        scals_ps = np.ones((ns, na))

        for i in range (len(self.freqs)):
            
            w = self.weights[i]
            R = self.R_ops[i]
            RB = self.RBinterps[i][0](p, RB, scals_p)
            RHTC = self.RHTCinterps[i][0](s, RHTC, scals_s)
            CHTRB = self.CHTRBinterps[i][0](p, s, CHTRB, scals_ps)
            S = sp.linalg.inv(Id + CHTRB@K)
            M = LowRankLinearOperator(self.comm, RB, K@S, RHTC)

            # Gradient with respect to K
            SHT = S.conj().T
            SHTmat = PETSc.Mat().createDense(SHT.shape, None, SHT, commself)
            F1K.mult(1.0, 0.0, RHTC, SHTmat)
            F4K = self.FFHT.apply_mat(F1K, F4K)
            F2K = R.apply_mat(F4K, F2K)
            F3K = M.apply_mat(F4K, F3K)
            bv_add(-1.0, F3K, F2K)
            F5K = self.DDHT.apply_mat(F3K, F5K)
            grad_Ki_mat = F5K.dot(RB)
            grad_Ki = (Idna - (K@S@CHTRB).conj().T)@grad_Ki_mat.getDenseArray()
            grad_K += 2*w*grad_Ki
            grad_Ki_mat.destroy()
            SHTmat.destroy()

            # Gradient with respect to p
            SHTKHT = S.conj().T@K.conj().T
            Mat = PETSc.Mat().createDense(SHTKHT.shape, None, SHTKHT, commself)
            F1B.mult(1.0, 0.0, RHTC, Mat)
            F4B = self.FFHT.apply_mat(F1B, F4B)
            F2B = R.apply_mat(F4B, F2B)
            F3B = M.apply_mat(F4B, F3B)
            bv_add(-1.0, F3B, F2B)
            F5B = self.DDHT.apply_mat(F3B, F5B)
            for j in range (len(p)):
                l = j//num_params_actu          # Column of B affected by pj
                k = np.mod(j, num_params_actu)  # Index to identify df/dpj
                scals_dp = np.zeros(na)
                scals_dp[l] = 1.0
                scals_dps = np.zeros((ns, na))
                scals_dps[:, l] = 1.0
                Mj = K@S@self.CHTRBinterps[i][k+1](p, s, dCHTRB, scals_dps)
                Mjmat = PETSc.Mat().createDense(Mj.shape, None, Mj, commself)
                dRB = self.RBinterps[i][k+1](p, dRB, scals_dp)
                dRB.mult(-1.0, 1.0, RB, Mjmat)
                bv_conj(dRB)
                grad_p[j] += 2*w*_compute_bv_contraction(self.comm, dRB, F5B)
                Mjmat.destroy()
            Mat.destroy()

            # Gradient with respect to s
            KS = K@S
            KSmat = PETSc.Mat().createDense(KS.shape, None, KS, commself)
            F1C.mult(1.0, 0.0, RB, KSmat)
            F4C = self.DDHT.apply_mat(F1C, F4C)
            F2C = R.apply_hermitian_transpose_mat(F4C, F2C)
            F3C = M.apply_hermitian_transpose_mat(F4C, F3C)
            bv_add(-1.0, F3C, F2C)
            F5C = self.FFHT.apply_mat(F3C, F5C)
            for j in range (len(s)):
                l = j//num_params_sens          # Column of C affected by sj
                k = np.mod(j, num_params_sens)  # Index to identify df/dpj
                scals_ds = np.zeros(ns)
                scals_ds[l] = 1.0
                scals_dps = np.zeros((ns, na))
                scals_dps[l,] = 1.0
                dCHTRB = self.CHTRBinterps[i][k+1+num_params_actu](p, \
                                                        s, dCHTRB, scals_dps)
                Mj = (dCHTRB@K@S).conj().T
                Mjmat = PETSc.Mat().createDense(Mj.shape, None, Mj, commself)
                dRHTC = self.RHTCinterps[i][k+1](s, dRHTC, scals_ds)
                dRHTC.mult(-1.0, 1.0, RHTC, Mjmat)
                bv_conj(dRHTC)
                grad_s[j] += 2*w*_compute_bv_contraction(self.comm, dRHTC, F5C)
                Mjmat.destroy()
            KSmat.destroy()

            # process = psutil.Process(os.getpid())
            # value = process.memory_info().rss/(1024 * 1024)
            # value = sum(comm.allgather(value))
            # if opt_obj.comm.Get_rank() == 0:
            #     print(f"Iteration {i} usage {value} MB")
        objects = [F1K, F2K, F3K, F4K, F5K, F1B, F2B, F3B, F4B, F5B, F1C, F2C, \
                   F3C, F4C, F5C, RB, RHTC, dRB, dRHTC]
        for obj in objects: obj.destroy()
        return grad_p.real, grad_K.real, grad_s.real

class BodeFormula:

    def __init__(self, comm, frequencies, weights, CHTRBinterps, \
                 gamma, alpha, open_loop_evals):
        
        self.comm = comm
        self.freqs = frequencies
        self.weights = weights
        self.CHTRBinterps = CHTRBinterps
        self.gamma = gamma
        self.alpha = alpha
        self.open_loop_evals = open_loop_evals

    def evaluate_cost(self, params):
        p, K, s = params
        na, ns = K.shape
        CHTRB = np.zeros((K.shape[-1], K.shape[0]), dtype=np.complex128)
        Id = np.eye(K.shape[-1])
        scals_ps = np.ones((ns, na))
        integral = 0
        f = 1./np.pi if np.min(self.freqs) > 0.0 else 1./(2*np.pi)
        for i in range (len(self.freqs) - 1):
            L = self.CHTRBinterps[i][0](p, s, CHTRB, scals_ps)@K
            _, svals, _ = sp.linalg.svd(Id + L)
            integral += f*self.weights[i]*np.sum(np.log(1./svals))
        L = self.CHTRBinterps[-1][0](p, s, CHTRB, scals_ps)@K
        chi_gamma = np.sum(self.open_loop_evals.real - self.gamma) - \
                    0.5*np.trace(1j*self.freqs[-1]*L).real - integral.real
        exp_arg = self.alpha*np.abs(chi_gamma)
        value = chi_gamma**2*np.exp(exp_arg) if exp_arg < np.log(1e13) else 1e13
        return value
        
    def evaluate_gradient(self, params):

        p, K, s = params
        na, ns = K.shape
        num_params_actu = len(p)//na
        num_params_sens = len(s)//ns

        CHTRB = np.zeros((ns, na), dtype=np.complex128)
        Id = np.eye(ns)
        dCHTRB = np.zeros_like(CHTRB)

        grad_K = np.zeros(K.shape, dtype=np.complex128)
        grad_p = np.zeros(len(p), dtype=np.complex128)
        grad_s = np.zeros(len(s), dtype=np.complex128)

        scals_ps = np.ones((ns, na))

        integral = 0
        f = 1./np.pi if np.min(self.freqs) > 0.0 else 1./(2*np.pi)
        for i in range (len(self.freqs) - 1):
            wi = self.weights[i]

            # Gradient with respect to K
            CHTRB = self.CHTRBinterps[i][0](p, s, CHTRB, scals_ps)
            S = sp.linalg.inv(Id + CHTRB@K)
            grad_K += f*wi*(S@CHTRB).conj().T

            # Gradient with respect to p
            for j in range (len(p)):
                l = j//num_params_actu          # Column of B affected by pj
                k = np.mod(j, num_params_actu)  # Index to identify df/dpj
                scals_dps = np.zeros((ns, na))
                scals_dps[:, l] = 1.0
                dCHTRB = self.CHTRBinterps[i][k+1](p, s, dCHTRB, scals_dps)
                grad_p[j] += f*wi*np.einsum('ij,ij', dCHTRB.conj(), \
                                            (K@S).conj().T)

            # Gradient with respect to s
            for j in range (len(s)):
                l = j//num_params_sens          # Column of C affected by sj
                k = np.mod(j, num_params_sens)  # Index to identify df/dpj
                scals_dps = np.zeros((ns, na))
                scals_dps[l,] = 1.0
                dCHTRB = self.CHTRBinterps[i][k+1+num_params_actu](p, \
                                                        s, dCHTRB, scals_dps)
                grad_s[j] += f*wi*np.einsum('ij,ij', dCHTRB.conj(), \
                                            (K@S).conj().T)

            # Cost function evaluation
            _, svals, _ = sp.linalg.svd(S)
            integral += f*wi*np.sum(np.log(svals))


        # Gradient with respect to K
        CHTRB = self.CHTRBinterps[-1][0](p, s, CHTRB, scals_ps)
        grad_K -= 0.5*(1j*self.freqs[-1]*CHTRB).conj().T

        # Gradient with respect to p
        for j in range (len(p)):
            l = j//num_params_actu          # Column of B affected by pj
            k = np.mod(j, num_params_actu)  # Index to identify df/dpj
            scals_dps = np.zeros((ns, na))
            scals_dps[:, l] = 1.0
            dCHTRB = self.CHTRBinterps[-1][k+1](p, s, dCHTRB, scals_dps)
            grad_p[j] -= 0.5*np.einsum('ij,ij', dCHTRB.conj(), \
                                       (1j*self.freqs[-1]*K).conj().T)

        # Gradient with respect to s
        for j in range (len(s)):
            l = j//num_params_sens          # Column of C affected by sj
            k = np.mod(j, num_params_sens)  # Index to identify df/dpj
            scals_dps = np.zeros((ns, na))
            scals_dps[l,] = 1.0
            dCHTRB = self.CHTRBinterps[-1][k+1+num_params_actu](p, \
                                                    s, dCHTRB, scals_dps)
            grad_s[j] -= 0.5*np.einsum('ij,ij', dCHTRB.conj(), \
                                       (1j*self.freqs[-1]*K).conj().T)

        
        # Cost function evaluation
        L = self.CHTRBinterps[-1][0](p, s, CHTRB, scals_ps)@K
        chi_gamma = np.sum(self.open_loop_evals.real - self.gamma) - \
                    0.5*np.trace(1j*self.freqs[-1]*L).real - integral.real
        
        exp_arg = self.alpha*np.abs(chi_gamma)
        if exp_arg < np.log(1e13):
            expval = np.exp(exp_arg)
            value = (2*chi_gamma + self.alpha*chi_gamma**2)*expval \
                if chi_gamma > 0 else \
                    (2*chi_gamma - self.alpha*chi_gamma**2)*expval
        else:
            value = 1e13
        
        grad_K *= value
        grad_s *= value
        grad_p *= value

        return grad_p.real, grad_K.real, grad_s.real


def create_objective_and_gradient(manifold, WhichGrads, \
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
            f"At least one of H2Comp or StabComp should be provided"
        )

    euclidean_hessian = None

    @pymanopt.function.numpy(manifold)
    def cost(*params):
        JH2 = H2Comp.evaluate_cost(params) if H2Comp != None else 0.0
        JS = StabComp.evaluate_cost(params) if StabComp != None else 0.0
        return JH2 + JS
    
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):
        grad_pH2, grad_KH2, grad_sH2 = 0.0, 0.0, 0.0
        grad_pStb, grad_KStb, grad_sStb = 0.0, 0.0, 0.0
        if H2Comp != None:
            grad_pH2, grad_KH2, grad_sH2 = H2Comp.evaluate_gradient(params)
        if StabComp != None:
            grad_pStb, grad_KStb, grad_sStb = StabComp.evaluate_gradient(params)
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
        
