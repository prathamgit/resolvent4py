from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np
import scipy as sp
import pymanopt

from ..io_functions import read_dense_matrix
from ..io_functions import read_coo_matrix
from ..io_functions import read_vector
from ..petsc4py_helper_functions import petscprint
from ..petsc4py_helper_functions import compute_local_size
from ..petsc4py_helper_functions import compute_dense_inverse
from ..petsc4py_helper_functions import distributed_to_sequential_matrix
from ..petsc4py_helper_functions import sequential_to_distributed_matrix
from ..linear_operators import MatrixLinearOperator
from ..linear_operators import LowRankLinearOperator
from ..linear_operators import LowRankUpdatedLinearOperator
from ..solvers_and_preconditioners_functions import create_mumps_solver
from ..applications import eigendecomposition

def compute_left_and_right_eigendecomposition(lin_op, krylov_dim, n_evals):

    Df, V_ = eigendecomposition(lin_op, lin_op.solve, krylov_dim, n_evals)
    Da, W = eigendecomposition(lin_op, lin_op.solve_hermitian_transpose, \
                                krylov_dim, n_evals)
    
    df = -1./Df.getDiagonal().getArray()
    da = -1./Da.getDiagonal().getArray()
    idces = []
    for j in range (len(df)):
        idces.append(np.argmin(np.abs(da - df[j].conj())))

    W_ = SLEPc.BV().createFromMat(W)
    W_.setFromOptions()
    for j in range (len(idces)):
        w = W.getColumnVector(idces[j])
        W_.insertVec(j, w)

    W.destroy()
    Wmat = W_.getMat()
    W = Wmat.copy()
    W_.restoreMat(Wmat)
    W_.destroy()
    W.hermitianTranspose()
    M = W.matMult(V_)
    Minv = compute_dense_inverse(lin_op.get_comm(), M)
    W.hermitianTranspose()
    V = V_.matMult(Minv)
    V_.destroy()
    Da.destroy()

    return V, Df, W

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
        F1 = L1_action(L2.U.matMult(L2.Sigma))
        L2.V.hermitianTranspose()
        F = L2.V.matMult(F1)
        L2.V.hermitianTranspose()
        F1.destroy()
    else:
        L2.Sigma.hermitianTranspose()
        F1 = L2.V.matMult(L2.Sigma)
        L2.Sigma.hermitianTranspose()
        F2 = L1_action(F1)
        F1.destroy()
        L2.U.hermitianTranspose()
        F = L2.U.matMult(F2)
        L2.U.hermitianTranspose()
        F2.destroy()
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
        self.stab_params = stability_params
    
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
        alpha, beta = self.stab_params[:2]
        try:    value = np.exp(alpha*(val.real + beta)).real
        except: value = 1e12
        return value
    
    def evaluate_gradient_exponential(self,val):
        alpha = self.stab_params[0]
        return alpha*self.evaluate_exponential(val)


def create_objective_and_gradient(manifold,opt_obj):
    r"""
        :param manifold: one of the pymanopt manifolds
        :param opt_obj: instance of the optimization objects class
    """
    
    euclidean_hessian = None

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
        D_, _ = eigendecomposition(lin_op, lin_op.solve, \
                                   opt_obj.stab_params[2], \
                                   opt_obj.stab_params[3])
        D = -1./D_.getDiagonal().getArray()
        Jd = None
        if rank == 0:
            Jd = np.sum([opt_obj.evaluate_exponential(D[k]) \
                         for k in range (len(D))])
        J += opt_obj.comm.bcast(Jd, root=0)
        D_.destroy()
        lin_op.destroy()

        return J.real
    
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):
        """
            Evaluate the euclidean gradient of the cost function with 
            respect to the parameters
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
        grad_K = Kd.copy()
        grad_K.scale(0.0)
        for i in range (len(opt_obj.freqs)):

            wi = opt_obj.weights[i]

            M, Linv, KLinv = assemble_woodbury_low_rank_operator(opt_obj.comm, \
                                                    opt_obj.R_ops[i], B, C, Kd)
            
            # Gradient with respect to K
            Linv.hermitianTranspose()
            F1 = M.V.matMult(Linv)
            Linv.hermitianTranspose()
            F2 = M.apply_mat(F1)
            F2.axpy(-1.0,opt_obj.R_ops[i].apply_mat(F1))
            M.U.hermitianTranspose()
            grad_K_i = M.U.matMult(F2)
            KLinv.hermitianTranspose()
            grad_K_i.axpy(-1.0, M.U.matMult(C.matMult(KLinv.matMult(grad_K_i))))
            KLinv.hermitianTranspose()
            M.U.hermitianTranspose()
            F1.destroy()
            F2.destroy()
            grad_K.axpy(2.0*wi,grad_K_i)
            grad_K_i.destroy()

            # Gradient with respect to p
            KLinv.hermitianTranspose()
            F1 = M.V.matMult(KLinv)
            KLinv.hermitianTranspose()
            F2 = M.apply_mat(F1)
            F2.axpy(-1.0,opt_obj.R_ops[i].apply_mat(F1))
            grad_B_i = opt_obj.R_ops[i].apply_hermitian_transpose_mat(F2)
            B.hermitianTranspose()
            KLinv.hermitianTranspose()
            grad_B_i.axpy(-1.0, M.V.matMult(KLinv.matMult(B.matMult(grad_B_i))))
            B.hermitianTranspose()
            KLinv.hermitianTranspose()
            F1.destroy()
            F2.destroy()
            for j in range (len(p)):
                pjp = p.copy()
                pjm = p.copy()
                pjp[j] += 1e-5
                pjm[j] -= 1e-5
                dB = opt_obj.compute_dB(pjp, pjm)/(2e-5)
                dB.conjugate()
                grad_p[j] += 2*wi*compute_double_contraction(opt_obj.comm, \
                                                             dB, grad_B_i).real
                dB.destroy()
            grad_B_i.destroy()

            # Gradient with respect to s
            F1 = M.U.matMult(KLinv)
            F2 = M.apply_hermitian_transpose_mat(F1)
            F2.axpy(-1.0,opt_obj.R_ops[i].apply_hermitian_transpose_mat(F1))
            grad_C_i = opt_obj.R_ops[i].apply_mat(F2)
            C.hermitianTranspose()
            grad_C_i.axpy(-1.0,M.U.matMult(KLinv.matMult(C.matMult(grad_C_i))))
            C.hermitianTranspose()
            F1.destroy()
            F2.destroy()
            for j in range (len(s)):
                sjp = s.copy()
                sjm = s.copy()
                sjp[j] += 1e-5
                sjm[j] -= 1e-5
                dC = opt_obj.compute_dC(sjp, sjm)/(2e-5)
                dC.conjugate()
                grad_s[j] += 2*wi*compute_double_contraction(opt_obj.comm, \
                                                             dC, grad_C_i).real
                dC.destroy()
            grad_C_i.destroy()

            M.destroy()

        # Stability-promoting penalty
        lin_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.A, B, Kd, C)
        V, D_, W = compute_left_and_right_eigendecomposition(lin_op,\
                                                    opt_obj.stab_params[2], \
                                                    opt_obj.stab_params[3])
        D = D_.getDiagonal().getArray()
        M_ = np.asarray([opt_obj.evaluate_gradient_exponential(D[k]) \
                     for k in range (len(D))])
        M = D_.duplicate()
        i0, i1 = M.getOwnershipRange()
        for i in range (i0, i1):
            M.setValues(i,i,M_[i - i0])
        M.assemble(None)
        # lin_op.destroy()

        # Gradient with respect to K
        V.hermitianTranspose()
        B.hermitianTranspose()
        grad_K.axpy(-1.0, B.matMult(W.matMult(M.matMult(V.matMult(C)))))
        B.hermitianTranspose()
        V.hermitianTranspose()

        # Gradient with respect to p
        V.hermitianTranspose()
        Kd.hermitianTranspose()
        grad_B = W.matMult(M.matMult(V.matMult(C.matMult(Kd))))
        Kd.hermitianTranspose()
        V.hermitianTranspose()
        for j in range (len(p)):
            pjp = p.copy()
            pjm = p.copy()
            pjp[j] += 1e-5
            pjm[j] -= 1e-5
            dB = opt_obj.compute_dB(pjp, pjm)/(2e-5)
            dB.conjugate()
            grad_p[j] -= compute_double_contraction(opt_obj.comm, \
                                                    dB, grad_B).real
            dB.destroy()
        grad_B.destroy()

        # Gradient with respect to s
        W.hermitianTranspose()
        grad_C = V.matMult(M.matMult(W.matMult(B.matMult(Kd))))
        W.hermitianTranspose()
        for j in range (len(s)):
            sjp = s.copy()
            sjm = s.copy()
            sjp[j] += 1e-5
            sjm[j] -= 1e-5
            dC = opt_obj.compute_dC(sjp, sjm)/(2e-5)
            dC.conjugate()
            grad_s[j] -= compute_double_contraction(opt_obj.comm, \
                                                    dC, grad_C).real
            dC.destroy()
        grad_C.destroy()
        
        grad_K_seq = distributed_to_sequential_matrix(opt_obj.comm, grad_K)
        grad_K.destroy()
        grad_K = grad_K_seq.getDenseArray().copy().real
        grad_K_seq.destroy()
        B.destroy()
        C.destroy()
        
        return grad_p.real, grad_K, grad_s.real
    
    return cost, euclidean_gradient, euclidean_hessian


def test_euclidean_gradient(M, opt_obj, params, eps):

    xa, K, xs = params
    comm = opt_obj.comm
    
    cost, grad, _ = create_objective_and_gradient(M,opt_obj)
    J = cost(*params)
    grad_xa, grad_K, grad_xs = grad(*params)
    
    petscprint(comm,"Cost = %1.15e"%J)
    
    # Check Sb gradient
    delta = np.random.randn(*xa.shape)
    delta /= np.linalg.norm(delta)
    params_ = (xa + eps*delta, K, xs)   
    dfd = (cost(*params_) - J)/eps
    dgrad = (delta.conj().T@grad_xa).real
    error = np.abs(dfd - dgrad)
    percent_error = (error/np.abs(dfd)*100).real
    
    petscprint(comm,"------ Error xa grad -------------")
    petscprint(comm,"dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
    petscprint(comm,"---------------------------------")
    
    
    # Check K gradient
    delta = np.random.randn(*K.shape)
    delta /= np.sqrt(np.trace(delta.T@delta))
    params_ = (xa, K + eps*delta, xs)   
    dfd = (cost(*params_) - J)/eps
    dgrad = np.trace(delta.conj().T@grad_K).real
    error = np.abs(dfd - dgrad)
    percent_error = (error/np.abs(dfd)*100).real
    
    petscprint(comm,"------ Error K grad -------------")
    petscprint(comm,"dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
    petscprint(comm,"---------------------------------")
    
    
    # Check Sc gradient
    delta = np.random.randn(*xs.shape)
    delta /= np.linalg.norm(delta)
    params_ = (xa, K, xs + eps*delta)   
    dfd = (cost(*params_) - J)/eps
    dgrad = (delta.conj().T@grad_xs).real
    error = np.abs(dfd - dgrad)
    percent_error = (error/np.abs(dfd)*100).real
    
    petscprint(comm,"------ Error xs grad -------------")
    petscprint(comm,"dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
    petscprint(comm,"---------------------------------")
    
    

