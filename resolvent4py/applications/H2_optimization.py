from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pymanopt

from ..io_functions import read_dense_matrix
from ..io_functions import read_coo_matrix
from ..linear_operators import MatrixLinearOperator
from ..linear_operators import LowRankLinearOperator
from ..linear_operators import LowRankUpdatedLinearOperator
from ..solvers_and_preconditioners_functions import create_mumps_solver
from ..applications import eigendecomposition
from ..applications import right_and_left_eigendecomposition
from ..comms import sequential_to_distributed_matrix
from ..comms import distributed_to_sequential_matrix
from ..linalg import compute_local_size
from ..linalg import compute_dense_inverse
from ..linalg import compute_matrix_product_contraction
from ..linalg import compute_trace_product
from ..linalg import hermitian_transpose
from ..miscellaneous import petscprint


def compute_woodbury_update(comm, R, B, K, C):
    r"""
        Assemble low-rank operator representation of 

        .. math::
            
            M = RBK\underbrace{\left(I + C^*RB K\right)^{-1}}_{L^{-1}}C^*R

        :param R: a low-rank linear operator
        :param B: a dense PETSc matrix
        :type B: PETSc.Mat.Type.DENSE
        :param K: a dense PETSc matrix
        :type K: PETSc.Mat.Type.DENSE
        :param C: a dense PETSc matrix
        :type C: PETSc.Mat.Type.DENSE

        :return: a 2-tuple with a low-rank operator representation of :math:`M` 
            and the PETSc dense matrix :math:`L^{-1}`

    """
    RB = R.apply_mat(B)
    RHTC = R.apply_hermitian_transpose_mat(C)
    C.hermitianTranspose()
    CHTRB = C.matMult(RB)
    C.hermitianTranspose()
    L = CHTRB.matMult(K)
    Id = PETSc.Mat().createConstantDiagonal(L.getSizes(), 1.0, comm=comm)
    Id.convert(PETSc.Mat.Type.DENSE)
    L.axpy(1.0,Id,structure=PETSc.Mat.Structure.SAME)
    Linv = compute_dense_inverse(comm,L)
    KLinv = K.matMult(Linv)
    M = LowRankLinearOperator(comm, RB, KLinv, RHTC)
    return M, Linv

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
            dB = compute_dB(pjp, pjm)/(2e-5)
            dB.conjugate()
            grad_p[j] = compute_matrix_product_contraction(self.comm, \
                                                           dB, grad_B).real
            dB.destroy()
        return grad_p


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
        comm = opt_obj.comm
        p, K_, s = params

        B = opt_obj.compute_B(p)
        C = opt_obj.compute_C(s)
        na = B.getSizes()[-1][-1]
        ns = C.getSizes()[-1][-1]
        sizes = ((compute_local_size(na), na), (compute_local_size(ns), ns))
        Kseq = PETSc.Mat().createDense((na, ns), None, K_, MPI.COMM_SELF)
        K = PETSc.Mat().createDense(sizes, None, None, comm)
        sequential_to_distributed_matrix(Kseq, K)

        # H2 component of the cost function
        J = 0
        for i in range (len(opt_obj.freqs)):
            wi = opt_obj.weights[i]
            Ri = opt_obj.R_ops[i]
            Mi, _ = compute_woodbury_update(comm, Ri, B, K, C)
            J += wi*compute_trace_product(comm, Ri, Ri, True)
            J -= 2.0*wi*compute_trace_product(comm, Ri, Mi, True)
            J += wi*compute_trace_product(comm, Mi, Mi, True)
            Mi.destroy()
        
        # Stability-promoting component of the cost function
        cl_op = LowRankUpdatedLinearOperator(comm, opt_obj.A, B, K, C)
        krylov_dim, n_evals = opt_obj.stab_params[2], opt_obj.stab_params[3]
        Dd, _ = eigendecomposition(cl_op, cl_op.solve, krylov_dim, n_evals)
        evals = -1./Dd.getDiagonal().getArray().copy()
        Jd = np.sum([opt_obj.evaluate_exponential(val) for val in evals])
        J += opt_obj.comm.allreduce(Jd, op=MPI.SUM)
        Dd.destroy()
        cl_op.destroy()
        return J.real
    
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):
        """
            Evaluate the euclidean gradient of the cost function with 
            respect to the parameters
        """
        comm = opt_obj.comm
        p, K_, s = params

        B = opt_obj.compute_B(p)
        C = opt_obj.compute_C(s)
        na = B.getSizes()[-1][-1]
        ns = C.getSizes()[-1][-1]
        sizes = ((compute_local_size(na), na), (compute_local_size(ns), ns))
        Kseq = PETSc.Mat().createDense((na, ns), None, K_, MPI.COMM_SELF)
        K = PETSc.Mat().createDense(sizes, None, None, comm)
        sequential_to_distributed_matrix(Kseq, K)

        grad_K = K.duplicate()
        grad_K.zeroEntries()
        grad_p = np.zeros_like(p)
        grad_s = np.zeros_like(s)

        BT = hermitian_transpose(comm, B)
        CT = hermitian_transpose(comm, C)

        for i in range (len(opt_obj.freqs)):
            
            wi = opt_obj.weights[i]
            Ri = opt_obj.R_ops[i]
            M, Linv = compute_woodbury_update(comm, Ri, B, K, C)
            
            # Create matrices to hold the hermitian transposes
            LinvT = hermitian_transpose(comm, Linv)
            SigmaT = hermitian_transpose(comm, M.Sigma)
            UT = hermitian_transpose(comm, M.U)

            # Gradient with respect to K
            F1 = M.V.matMult(LinvT)
            F2 = M.apply_mat(F1)
            F3 = Ri.apply_mat(F1)
            F2.axpy(-1.0, F3)
            grad_K_i = UT.matMult(F2)
            F4 = SigmaT.matMult(grad_K_i)
            F5 = C.matMult(F4)
            F6 = UT.matMult(F5)
            grad_K_i.axpy(-1.0, F6)
            grad_K.axpy(2.0*wi, grad_K_i)
            mats = [F1, F2, F3, F4, F5, F6, grad_K_i]
            for mat in mats: mat.destroy()
            
            # Gradient with respect to p
            F1 = M.V.matMult(SigmaT)
            F2 = M.apply_mat(F1)
            F3 = Ri.apply_mat(F1)
            F2.axpy(-1.0, F3)
            grad_B_i = opt_obj.R_ops[i].apply_hermitian_transpose_mat(F2)
            F4 = BT.matMult(grad_B_i)
            F5 = SigmaT.matMult(F4)
            F6 = M.V.matMult(F5)
            grad_B_i.axpy(-1.0, F6)
            grad_p += 2*wi*opt_obj.compute_dBdp(p, grad_B_i, opt_obj.compute_dB)
            mats = [F1, F2, F3, F4, F5, F6, grad_B_i]
            for mat in mats: mat.destroy()
            
            # Gradient with respect to s
            F1 = M.U.matMult(M.Sigma)
            F2 = M.apply_hermitian_transpose_mat(F1)
            F3 = opt_obj.R_ops[i].apply_hermitian_transpose_mat(F1)
            F2.axpy(-1.0, F3)
            grad_C_i = opt_obj.R_ops[i].apply_mat(F2)
            F4 = CT.matMult(grad_C_i)
            F5 = M.Sigma.matMult(F4)
            F6 = M.U.matMult(F5)
            grad_C_i.axpy(-1.0, F6)
            grad_s += 2*wi*opt_obj.compute_dBdp(s, grad_C_i, opt_obj.compute_dC)
            mats = [F1, F2, F3, F4, F5, F6, grad_C_i]
            for mat in mats: mat.destroy()

            mats = [LinvT, UT, SigmaT]
            for mat in mats: mat.destroy()
            M.destroy()
        
        # Stability-promoting penalty
        cl_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.A, B, K, C)
        krylov_dim, n_evals = opt_obj.stab_params[2], opt_obj.stab_params[3]
        V, D, W = right_and_left_eigendecomposition(cl_op, cl_op.solve, \
                                                    krylov_dim, n_evals, \
                                                    lambda x: -1./x)
        Da = D.getDiagonal().getArray()
        Ma = np.asarray([opt_obj.evaluate_gradient_exponential(v) for v in Da])
        Mvec = PETSc.Vec().createWithArray(Ma, comm=comm)
        M = PETSc.Mat().createDiagonal(Mvec)
        cl_op.destroy()

        WT = hermitian_transpose(comm, W)
        VT = hermitian_transpose(comm, V)
        KT = hermitian_transpose(comm, K)

        # Gradient with respect to K
        F1 = VT.matMult(C)
        F2 = M.matMult(F1)
        F3 = W.matMult(F2)
        F4 = BT.matMult(F3)
        grad_K.axpy(-1.0, F4)
        mats = [F1, F2, F3, F4]
        for mat in mats: mat.destroy()
        
        # Gradient with respect to p
        F1 = C.matMult(KT)
        F2 = VT.matMult(F1)
        F3 = M.matMult(F2)
        grad_B = W.matMult(F3)
        grad_p -= opt_obj.compute_dBdp(p, grad_B, opt_obj.compute_dB)
        mats = [F1, F2, F3, grad_B]
        for mat in mats: mat.destroy()

        # Gradient with respect to s
        F1 = B.matMult(K)
        F2 = WT.matMult(F1)
        F3 = M.matMult(F2)
        grad_C = V.matMult(F3)
        grad_s -= opt_obj.compute_dBdp(s, grad_C, opt_obj.compute_dC)
        mats = [F1, F2, F3, grad_C]
        for mat in mats: mat.destroy()

        grad_K_seq = distributed_to_sequential_matrix(opt_obj.comm, grad_K)
        grad_K_ = grad_K_seq.getDenseArray().copy().real

        mats = [VT, WT, KT, grad_K_seq, V, W, B, C, BT, CT, grad_K]
        for mat in mats: mat.destroy()
        
        return grad_p, grad_K_, grad_s
    
    return cost, euclidean_gradient, euclidean_hessian


def test_euclidean_gradient(M, opt_obj, params, eps):

    xa, K, xs = params
    comm = opt_obj.comm
    rank = comm.Get_rank()
    
    cost, grad, _ = create_objective_and_gradient(M,opt_obj)
    J = cost(*params)
    grad_xa, grad_K, grad_xs = grad(*params)
    
    petscprint(comm,"Cost = %1.15e"%J)
    
    # Check Sb gradient
    if rank == 0:
        delta = np.random.randn(*xa.shape)
        delta /= np.linalg.norm(delta)
        params_ = (xa + eps*delta, K, xs)
        print(grad_xa)
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
        print(grad_K)
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
        print(grad_xs)
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
        
    

