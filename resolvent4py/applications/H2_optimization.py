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

def _compute_left_and_right_eigendecomposition(lin_op, krylov_dim, n_evals):

    Df, V_ = eigendecomposition(lin_op, lin_op.solve, krylov_dim, n_evals)
    Da, W = eigendecomposition(lin_op, lin_op.solve_hermitian_transpose, \
                                krylov_dim, n_evals)
    
    Dfseq = distributed_to_sequential_matrix(lin_op.get_comm(), Df)
    Daseq = distributed_to_sequential_matrix(lin_op.get_comm(), Da)

    D = compute_dense_inverse(lin_op.get_comm(), Df)
    D.scale(-1.0)

    df = -1./Dfseq.getDiagonal().getArray()
    da = -1./Daseq.getDiagonal().getArray()
    idces = []
    for j in range (len(df)):
        idces.append(np.argmin(np.abs(da - df[j].conj())))
    
    W_ = SLEPc.BV().createFromMat(W)
    W_.setFromOptions()
    for j in range (len(idces)):
        w = W.getColumnVector(idces[j])
        W_.insertVec(j, w)
        petscprint(lin_op.get_comm(), "%d/%d"%(j,len(idces)))

    print(idces)
    W.destroy()
    Wmat = W_.getMat()
    W = Wmat.copy()
    W_.restoreMat(Wmat)
    W_.destroy()
    W.hermitianTranspose()
    M = W.matMult(V_)
    # M.view()
    Minv = compute_dense_inverse(lin_op.get_comm(), M)
    W.hermitianTranspose()
    V = V_.matMult(Minv)
    V_.destroy()
    Da.destroy()

    return V, D, W

def _compute_double_contraction(comm, Mat1, Mat2):
    Mat1_array = Mat1.getDenseArray()
    Mat2_array = Mat2.getDenseArray()
    value = comm.allreduce(np.sum(Mat1_array*Mat2_array), op=MPI.SUM)
    return value

def _compute_trace(comm, L1, L2, L2_hermitian_transpose=False):
    r"""
        Compute the trace of the product of two low-rank operators :math:`L_1` 
        and :math:`L_2` (see 
        :meth:`resolvent4py.linear_operators.LowRankLinearOperator`). If
        :code:`L2_hermitian_transpose==False`, compute 
        :math:`\text{Tr}(L_1 L_2)`, else :math:`\text{Tr}(L_1 L_2^*)`.

        :param comm: MPI communicator
        :param L1: low-rank linear operator
        :param L2: low-rank linear operator
        :param L2_hermitian_transpose: [optional] :code:`True` or :code:`False`
        :type L2_hermitian_transpose: bool

        :rtype: PETSc scalar
    """
    if L2_hermitian_transpose == False:
        F1 = L1.apply_mat(L2.U.matMult(L2.Sigma))
        L2.V.hermitianTranspose()
        F = L2.V.matMult(F1)
        L2.V.hermitianTranspose()
        F1.destroy()
    else:
        L2.Sigma.hermitianTranspose()
        F1 = L2.V.matMult(L2.Sigma)
        L2.Sigma.hermitianTranspose()
        F2 = L1.apply_mat(F1)
        F1.destroy()
        L2.U.hermitianTranspose()
        F = L2.U.matMult(F2)
        L2.U.hermitianTranspose()
        F2.destroy()
    trace = comm.allreduce(np.sum(F.getDiagonal().getArray()), op=MPI.SUM)
    return trace

def _assemble_woodbury_low_rank_operator(comm, R, B, C, Kd):
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
    linOp = LowRankLinearOperator(comm, RB, KLinv, RHTC)
    return linOp, Linv

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
        return value
    
    def evaluate_gradient_exponential(self, val):
        alpha = self.stab_params[0]
        return alpha*self.evaluate_exponential(val)
    
    def compute_finite_difference(self, p, grad_B, compute_dB):

        grad_p = np.zeros_like(p)
        for j in range (len(p)):
            pjp = p.copy()
            pjm = p.copy()
            pjp[j] += 1e-5
            pjm[j] -= 1e-5
            dB = compute_dB(pjp, pjm)/(2e-5)
            dB.conjugate()
            grad_p[j] = _compute_double_contraction(self.comm, dB, grad_B).real
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
            M, _ = _assemble_woodbury_low_rank_operator(opt_obj.comm, \
                                                        opt_obj.R_ops[i], \
                                                        B, C, Kd)
            
            Ji += _compute_trace(opt_obj.comm, opt_obj.R_ops[i], \
                                opt_obj.R_ops[i], True)
            Ji += -2.0*_compute_trace(opt_obj.comm, opt_obj.R_ops[i], M, True)
            Ji += _compute_trace(opt_obj.comm, M, M, True)
            J += opt_obj.weights[i]*Ji
            M.destroy()
        
        # Stability-promoting component of the cost function
        lin_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.A, B, Kd, C)
        D_, _ = eigendecomposition(lin_op, lin_op.solve, \
                                   opt_obj.stab_params[2], \
                                   opt_obj.stab_params[3])
        D = -1./D_.getDiagonal().getArray()
        Jd = np.sum([opt_obj.evaluate_exponential(D[k]) \
                     for k in range (len(D))])
        J += opt_obj.comm.allreduce(Jd, op=MPI.SUM)
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
        sizesLinvT = ((nsloc, ns), (nsloc, ns))
        sizesMUT = ((naloc, na), B.getSizes()[0])
        sizesSigmaT = ((nsloc, ns), (naloc, na))
        sizesCT = ((nsloc, ns), C.getSizes()[0])
        LinvT = PETSc.Mat().createDense(sizesLinvT, comm=opt_obj.comm)
        MUT = PETSc.Mat().createDense(sizesMUT, comm=opt_obj.comm)
        MSigmaT = PETSc.Mat().createDense(sizesSigmaT, comm=opt_obj.comm)
        BT = PETSc.Mat().createDense(sizesMUT, comm=opt_obj.comm)
        CT = PETSc.Mat().createDense(sizesCT, comm=opt_obj.comm)
        mats = [LinvT, MUT, MSigmaT, BT, CT]
        for mat in mats: mat.setUp()
        for i in range (len(opt_obj.freqs)):
            
            petscprint(opt_obj.comm, "Iteration %d"%i)
            wi = opt_obj.weights[i]
            M, Linv = _assemble_woodbury_low_rank_operator(opt_obj.comm, \
                                                    opt_obj.R_ops[i], B, C, Kd)
            Linv.setTransposePrecursor(LinvT)
            M.U.setTransposePrecursor(MUT)
            M.Sigma.setTransposePrecursor(MSigmaT)
            B.setTransposePrecursor(BT)
            C.setTransposePrecursor(CT)
            Linv.hermitianTranspose(LinvT)
            M.U.hermitianTranspose(MUT)
            M.Sigma.hermitianTranspose(MSigmaT)
            B.hermitianTranspose(BT)
            C.hermitianTranspose(CT)

            # Gradient with respect to K
            F1 = M.V.matMult(LinvT)
            F2 = M.apply_mat(F1)
            F3 = opt_obj.R_ops[i].apply_mat(F1)
            F2.axpy(-1.0, F3)
            grad_K_i = MUT.matMult(F2)
            F4 = MSigmaT.matMult(grad_K_i)
            F5 = C.matMult(F4)
            F6 = MUT.matMult(F5)
            grad_K_i.axpy(-1.0, F6)
            grad_K.axpy(2.0*wi, grad_K_i)
            mats = [F1, F2, F3, F4, F5, F6, grad_K_i]
            for mat in mats: mat.destroy()
            
            # Gradient with respect to p
            F1 = M.V.matMult(MSigmaT)
            F2 = M.apply_mat(F1)
            F3 = opt_obj.R_ops[i].apply_mat(F1)
            F2.axpy(-1.0, F3)
            grad_B_i = opt_obj.R_ops[i].apply_hermitian_transpose_mat(F2)
            F4 = BT.matMult(grad_B_i)
            F5 = MSigmaT.matMult(F4)
            F6 = M.V.matMult(F5)
            grad_B_i.axpy(-1.0, F6)
            grad_p += 2*wi*opt_obj.compute_finite_difference(p, grad_B_i, \
                                                             opt_obj.compute_dB)
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
            grad_s += 2*wi*opt_obj.compute_finite_difference(s, grad_C_i, \
                                                             opt_obj.compute_dC)
            mats = [F1, F2, F3, F4, F5, F6, grad_C_i]
            for mat in mats: mat.destroy()
            M.destroy()
        
        # Stability-promoting penalty
        lin_op = LowRankUpdatedLinearOperator(opt_obj.comm, opt_obj.A, B, Kd, C)
        V, D_, W = _compute_left_and_right_eigendecomposition(lin_op,\
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
        lin_op.destroy()

        sizesVT = (V.getSizes()[-1], V.getSizes()[0])
        sizesWT = (W.getSizes()[-1], W.getSizes()[0])
        sizesKdT = (Kd.getSizes()[-1], Kd.getSizes()[0])
        WT = PETSc.Mat().createDense(sizesWT, comm=opt_obj.comm)
        VT = PETSc.Mat().createDense(sizesVT, comm=opt_obj.comm)
        KdT = PETSc.Mat().createDense(sizesKdT, comm=opt_obj.comm)
        mats = [VT, WT, KdT]
        for mat in mats: mat.setUp()
        W.setTransposePrecursor(WT)
        W.hermitianTranspose(WT)
        V.setTransposePrecursor(VT)
        V.hermitianTranspose(VT)
        Kd.setTransposePrecursor(KdT)
        Kd.hermitianTranspose(KdT)

        # Gradient with respect to K
        F1 = VT.matMult(C)
        F2 = M.matMult(F1)
        F3 = W.matMult(F2)
        F4 = BT.matMult(F3)
        grad_K.axpy(-1.0, F4)
        mats = [F1, F2, F3, F4]
        for mat in mats: mat.destroy()
        
        # Gradient with respect to p
        F1 = C.matMult(KdT)
        F2 = VT.matMult(F1)
        F3 = M.matMult(F2)
        grad_B = W.matMult(F3)
        grad_p += -opt_obj.compute_finite_difference(p, grad_B, \
                                                     opt_obj.compute_dB)
        mats = [F1, F2, F3, grad_B]
        for mat in mats: mat.destroy()

        # Gradient with respect to s
        F1 = B.matMult(Kd)
        F2 = WT.matMult(F1)
        F3 = M.matMult(F2)
        grad_C = V.matMult(F3)
        grad_s += -opt_obj.compute_finite_difference(s, grad_C, \
                                                     opt_obj.compute_dC)
        mats = [F1, F2, F3, grad_C]
        for mat in mats: mat.destroy()

        grad_K_seq = distributed_to_sequential_matrix(opt_obj.comm, grad_K)
        grad_K_ = grad_K_seq.getDenseArray().copy().real

        mats = [VT, WT, KdT, BT, CT, MUT, grad_K_seq, V, W, B, C]
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
        
    

