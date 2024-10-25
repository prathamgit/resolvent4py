from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np
import scipy as sp

from ..linalg import compute_dense_inverse
from ..linalg import enforce_complex_conjugacy
from ..miscellaneous import copy_mat_from_bv
from ..miscellaneous import create_dense_matrix
from ..miscellaneous import petscprint
from ..comms import sequential_to_distributed_matrix
from ..comms import distributed_to_sequential_matrix

def arnoldi_iteration(lin_op, lin_op_action, krylov_dim):
    r"""
        This function uses the Arnoldi iteration algorithm to compute an 
        orthonormal basis and the corresponding Hessenberg matrix 
        for the range of the linear operator specified by
        :code:`lin_op` and :code:`lin_op_action`.

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: the method (e.g., :code:`lin_op.apply`,
            :code:`lin_op.solve_hermitian_transpose` or others) 
            used to compute the action of the linear operator against a vector. 
            This argument allows the user to specify whether they want 
            the eigendecomposition of :math:`L`, :math:`L^*`, 
            :math:`L^{-1}` or :math:`L^{-*}`.
        :param krylov_dim: dimension of the Krylov subspace
        :type kyrlov_dim: int

        :return: a 2-tuple with an orthonormal basis for the Krylov subspace
            and the Hessenberg matrix
        :rtype: (SLEPc BV, numpy.ndarray)
    """
    comm = lin_op.get_comm()
    sizes = lin_op.get_dimensions()[0]
    nblocks = lin_op.get_nblocks()
    
    # Initialize the BV structure and the Hessenberg matrix
    Q = SLEPc.BV().create(comm=comm)
    Q.setSizes(sizes,krylov_dim)
    Q.setFromOptions()
    H = np.zeros((krylov_dim,krylov_dim),dtype=np.complex128)
    # Draw the first vector at random
    q = Q.createVec()
    q.setArray(np.random.randn(sizes[0])) if lin_op.real == True else \
        q.setArray(np.random.randn(sizes[0]) + 1j*np.random.randn(sizes[0]))
    enforce_complex_conjugacy(comm, nblocks) if lin_op.block_cc == True \
        else None
    q.scale(1./q.norm())
    Q.insertVec(0,q)
    # Perform Arnoldi iteration
    for k in range(1,krylov_dim+1):
        v = lin_op_action(q)
        string = "Arnoldi iteration (%d/%d) - ||Aq|| "%(k, krylov_dim) + \
                    "= %1.15e"%(v.norm())
        petscprint(comm, string)
        for j in range (k):
            qj = Q.getColumn(j)
            H[j,k-1] = v.dot(qj)
            v.axpy(-H[j,k-1],qj)
            Q.restoreColumn(j,qj)
        if k < krylov_dim:
            H[k,k-1] = v.norm()
            v.scale(1./H[k,k-1])
            Q.insertVec(k,v)
        q = v.copy()
        v.destroy()
    return (Q, H)


def eigendecomposition(lin_op, lin_op_action, krylov_dim, n_evals):
    r"""
        Compute the eigendecomposition of the linear operator :math:`L` 
        specified by :code:`lin_op` and :code:`lin_op_action`.

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: the method (e.g., :code:`lin_op.apply`,
            :code:`lin_op.solve_hermitian_transpose` or others) 
            used to compute the action of the linear operator against a vector. 
            This argument allows the user to specify whether they want 
            the eigendecomposition of :math:`L`, :math:`L^*`, 
            :math:`L^{-1}` or :math:`L^{-*}`.
        :param krylov_dim: dimension of the Arnoldi Krylov subspace
        :type krylov_dim: int
        :param n_evals: number of largest eigenvalues to return
        :type n_evals: int

        :return: a 2-tuple with the :code:`n_evals` largest-magnitude 
            eigenvalues and corresponding eigenvectors
        :rtype: (PETSc.Mat.Type.DENSE, PETSc.Mat.Type.DENSE)
    """
    Q, H = arnoldi_iteration(lin_op, lin_op_action, krylov_dim)
    evals, evecs = sp.linalg.eig(H)
    idces = np.flipud(np.argsort(np.abs(evals)).reshape(-1))[:n_evals]
    evals = evals[idces]
    evecs = evecs[:,idces]
    evecs_ = PETSc.Mat().createDense(evecs.shape,None,evecs,comm=MPI.COMM_SELF)
    Q.multInPlace(evecs_,0,n_evals)
    Q.setActiveColumns(0,n_evals)
    Qmat = copy_mat_from_bv(Q)
    Q.destroy()

    evals_vec = PETSc.Vec().createWithArray(evals, comm=MPI.COMM_SELF)
    Dseq = PETSc.Mat().createDiagonal(evals_vec)
    Dseq.convert(PETSc.Mat.Type.DENSE)
    sizes_D = Qmat.getSizes()[-1]
    D = create_dense_matrix(lin_op.get_comm(), (sizes_D, sizes_D))
    sequential_to_distributed_matrix(Dseq, D)
    return (D, Qmat)

def right_and_left_eigendecomposition(lin_op, lin_op_action, krylov_dim, \
                                      n_evals, process_evals=None):
    r"""
        Compute the right and left eigendecomposition of the linear operator 
        :math:`L` specified by :code:`lin_op` and :code:`lin_op_action`.

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: the method (e.g., :code:`lin_op.apply`,
            :code:`lin_op.solve_hermitian_transpose` or others) 
            used to compute the action of the linear operator against a vector. 
            This argument allows the user to specify whether they want 
            the eigendecomposition of :math:`L`, :math:`L^*`, 
            :math:`L^{-1}` or :math:`L^{-*}`.
        :param krylov_dim: dimension of the Arnoldi Krylov subspace
        :type krylov_dim: int
        :param n_evals: number of largest eigenvalues to return
        :type n_evals: int
        :param process_evals: [optional] a function to process the eigenvalues.
            The default is the identity function.

        :return: a 2-tuple with the :code:`n_evals` largest-magnitude 
            eigenvalues and corresponding eigenvectors
        :rtype: (PETSc.Mat.Type.DENSE, PETSc.Mat.Type.DENSE, \
            PETSc.Mat.Type.DENSE)
    """
    
    if lin_op_action == lin_op.solve:
        lin_op_action_adj = lin_op.solve_hermitian_transpose
    elif lin_op_action == lin_op.apply:
        lin_op_action_adj = lin_op.apply_hermitian_transpose
    process_evals = (lambda x: x) if process_evals == None else process_evals

    # Compute the right and left eigendecompositions
    Dfwd, Qfwd = eigendecomposition(lin_op, lin_op_action, krylov_dim, n_evals)
    Dadj, Qadj = eigendecomposition(lin_op, lin_op_action_adj, \
                                    krylov_dim, n_evals)
    # Match the right and left eigenvalues/vectors
    Dfwd_seq = distributed_to_sequential_matrix(lin_op.get_comm(), Dfwd)
    Dadj_seq = distributed_to_sequential_matrix(lin_op.get_comm(), Dadj)
    Dfwd_seq_ = process_evals(Dfwd_seq.getDiagonal().getArray().copy())
    Dadj_seq_ = process_evals(Dadj_seq.getDiagonal().getArray().copy())
    idces = [np.argmin(np.abs(Dfwd_seq_ - val.conj())) for val in Dadj_seq_]
    Qadj_array = Qadj.getDenseArray().copy()
    for j in range (len(idces)):
        q = Qadj.getColumnVector(idces[j])
        Qadj_array[:,j] = q.getArray()
        q.destroy()
    offset, _ = Qadj.getOwnershipRange()
    rows = np.arange(Qadj_array.shape[0], dtype=np.int64) + offset
    cols = np.arange(Qadj_array.shape[-1], dtype=np.int64)
    Qadj.setValues(rows, cols, Qadj_array.reshape(-1))
    Qadj.assemble(None)
    # Biorthogonalize the eigenvectors
    Qadj.hermitianTranspose()
    M = Qadj.matMult(Qfwd)
    Qadj.hermitianTranspose()
    Minv = compute_dense_inverse(lin_op.get_comm(), M)
    V = Qfwd.matMult(Minv)
    Qfwd.destroy()
    # Assemble the processed eigenvalue matrix
    evals_vec = PETSc.Vec().createWithArray(Dfwd_seq_, comm=MPI.COMM_SELF)
    Dseq = PETSc.Mat().createDiagonal(evals_vec)
    Dseq.convert(PETSc.Mat.Type.DENSE)
    sizes_D = Qadj.getSizes()[-1]
    D = create_dense_matrix(lin_op.get_comm(), (sizes_D, sizes_D))
    sequential_to_distributed_matrix(Dseq, D)
    return (V, D, Qadj)