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
    block_cc = lin_op._block_cc
    
    # Initialize the BV structure and the Hessenberg matrix
    Q = SLEPc.BV().create(comm=comm)
    Q.setSizes(sizes, krylov_dim)
    Q.setFromOptions()
    H = np.zeros((krylov_dim, krylov_dim),dtype=np.complex128)
    # Draw the first vector at random
    q = Q.createVec()
    qa = np.random.randn(sizes[0]) + 1j*np.random.randn(sizes[0])
    qa = qa.real if lin_op._real == True else qa
    q.setArray(qa)
    enforce_complex_conjugacy(comm, q, nblocks) if block_cc == True else None
    q.scale(1./q.norm())
    Q.insertVec(0,q)
    # Perform Arnoldi iteration
    for k in range(1,krylov_dim+1):
        v = lin_op_action(q)
        string = "Arnoldi iteration (%d/%d) - ||Aq|| "%(k, krylov_dim) + \
                    "= %1.15e"%(v.norm())
        # petscprint(comm, string)
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


def eig(lin_op, lin_op_action, krylov_dim, n_evals, process_evals=None):
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
        :rtype: (np.ndarray, SLEPc.BV)
    """
    Q, H = arnoldi_iteration(lin_op, lin_op_action, krylov_dim)
    evals, evecs = sp.linalg.eig(H)
    idces = np.flipud(np.argsort(np.abs(evals)).reshape(-1))[:n_evals]
    evals = evals[idces]
    evecs = evecs[:,idces]
    evecs_ = PETSc.Mat().createDense(evecs.shape,None,evecs,comm=MPI.COMM_SELF)
    Q.multInPlace(evecs_,0,n_evals)
    Q.setActiveColumns(0,n_evals)
    process_evals = (lambda x: x) if process_evals == None else process_evals
    evals = process_evals(evals)
    return (np.diag(evals), Q)

def right_and_left_eig(lin_op, lin_op_action, krylov_dim, \
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
    else:
        raise ValueError (
            f"lin_op_action must be one of lin_op.solve or lin_op.apply where "
            f"lin_op is the first argument of the function."
        )
    # Compute the right and left eigendecompositions
    Dfwd, Qfwd = eig(lin_op, lin_op_action, krylov_dim, n_evals, process_evals)
    Dadj, Qadj = eig(lin_op, lin_op_action_adj, krylov_dim, \
                     n_evals, process_evals)
    # Match the right and left eigenvalues/vectors
    Dfwdd = np.diag(Dfwd)
    Dadjd = np.diag(Dadj)
    idces = [np.argmin(np.abs(Dfwdd - val.conj())) for val in Dadjd]
    Qadj_ = Qadj.copy()
    for j in range (len(idces)):
        q_ = Qadj_.getColumn(idces[j])
        Qadj.insertVec(j, q_)
        Qadj_.restoreColumn(idces[j], q_)
    Qadj_.destroy()
    # Biorthogonalize the eigenvectors
    M = Qfwd.dot(Qadj)
    Minv = sp.linalg.inv(M.getArray())
    Minv = PETSc.Mat().createDense(Minv.shape, None, Minv, MPI.COMM_SELF)
    M.destroy()
    Qfwd.multInPlace(Minv, 0, Minv.shape[-1])
    return (Qfwd, Dfwd, Qadj)