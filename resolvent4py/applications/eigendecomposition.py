from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np
import scipy as sp

from ..petsc4py_helper_functions import enforce_complex_conjugacy

def arnoldi_iteration(lin_op, lin_op_action, krylov_dim):
    r"""
        This function uses the Arnoldi iteration algorithm to compute the 
        eigendecomposition of the linear operator :math:`L` specified by
        :code:`lin_op`. 

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: the method (e.g., :code:`lin_op.apply` or
            :code:`lin_op.solve_hermitian_transpose`) used to compute the action
            of the linear operator against a vector. This argument allows the 
            user to specify whether they want the eigendecomposition of 
            :math:`L`, :math:`L^*`, :math:`L^{-1}` or :math:`L^{-*}`.
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
        specified by :code:`lin_op`. 

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: the method (e.g., :code:`lin_op.apply` or
            :code:`lin_op.solve_hermitian_transpose`) used to compute the action
            of the linear operator against a vector. This argument allows the 
            user to specify whether they want the eigendecomposition of 
            :math:`L`, :math:`L^*`, :math:`L^{-1}` or :math:`L^{-*}`.
        :param krylov_dim: dimension of the Arnoldi Krylov subspace
        :type krylov_dim: int
        :param n_evals: number of largest eigenvalues to return
        :type n_evals: int

        :return: a 2-tuple with the :code:`n_evals` largest eigenvalues and
            corresponding eigenvectors
        :rtype: (PETSc Mat, PETSc Mat)
    """
    Q, H = arnoldi_iteration(lin_op, lin_op_action, krylov_dim)
    evals, evecs = sp.linalg.eig(H)
    idces = np.flipud(np.argsort(np.abs(evals)).reshape(-1))[:n_evals]
    evals = evals[idces]
    evecs = evecs[:,idces]
    evecs_ = PETSc.Mat().createDense(evecs.shape,None,evecs,comm=MPI.COMM_SELF)
    Q.multInPlace(evecs_,0,n_evals)
    Q.setActiveColumns(0,n_evals)

    Q_ = Q.getMat()
    Q_mat = Q_.copy()
    Q.restoreMat(Q_)
    Q.destroy()

    sizes_D = Q_mat.getSizes()[-1]
    D = PETSc.Mat().createDense((sizes_D, sizes_D), comm=lin_op.get_comm())
    D.setUp()
    for i in range (*D.getOwnershipRange()):
        D.setValues(i,i,evals[i])
    D.assemble(None)
    return (D, Q_mat)