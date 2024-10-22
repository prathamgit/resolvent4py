from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import scipy as sp
import numpy as np

from ..linalg import enforce_complex_conjugacy
from ..miscellaneous import create_dense_matrix
from ..miscellaneous import copy_mat_from_bv

def randomized_svd(lin_op, lin_op_action, n_rand, n_loops, n_svals):
    r"""
        Compute the randomized SVD of the linear operator :math:`L` 
        specified by :code:`lin_op`. 

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: one of :code:`lin_op.apply_mat` or 
            :code:`lin_op.solve_mat`
        :param n_rand: number of random vectors to use
        :type n_rand: int
        :param n_loops: number of randomized svd iterations
        :type n_loops: int
        :param n_svals: number of singular triplets to return
        :type n_svals: int

        :return: :math:`(U,\,\Sigma,\, V)` a 3-tuple with the leading 
            :code:`n_svals` singular values and corresponding left and \
            right singular vectors
        :rtype: (PETSc Mat, PETSc Mat, PETSc Mat)
    """
    if lin_op_action == lin_op.apply_mat:
        lin_op_action_adj = lin_op.apply_hermitian_transpose_mat
    if lin_op_action == lin_op.solve_mat:
        lin_op_action_adj = lin_op.solve_hermitian_transpose_mat
    
    # Assemble random BV
    rowsizes = lin_op.get_dimensions()[0]
    X = SLEPc.BV().create(comm=lin_op.get_comm())
    X.setSizes(rowsizes,n_rand)
    X.setFromOptions()
    X.setRandomNormal()
    for j in range (n_rand):
        xj = X.getColumn(j)
        if lin_op.real:
            rows = np.arange(rowsizes[0], dtype=np.int64) + \
                xj.getOwnershipRange()[0]
            array = xj.getArray()
            xj.setValues(rows, array)
            xj.assemble(None)
        if lin_op.block_cc:
            enforce_complex_conjugacy(lin_op.get_comm(), xj, \
                                      lin_op.get_nblocks())
        X.restoreColumn(j, xj)
    X.orthogonalize(None)


    Qadj = X.duplicate()
    X_mat = X.getMat()
    Qadj_mat = Qadj.getMat()
    lin_op_action_adj(X_mat, Qadj_mat)
    X.restoreMat(X_mat)
    Qadj.restoreMat(Qadj_mat)
    X.destroy()
    Qadj.orthogonalize(None)
    Qfwd = Qadj.duplicate()
    j = 0
    while j < n_loops:
        Qadj_mat = Qadj.getMat()
        Qfwd_mat = Qfwd.getMat()
        lin_op_action(Qadj_mat, Qfwd_mat)
        Qfwd.restoreMat(Qfwd_mat)
        Qfwd.orthogonalize(None)
        Qfwd_mat = Qfwd.getMat()
        lin_op_action_adj(Qfwd_mat, Qadj_mat)
        Qfwd.restoreMat(Qfwd_mat)
        Qadj.restoreMat(Qadj_mat)
        j += 1
    R = create_dense_matrix(MPI.COMM_SELF, (n_rand, n_rand))
    Qadj.orthogonalize(R)
    u, s, v = sp.linalg.svd(R.getDenseArray())
    v = v.conj().T
    s = s[:n_svals]
    u = u[:,:n_svals]
    v = v[:,:n_svals]
    u = PETSc.Mat().createDense((n_rand,n_svals), None, u, comm=MPI.COMM_SELF)
    v = PETSc.Mat().createDense((n_rand,n_svals), None, v, comm=MPI.COMM_SELF)
    
    Qfwd.multInPlace(v,0,n_svals)
    Qfwd.setActiveColumns(0,n_svals)
    Qadj.multInPlace(u,0,n_svals)
    Qadj.setActiveColumns(0,n_svals)
    Qfwd_mat = copy_mat_from_bv(Qfwd)
    Qfwd.destroy()
    Qadj_mat = copy_mat_from_bv(Qadj)
    Qadj.destroy()

    sizes_S = Qfwd_mat.getSizes()[-1]
    S = create_dense_matrix(lin_op.get_comm(), (sizes_S, sizes_S))
    for i in range (*S.getOwnershipRange()):
        S.setValues(i,i,s[i])
    S.assemble(None)
    return (Qfwd_mat, S, Qadj_mat)