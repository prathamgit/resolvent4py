from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np
import scipy as sp

from ..petsc4py_helper_functions import enforce_complex_conjugacy

def _matrix_matrix_product(lin_op_action, X, Y):
    r"""
        Compute :code:`Y = lin_op_action(X)`

        :param lin_op_action: one of the :code:`apply`, 
            :code:`apply_hermitian_transpose`, :code:`solve` or 
            :code:`solve_hermitian_transpose` methods from any child class of
            the :code:`LinearOperator` class
        :param X: a SLEPc BV
        :type X: `BV`_
        :param Y: a SLEPc BV
        :type Y: `BV`_

        :rtype: None
    """
    _, ncols = X.getSizes()
    for j in range (ncols):
        x = X.getColumn(j)
        y = lin_op_action(x)
        Y.insertVec(j,y)
        X.restoreColumn(j,x)
    

def randomized_svd(lin_op, lin_op_action, n_rand, n_iter, n_svals):
    r"""
        Compute the randomized SVD of the linear operator :math:`L` 
        specified by :code:`lin_op`. 

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: one of :code:`lin_op.apply` or 
            :code:`lin_op.solve`
        :param n_rand: number of random vectors to use
        :type n_rand: int
        :param n_iter: number of randomized svd iterations
        :type n_iter: int
        :param n_svals: number of singular triplets to return
        :type n_svals: int

        :return: a 3-tuple with the :code:`n_svals` singular values and 
            corresponding left and right singular vectors
        :rtype: (SLEPc BV, numpy, SLEPc BV)
    """

    if lin_op_action == lin_op.apply:
        lin_op_action_adj = lin_op.apply_hermitian_transpose
    if lin_op_action == lin_op.solve:
        lin_op_action_adj = lin_op.solve_hermitian_transpose

    X = SLEPc.BV().create(comm=lin_op.get_comm())
    X.setSizes(lin_op.get_dimensions()[0],n_rand)
    X.setRandomNormal()
    X.orthogonalize(None)
    Qadj = X.duplicate()
    Qfwd = X.duplicate()
    _matrix_matrix_product(lin_op_action_adj, X, Qadj)
    Qadj.orthogonalize(None)
    j = 0
    while j < n_iter:
        _matrix_matrix_product(lin_op_action, Qadj, Qfwd)
        Qfwd.orthogonalize(None)
        _matrix_matrix_product(lin_op_action_adj, Qfwd, Qadj)
        j += 1
    R = PETSc.Mat().createDense((n_rand,n_rand),comm=MPI.COMM_SELF)
    R.setUp()
    Qadj.orthogonalize(R)
    u, s, v = sp.linalg.svd(R.getDenseArray())
    v = v.conj().T
    u = u[:,:n_svals]
    s = s[:n_svals]
    v = v[:,:n_svals]
    Qfwd.multInPlace(v,0,n_svals)
    Qfwd.setActiveColumns(0,n_svals)
    Qadj.multInPlace(u,0,n_svals)
    Qadj.setActiveColumns(0,n_svals)
    return (Qfwd,s,Qadj)