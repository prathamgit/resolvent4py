from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import scipy as sp

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
    

def randomized_svd(lin_op, lin_op_action, n_rand, n_loops, n_svals):
    r"""
        Compute the randomized SVD of the linear operator :math:`L` 
        specified by :code:`lin_op`. 

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: one of :code:`lin_op.apply` or 
            :code:`lin_op.solve`
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
    if lin_op_action == lin_op.apply:
        lin_op_action_adj = lin_op.apply_hermitian_transpose
    if lin_op_action == lin_op.solve:
        lin_op_action_adj = lin_op.solve_hermitian_transpose
    
    X = SLEPc.BV().create(comm=lin_op.get_comm())
    X.setSizes(lin_op.get_dimensions()[0],n_rand)
    X.setFromOptions()
    X.setRandomNormal()
    X.orthogonalize(None)
    Qadj = X.duplicate()
    Qfwd = X.duplicate()
    _matrix_matrix_product(lin_op_action_adj, X, Qadj)
    X.destroy()
    Qadj.orthogonalize(None)
    j = 0
    while j < n_loops:
        _matrix_matrix_product(lin_op_action, Qadj, Qfwd)
        Qfwd.orthogonalize(None)
        _matrix_matrix_product(lin_op_action_adj, Qfwd, Qadj)
        j += 1
    R = PETSc.Mat().createDense((n_rand,n_rand),comm=MPI.COMM_SELF)
    R.setUp()
    Qadj.orthogonalize(R)
    u, s, v = sp.linalg.svd(R.getDenseArray())
    v = v.conj().T
    u = PETSc.Mat().createDense((n_rand,n_svals),None,\
                                u[:,:n_svals],comm=MPI.COMM_SELF)
    v = PETSc.Mat().createDense((n_rand,n_svals),None,\
                                v[:,:n_svals],comm=MPI.COMM_SELF)
    s = s[:n_svals]
    
    Qfwd.multInPlace(v,0,n_svals)
    Qfwd.setActiveColumns(0,n_svals)
    Qadj.multInPlace(u,0,n_svals)
    Qadj.setActiveColumns(0,n_svals)

    Qfwd_ = Qfwd.getMat()
    Qfwd_mat = Qfwd_.copy()
    Qfwd.restoreMat(Qfwd_)
    Qfwd.destroy()
    Qadj_ = Qadj.getMat()
    Qadj_mat = Qadj_.copy()
    Qadj.restoreMat(Qadj_)
    Qadj.destroy()

    sizes_S = Qfwd_mat.getSizes()[-1]
    S = PETSc.Mat().createDense((sizes_S, sizes_S), comm=lin_op.get_comm())
    S.setUp()
    for i in range (*S.getOwnershipRange()):
        S.setValues(i,i,s[i])
    S.assemble(None)
    return (Qfwd_mat, S, Qadj_mat)