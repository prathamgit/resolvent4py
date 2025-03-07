from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc

from ..linalg import enforce_complex_conjugacy
from ..miscellaneous import create_dense_matrix
from ..miscellaneous import petscprint

# def randomized_svd(lin_op, lin_op_action, n_rand, n_loops, n_svals):
#     r"""
#         Compute the SVD of the linear operator :math:`L` 
#         specified by :code:`lin_op` using a randomized SVD algorithm.
        
#         :param lin_op: any child class of the :code:`LinearOperator` class
#         :param lin_op_action: one of :code:`lin_op.apply_mat` or 
#             :code:`lin_op.solve_mat`
#         :param n_rand: number of random vectors to use
#         :type n_rand: int
#         :param n_loops: number of randomized svd iterations
#         :type n_loops: int
#         :param n_svals: number of singular triplets to return
#         :type n_svals: int

#         :return: :math:`(U,\,\Sigma,\, V)` a 3-tuple with the leading 
#             :code:`n_svals` singular values and corresponding left and \
#             right singular vectors
#         :rtype: (SLEPc.BV with :code:`n_svals` columns, 
#             numpy.ndarray of size :code:`n_svals x n_svals`, 
#             SLEPc.BV with :code:`n_svals` columns)
#     """
#     if lin_op_action == lin_op.apply_mat:
#         lin_op_action_adj = lin_op.apply_hermitian_transpose_mat
#     if lin_op_action == lin_op.solve_mat:
#         lin_op_action_adj = lin_op.solve_hermitian_transpose_mat
#     # Assemble random BV
#     rowsizes = lin_op.get_dimensions()[0]
#     X = SLEPc.BV().create(comm=lin_op._comm)
#     X.setSizes(rowsizes, n_rand)
#     X.setType('mat')
#     X.setRandomNormal()
#     for j in range (n_rand):
#         xj = X.getColumn(j)
#         if lin_op._real:
#             row_offset = xj.getOwnershipRange()[0]
#             rows = np.arange(rowsizes[0], dtype=np.int64) + row_offset
#             array = xj.getArray()
#             xj.setValues(rows, array.real)
#             xj.assemble()
#         if lin_op._block_cc:
#             enforce_complex_conjugacy(lin_op._comm, xj, lin_op._nblocks)
#         X.restoreColumn(j, xj)
#     X.orthogonalize(None)
#     # Perform randomized SVD loop
#     Qadj = X.duplicate()
#     lin_op_action_adj(X, Qadj)
#     Qadj.orthogonalize(None)
#     X.destroy()
#     Qfwd = Qadj.duplicate()
#     R = create_dense_matrix(MPI.COMM_SELF, (n_rand, n_rand))
#     for j in range (n_loops):
#         lin_op_action(Qadj, Qfwd)
#         Qfwd.orthogonalize(None)
#         lin_op_action_adj(Qfwd, Qadj)
#         Qadj.orthogonalize(R)
#     # Compute low-rank SVD
#     u, s, v = sp.linalg.svd(R.getDenseArray())
#     v = v.conj().T
#     s = s[:n_svals]
#     u = u[:,:n_svals]
#     v = v[:,:n_svals]
#     u = PETSc.Mat().createDense((n_rand, n_svals), None, u, comm=MPI.COMM_SELF)
#     v = PETSc.Mat().createDense((n_rand, n_svals), None, v, comm=MPI.COMM_SELF)
#     Qfwd.multInPlace(v, 0, n_svals)
#     Qfwd.setActiveColumns(0, n_svals)
#     Qfwd.resize(n_svals, copy=True)
#     Qadj.multInPlace(u, 0, n_svals)
#     Qadj.setActiveColumns(0, n_svals)
#     Qadj.resize(n_svals, copy=True)
#     return (Qfwd, np.diag(s), Qadj)


def randomized_svd(lin_op, lin_op_action, n_rand, n_loops, n_svals):
    r"""
        Compute the SVD of the linear operator :math:`L` 
        specified by :code:`lin_op` using a randomized SVD algorithm.
        
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
        :rtype: (SLEPc.BV with :code:`n_svals` columns, 
            numpy.ndarray of size :code:`n_svals x n_svals`, 
            SLEPc.BV with :code:`n_svals` columns)
    """
    if lin_op_action != lin_op.apply_mat and lin_op_action != lin_op.solve_mat:
        raise ValueError (
            f"lin_op_action must be lin_op.apply_mat or lin_op.solve_mat."
        )
    if lin_op_action == lin_op.apply_mat:
        lin_op_action_adj = lin_op.apply_hermitian_transpose_mat
    if lin_op_action == lin_op.solve_mat:
        lin_op_action_adj = lin_op.solve_hermitian_transpose_mat
    # Assemble random BV (this will be multiplied against L^*)
    rowsizes = lin_op.get_dimensions()[0]
    X = SLEPc.BV().create(comm=lin_op._comm)
    X.setSizes(rowsizes, n_rand)
    X.setType('mat')
    X.setRandomNormal()
    for j in range (n_rand):
        xj = X.getColumn(j)
        if lin_op._real:
            row_offset = xj.getOwnershipRange()[0]
            rows = np.arange(rowsizes[0], dtype=np.int64) + row_offset
            array = xj.getArray()
            xj.setValues(rows, array.real)
            xj.assemble()
        if lin_op._block_cc:
            enforce_complex_conjugacy(lin_op._comm, xj, lin_op._nblocks)
        X.restoreColumn(j, xj)
    X.orthogonalize(None)
    # Perform randomized SVD loop
    Qadj = SLEPc.BV().create(comm=lin_op._comm)
    Qadj.setSizes(lin_op.get_dimensions()[-1], n_rand)
    Qadj.setType('mat')
    Qfwd = SLEPc.BV().create(comm=lin_op._comm)
    Qfwd.setSizes(lin_op.get_dimensions()[0], n_rand)
    Qfwd.setType('mat')
    lin_op_action_adj(X, Qadj)
    Qadj.orthogonalize(None)
    R = create_dense_matrix(MPI.COMM_SELF, (n_rand, n_rand))
    for j in range (n_loops):
        lin_op_action(Qadj, Qfwd)
        Qfwd.orthogonalize(None)
        lin_op_action_adj(Qfwd, Qadj)
        Qadj.orthogonalize(R)
    # Compute low-rank SVD
    u, s, v = sp.linalg.svd(R.getDenseArray())
    v = v.conj().T
    s = s[:n_svals]
    u = u[:,:n_svals]
    v = v[:,:n_svals]
    u = PETSc.Mat().createDense((n_rand, n_svals), None, u, comm=MPI.COMM_SELF)
    v = PETSc.Mat().createDense((n_rand, n_svals), None, v, comm=MPI.COMM_SELF)
    Qfwd.multInPlace(v, 0, n_svals)
    Qfwd.setActiveColumns(0, n_svals)
    Qfwd.resize(n_svals, copy=True)
    Qadj.multInPlace(u, 0, n_svals)
    Qadj.setActiveColumns(0, n_svals)
    Qadj.resize(n_svals, copy=True)
    return (Qfwd, np.diag(s), Qadj)