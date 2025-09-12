__all__ = [
    "create_dense_matrix",
    "create_AIJ_identity",
    "mat_solve_hermitian_transpose",
    "hermitian_transpose",
    "convert_coo_to_csr",
    "assemble_harmonic_resolvent_generator",
]

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from typing import Optional

from .miscellaneous import get_mpi_type


def create_dense_matrix(
    comm: PETSc.Comm, sizes: tuple[tuple[int, int], tuple[int, int]]
) -> PETSc.Mat:
    r"""
    Create dense matrix

    :param comm: PETSc communicator
    :type comm: PETSc.Comm
    :param sizes: tuple[tuple[int, int], tuple[int, int]]

    :rtype: PETSc.Mat.Type.DENSE
    """
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.setUp()
    return M


def create_AIJ_identity(
    comm: PETSc.Comm, sizes: tuple[tuple[int, int], tuple[int, int]]
) -> PETSc.Mat:
    r"""
    Create identity matrix of sparse AIJ type

    :param comm: MPI Communicator
    :type comm: PETSc.Comm
    :param sizes: see `MatSizeSpec <MatSizeSpec_>`_
    :type sizes: tuple[tuple[int, int], tuple[int, int]]

    :return: identity matrix
    :rtype: PETSc.Mat.Type.AIJ
    """
    Id = PETSc.Mat().createConstantDiagonal(sizes, 1.0, comm)
    Id.convert(PETSc.Mat.Type.AIJ)
    return Id


def mat_solve_hermitian_transpose(
    ksp: PETSc.KSP, X: PETSc.Mat, Y: Optional[PETSc.Mat] = None
) -> PETSc.Mat:
    r"""
    Solve :math:`A^{-*}X = Y`, where :math:`X` is a PETSc matrix of type
    :code:`PETSc.Mat.Type.DENSE`

    :param ksp: a KPS solver structure
    :type ksp: PETSc.KSP
    :param X: a dense PETSc matrix
    :type X: PETSc.Mat.Type.DENSE
    :param Y: a dense PETSc matrix
    :type Y: Optional[PETSc.Mat.Type.DENSE] defaults to :code:`None`

    :return: matrix to store the result
    :rtype: PETSc.Mat.Type.DENSE
    """
    sizes = X.getSizes()
    Yarray = np.zeros((sizes[0][0], sizes[-1][-1]), dtype=np.complex128)
    Y = X.duplicate() if Y == None else Y
    y = X.createVecLeft()
    for i in range(X.getSizes()[-1][-1]):
        x = X.getColumnVector(i)
        x.conjugate()
        ksp.solveTranspose(x, y)
        x.conjugate()
        y.conjugate()
        Yarray[:, i] = y.getArray()
        x.destroy()
    y.destroy()
    offset, _ = Y.getOwnershipRange()
    rows = np.arange(Yarray.shape[0], dtype=PETSc.IntType) + offset
    cols = np.arange(Yarray.shape[-1], dtype=PETSc.IntType)
    Y.setValues(rows, cols, Yarray.reshape(-1))
    Y.assemble(None)
    return Y


def hermitian_transpose(
    Mat: PETSc.Mat, in_place=False, MatHT=None
) -> PETSc.Mat:
    r"""
    Return the hermitian transpose of the matrix :code:`Mat`.

    :param Mat: PETSc matrix
    :type Mat: PETSc.Mat
    :param in_place: in-place transposition if :code:`True` and
        out of place otherwise
    :type in_place: Optional[bool] defaults to :code:`False`
    :param MatHT: [optional] matrix with the correct layout to hold the
        hermitian transpose of :code:`Mat`
    :param MatHT: Optional[PETSc.Mat] defaults to :code:`None`
    """
    if in_place == False:
        if MatHT == None:
            sizes = Mat.getSizes()
            MatHT = PETSc.Mat().create(comm=Mat.getComm())
            MatHT.setType(Mat.getType())
            MatHT.setSizes((sizes[-1], sizes[0]))
            MatHT.setUp()
        Mat.setTransposePrecursor(MatHT)
        Mat.hermitianTranspose(MatHT)
        return MatHT
    else:
        MatHT_ = Mat.hermitianTranspose()
        return MatHT_


def convert_coo_to_csr(arrays, sizes):
    """
    arrays: (rows, cols, vals) global COO triplets on each rank (could be empty on most ranks).
    sizes: ((n_local_rows, n_global_rows), (n_local_cols, n_global_cols))
           Same layout as you had; only row sizes are needed for ownership.
    Returns local CSR for this rank:
        row_ptr (len = n_local_rows+1), col_idx (len = nnz_local), vals_local
    """
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    rows, cols, vals = arrays
    rows = np.asarray(rows, dtype=PETSc.IntType, order='C')
    cols = np.asarray(cols, dtype=PETSc.IntType, order='C')
    vals = np.asarray(vals, dtype=PETSc.ScalarType, order='C')

    # ---- 1) Compute global ownership ranges for rows (block row distribution) ----
    # sizes[0][0] is this rank's local row count; allgather to get all
    local_nrows = np.asarray(sizes[0][0], dtype=PETSc.IntType)
    all_local   = np.asarray(comm.allgather(int(local_nrows)), dtype=PETSc.IntType)
    row_starts  = np.concatenate(([0], np.cumsum(all_local[:-1], dtype=PETSc.IntType)))
    row_ends    = row_starts + all_local  # exclusive
    my_row0     = row_starts[rank]
    my_nrows    = int(all_local[rank])

    # ---- 2) Decide owner rank for each nonzero by its global row ----
    # owner = smallest r such that rows < row_ends[r]
    owners = np.searchsorted(row_ends, rows, side='right').astype(PETSc.IntType)

    # ---- 3) Group entries by owner (stable) so each destination gets one contiguous slice ----
    # order by owners; no need to sort by (row,col) yet
    order = np.argsort(owners, kind='stable')
    owners = owners[order]
    rows   = rows[order]
    cols   = cols[order]
    vals   = vals[order]

    sendcounts = np.bincount(owners, minlength=size).astype(np.int32)
    sdispls    = np.concatenate(([0], np.cumsum(sendcounts[:-1], dtype=np.int64))).astype(np.int64)

    # ---- 4) Exchange counts to know recv sizes; then Alltoallv the triplets ----
    recvcounts = np.asarray(comm.alltoall(sendcounts.tolist()), dtype=np.int32)
    rdispls    = np.concatenate(([0], np.cumsum(recvcounts[:-1], dtype=np.int64))).astype(np.int64)
    total_recv = int(recvcounts.sum())

    r_rows = np.empty(total_recv, dtype=PETSc.IntType)
    r_cols = np.empty(total_recv, dtype=PETSc.IntType)
    r_vals = np.empty(total_recv, dtype=PETSc.ScalarType)

    # mpi4py maps numpy dtype -> MPI datatype automatically
    comm.Alltoallv([rows, (sendcounts, sdispls)],  [r_rows, (recvcounts, rdispls)])
    comm.Alltoallv([cols, (sendcounts, sdispls)],  [r_cols, (recvcounts, rdispls)])
    comm.Alltoallv([vals, (sendcounts, sdispls)],  [r_vals, (recvcounts, rdispls)])

    # ---- 5) Convert to local row IDs and build CSR pointers with bincount ----
    if total_recv == 0:
        row_ptr = np.zeros(my_nrows + 1, dtype=PETSc.IntType)
        return row_ptr, r_cols, r_vals

    # make local rows start at 0
    r_rows -= my_row0

    # (Optional) sort locally by (row, col) if needed
    # Keeping entries per row grouped is enough for CSR; sort by col if you want monotone col_idx
    sort_idx = np.lexsort((r_cols, r_rows))
    r_rows = r_rows[sort_idx]
    r_cols = r_cols[sort_idx]
    r_vals = r_vals[sort_idx]

    # (Optional) combine duplicates (same row & col)
    same = (np.diff(r_rows, prepend=-1) == 0) & (np.diff(r_cols, prepend=-1) == 0)
    if same.any():
        # reduce by segments
        new_flags = ~same
        seg_ids   = np.cumsum(new_flags) - 1
        # sum vals per segment
        vals_sum = np.add.reduceat(r_vals, np.flatnonzero(new_flags))
        rows_u   = r_rows[new_flags]
        cols_u   = r_cols[new_flags]
        r_rows, r_cols, r_vals = rows_u, cols_u, vals_sum

    # Row pointer via bincount (length = my_nrows)
    counts = np.bincount(r_rows, minlength=my_nrows)
    row_ptr = np.empty(my_nrows + 1, dtype=PETSc.IntType)
    row_ptr[0] = 0
    np.cumsum(counts, out=row_ptr[1:])

    return row_ptr, r_cols, r_vals


def assemble_harmonic_resolvent_generator(
    A: PETSc.Mat, freqs: np.array
) -> PETSc.Mat:
    r"""
    Assemble :math:`T = -M + A`, where :math:`A` is the output of
    :func:`resolvent4py.utils.io.read_harmonic_balanced_matrix`
    and :math:`M` is a block
    diagonal matrix with block :math:`k` given by :math:`M_k = i k \omega I`
    and :math:`k\omega` is the :math:`k` th entry of :code:`freqs`.

    :param A: assembled PETSc matrix
    :type A: PETSc.Mat
    :param freqs: array :math:`\omega\left(\ldots, -1, 0, 1, \ldots\right)`
    :type freqs: np.array

    :rtype: PETSc.Mat
    """
    rows_lst = []
    vals_lst = []

    rows = np.arange(*A.getOwnershipRange())
    N = A.getSizes()[0][-1] // len(freqs)
    for i in range(len(freqs)):
        idces = np.intersect1d(rows, np.arange(N * i, N * (i + 1)))
        if len(idces) > 0:
            rows_lst.extend(idces)
            vals_lst.extend(-1j * freqs[i] * np.ones(len(idces)))

    rows = np.asarray(rows_lst, dtype=PETSc.IntType)
    vals = np.asarray(vals_lst, dtype=np.complex128)

    rows_ptr, cols, vals = convert_coo_to_csr([rows, rows, vals], A.getSizes())
    M = PETSc.Mat().create(A.getComm())
    M.setSizes(A.getSizes())
    M.setPreallocationCSR((rows_ptr, cols))
    M.setValuesCSR(rows_ptr, cols, vals, True)
    M.assemble(False)
    M.axpy(1.0, A)
    return M
