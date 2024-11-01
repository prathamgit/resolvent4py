import numpy as np
import scipy as sp

from petsc4py import PETSc
from mpi4py import MPI
from .miscellaneous import get_mpi_type
from .miscellaneous import petscprint


def compute_local_size(Nglob):
    r"""
        Given the global size :code:`Nglob` of a vector, compute 
        the local size :code:`Nloc` that will be owned by each processor
        in the MPI pool.

        :param Nglob: global size
        :type Nglob: int
        :return: local size
        :rtype: int
    """
    size, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
    Nloc = Nglob//size + 1 if np.mod(Nglob,size) > rank else Nglob//size
    return Nloc

def mat_solve_hermitian_transpose(ksp, X, Y=None):
    r"""
        Solve :math:`A^{-1}X = Z`, where :math:`X` is a PETSc matrix of type
        :code:`PETSc.Mat.Type.DENSE`
        
        :param ksp: a KPS solver structure
        :type ksp: PETSc.KSP
        :param X: a dense PETSc matrix
        :type X: PETSc.Mat.Type.DENSE

        :return: a matrix of the same type and size as :math:`X`
        :rtype: PETSc.Mat.Type.DENSE
    """
    sizes = X.getSizes()
    Yarray = np.zeros((sizes[0][0], sizes[-1][-1]), dtype=np.complex128)
    Y = X.duplicate() if Y == None else Y
    y = X.createVecLeft()
    for i in range (X.getSizes()[-1][-1]):
        x = X.getColumnVector(i)
        x.conjugate()
        ksp.solveTranspose(x, y)
        x.conjugate()
        y.conjugate()
        Yarray[:,i] = y.getArray()
        x.destroy()
    y.destroy()
    offset, _ = Y.getOwnershipRange()
    rows = np.arange(Yarray.shape[0], dtype=np.int32) + offset
    cols = np.arange(Yarray.shape[-1], dtype=np.int32)
    Y.setValues(rows, cols, Yarray.reshape(-1))
    Y.assemble(None)
    return Y

def compute_dense_inverse(comm, M):
    r"""
        Compute :math:`M^{-1}`
        
        :param comm: MPI communicator
        :param M: a dense PETSc matrix
        :type M: PETSc.Mat.Type.DENSE

        :return: the inverse of :math:`M`
        :rtype: PETSc.Mat.Type.DENSE
    """
    # Gather M data on all processors
    M_array = M.getDenseArray().copy().reshape(-1)
    size_vec_local = len(M_array)
    size_vec_global = sum(comm.allgather(size_vec_local))
    size_vec = (size_vec_local, size_vec_global)
    M_vec = PETSc.Vec().createWithArray(M_array, size_vec, None, comm=comm)
    scatter, Mseq_vec = PETSc.Scatter().toAll(M_vec)
    scatter.scatter(M_vec, Mseq_vec, addv=PETSc.InsertMode.INSERT)
    # Compute the inverse using scipy and scatter back to all
    sizes = M.getSizes()
    indices = np.arange(Mseq_vec.getSize())
    Mseq_array = Mseq_vec.getArray().reshape((sizes[0][-1], sizes[0][-1]))
    Mseq_vec.setValues(indices, sp.linalg.inv(Mseq_array).reshape(-1))
    Mseq_vec.assemble()
    scatter.scatter(Mseq_vec, M_vec, addv=PETSc.InsertMode.INSERT,\
                    mode=PETSc.ScatterMode.REVERSE)
    Minv_array = M_vec.getArray().reshape((sizes[0][0], sizes[-1][-1]))
    Minv = PETSc.Mat().createDense(sizes, None, Minv_array, comm=comm)
    return Minv

def compute_matrix_product_contraction(comm, M, P):
    r"""
        Compute :math:`\sum_{i,j} M_{i,j}P_{i,j}`
        
        :param comm: MPI communicator
        :param M: PETSc matrix
        :type M: PETSc.Mat.Type.DENSE
        :param P: PETSc matrix
        :type P: PETSc.Mat.Type.DENSE

        :rtype: PETSc scalar
    """
    M_array = M.getDenseArray()
    P_array = P.getDenseArray()
    value = comm.allreduce(np.sum(M_array*P_array), op=MPI.SUM)
    return value

def compute_trace_product(comm, L1, L2, L2_hermitian_transpose=False):
    r"""
        If :code:`L2_hermitian_transpose==False`, compute 
        :math:`\text{Tr}(L_1 L_2)`, else :math:`\text{Tr}(L_1 L_2^*)`.

        :param comm: MPI communicator
        :param L1: low-rank linear operator
        :param L2: any linear operator
        :param L2_hermitian_transpose: [optional] :code:`True` or :code:`False`
        :type L2_hermitian_transpose: bool

        :rtype: PETSc scalar
    """
    L2_action = L2.apply_mat if L2_hermitian_transpose == False else \
        L2.apply_hermitian_transpose_mat
    F1 = L1.U.matMult(L1.Sigma)
    F2 = L2_action(F1)
    L1.V.hermitianTranspose()
    F3 = L1.V.matMult(F2)
    L1.V.hermitianTranspose()
    F3diag = F3.getDiagonal().getArray()
    trace = comm.allreduce(np.sum(F3diag), op=MPI.SUM)
    F1.destroy()
    F2.destroy()
    F3.destroy()
    return trace

def hermitian_transpose(comm, Mat, in_place=False, MatHT=None):
    r"""
        Return the hermitian transpose of the matrix :code:`Mat`.

        :param comm: MPI communicator
        :param Mat: PETSc matrix of any kind
        :param in_place: [optional] in place transposition if :code:`True` and
            out of place otherwise
        :type in_place: bool
        :param MatHT: [optional] matrix with the correct layout to hold the
            hermitian transpose of :code:`Mat`
        :param Mat: PETSc matrix of any kind
    """
    if in_place == False:
        if MatHT == None:
            sizes = Mat.getSizes()
            MatHT = PETSc.Mat().create(comm)
            MatHT.setType(Mat.getType())
            MatHT.setSizes((sizes[-1], sizes[0]))
            MatHT.setUp()
        Mat.setTransposePrecursor(MatHT)
        Mat.hermitianTranspose(MatHT)
        return MatHT
    else:
        MatHT_ = Mat.hermitianTranspose()
        return MatHT_

def convert_coo_to_csr(comm, arrays, sizes):
    r"""
        Convert arrays = [row indices, col indices, values] for COO matrix 
        assembly to [row pointers, col indices, values] for CSR matrix assembly. 
        (petsc4py currently does not support coo matrix assembly, hence the need 
        to convert.)
        
        :param comm: MPI communicator (only MPI.COMM_WORLD is supported for now)
        :param arrays: a list of numpy arrays (e.g., arrays = [rows,cols,vals])
        :param sizes: matrix size
        :type sizes: `MatSizeSpec <MatSizeSpec_>`_

        :return: csr row pointers, column indices and matrix values for CSR 
            matrix assembly
        :rtype: list
    """

    rank = comm.Get_rank()
    pool = np.arange(comm.Get_size())
    rows, cols, vals = arrays
    idces = np.argsort(rows).reshape(-1)
    rows, cols, vals = rows[idces], cols[idces], vals[idces]

    mat_row_sizes_local = np.asarray(comm.allgather(sizes[0][0]),dtype=np.int32)
    mat_row_displ = np.concatenate(([0],np.cumsum(mat_row_sizes_local[:-1])))
    ownership_ranges = np.zeros((comm.Get_size(),2),dtype=np.int32)
    ownership_ranges[:,0] = mat_row_displ
    ownership_ranges[:-1,1] = ownership_ranges[1:,0]
    ownership_ranges[-1,1] = sizes[0][-1]

    send_rows, send_cols = [], []
    send_vals, lengths = [], []
    for i in pool:
        idces = np.argwhere((rows >= ownership_ranges[i,0]) & \
                            (rows < ownership_ranges[i,1])).reshape(-1)
        lengths.append(np.asarray([len(idces)],dtype=np.int32))
        send_rows.append(rows[idces])
        send_cols.append(cols[idces])
        send_vals.append(vals[idces])
    
    recv_bufs = [np.empty(1, dtype=np.int32) for _ in pool]
    recv_reqs = [comm.Irecv(bf,source=i) for (bf,i) in zip(recv_bufs,pool)]
    send_reqs = [comm.Isend(sz,dest=i) for (i,sz) in enumerate(lengths)]
    MPI.Request.waitall(send_reqs + recv_reqs)
    lengths = [buf[0] for buf in recv_bufs]

    dtypes = [np.int32, np.int32, np.complex128]
    my_arrays = []
    for (j,array) in enumerate([send_rows, send_cols, send_vals]):
        
        dtype = dtypes[j]
        mpi_type = get_mpi_type(np.dtype(dtype))
        recv_bufs = [[np.empty(lengths[i],dtype=dtype), mpi_type] for i in pool]
        recv_reqs = [comm.Irecv(bf,source=i) for (bf,i) in zip(recv_bufs,pool)]
        send_reqs = [comm.Isend(array[i],dest=i) for i in pool]
        MPI.Request.waitall(send_reqs + recv_reqs)
        my_arrays.append([recv_bufs[i][0] for i in pool])

    my_rows, my_cols, my_vals = [], [], []
    for i in pool:
        my_rows.extend(my_arrays[0][i])
        my_cols.extend(my_arrays[1][i])
        my_vals.extend(my_arrays[2][i])

    my_rows = np.asarray(my_rows,dtype=np.int32) - ownership_ranges[rank,0]
    my_cols = np.asarray(my_cols,dtype=np.int32)
    my_vals = np.asarray(my_vals,dtype=np.complex128)

    idces = np.argsort(my_rows).reshape(-1)
    my_rows = my_rows[idces]
    my_cols = my_cols[idces]
    my_vals = my_vals[idces]

    ni = 0
    my_rows_ptr = np.zeros(sizes[0][0]+1,dtype=np.int32)
    for i in range (sizes[0][0]):
        ni += np.count_nonzero(my_rows == i)
        my_rows_ptr[i+1] = ni
    
    return my_rows_ptr, my_cols, my_vals


def enforce_complex_conjugacy(comm, vec, nblocks):
    r"""
        Suppose we have a vector

        .. math::
            v = \left(\ldots,v_{-1},v_{0},v_{1},\ldots\right)

        where :math:`v_i` are complex vectors. This function enforces 
        :math:`v_{-i} = \overline{v_{i}}` for all :math:`i` (this implies that
        :math:`v_0` will be purely real).

        :param vec: vector :math:`v` described above
        :type vec: PETSc.Vec.Type.STANDARD
        :param nblocks: number of vectors :math:`v_i` in :math:`v`. This must 
            be an odd number.
        :type nblocks: int

        :rtype: None
    """
    if np.mod(nblocks,2) == 0:
        raise ValueError (
            "The number of blocks must be an odd number. "
            "Currently you set {nblocks} blocks."
            )
    scatter, vec_seq = PETSc.Scatter().toZero(vec)
    scatter.begin(vec, vec_seq, addv=PETSc.InsertMode.INSERT)
    scatter.end(vec, vec_seq, addv=PETSc.InsertMode.INSERT)
    if comm.Get_rank() == 0:
        array = vec_seq.getArray()
        block_size = len(array)//nblocks
        for i in range (nblocks//2):
            j = nblocks - 1 - i
            i0, i1 = i*block_size, (i+1)*block_size
            j0, j1 = j*block_size, (j+1)*block_size
            array[i0:i1] = array[j0:j1].conj()
        i = nblocks//2
        i0, i1 = i*block_size, (i+1)*block_size
        array[i0:i1] = array[i0:i1].real
        vec_seq.setValues(np.arange(len(array)), array)
        vec_seq.assemble()
    scatter.begin(vec_seq, vec, addv=PETSc.InsertMode.INSERT,\
                    mode=PETSc.ScatterMode.REVERSE)
    scatter.end(vec_seq, vec, addv=PETSc.InsertMode.INSERT,\
                mode=PETSc.ScatterMode.REVERSE)
    scatter.destroy()
    vec_seq.destroy()


def check_complex_conjugacy(comm, vec, nblocks):
    r"""
        Verify whether the components :math:`v_i` of the vector

        .. math::
            v = \left(\ldots,v_{-1},v_{0},v_{1},\ldots\right)

        satisfy :math:`v_{-i} = \overline{v_{i}}` for all :math:`i`.

        :param vec: vector :math:`v` described above
        :type vec: PETSc.Vec.Type.STANDARD
        :param nblocks: number of vectors :math:`v_i` in :math:`v`. This must 
            be an odd number.
        :type nblocks: int

        :return: :code:`True` if the components are complex-conjugates of each
            other and :code:`False` otherwise.
        :rtype: Bool
    """
    if np.mod(nblocks,2) == 0:
        raise ValueError (
            "The number of blocks must be an odd number. "
            "Currently you set {nblocks} blocks."
            )
    scatter, vec_seq = PETSc.Scatter().toZero(vec)
    scatter.begin(vec, vec_seq, addv=PETSc.InsertMode.INSERT)
    scatter.end(vec, vec_seq, addv=PETSc.InsertMode.INSERT)
    cc = None
    if comm.Get_rank() == 0:
        array = vec_seq.getArray()
        block_size = len(array)//nblocks
        array_block = np.zeros(block_size, dtype=np.complex128)
        for i in range (nblocks):
            i0, i1 = i*block_size, (i+1)*block_size
            array_block += array[i0:i1]
        cc = True if np.linalg.norm(array_block.imag) <= 1e-14 else False
    scatter.destroy()
    vec_seq.destroy()
    cc = comm.bcast(cc, root=0)
    return cc


def create_AIJ_identity(comm, sizes):
    r"""
        :param comm: MPI Communicator
        :param sizes: `MatSizeSpec`_

        :return: identity matrix
        :rtype: PETSc.Mat.Type.AIJ
    """
    Id = PETSc.Mat().createConstantDiagonal(sizes, 1.0, comm)
    Id.convert(PETSc.Mat.Type.AIJ)
    return Id