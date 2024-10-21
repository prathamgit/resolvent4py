import numpy as np
import scipy as sp
import pickle

from slepc4py import SLEPc
from petsc4py import PETSc
from mpi4py import MPI
from .miscellaneous import copy_mat_from_bv

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

def mat_solve_hermitian_transpose(ksp, X):
    r"""
        Solve :math:`A^{-1}X = Y`, where :math:`X` is a PETSc matrix of type
        :code:`PETSc.Mat.Type.DENSE`
        
        :param ksp: a KPS solver structure
        :type ksp: PETSc.KSP
        :param X: a dense PETSc matrix
        :type X: PETSc.Mat.Type.DENSE

        :return: a matrix of the same type and size as :math:`X`
        :rtype: PETSc.Mat.Type.DENSE
    """
    Y = SLEPc.BV().createFromMat(X)
    Y.setType(SLEPc.BV.Type.MAT)
    for i in range (Y.getSizes()[-1]):
        y = Y.getColumn(i)
        y.conjugate()
        ksp.solveTranspose(y,y)
        y.conjugate()
        Y.restoreColumn(i,y)
    Z = copy_mat_from_bv(Y)
    Y.destroy()
    return Z

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
    pool_rank, pool_size = comm.Get_rank(), comm.Get_size()
    rows, cols, vals = arrays
    idces = np.argsort(rows).reshape(-1)
    rows, cols, vals = rows[idces], cols[idces], vals[idces]

    mat_row_sizes_local = np.asarray(comm.allgather(sizes[0][0]),dtype=np.int64)
    mat_row_displ = np.concatenate(([0],np.cumsum(mat_row_sizes_local[:-1])))
    ownership_ranges = np.zeros((comm.Get_size(),2),dtype=np.int64)
    ownership_ranges[:,0] = mat_row_displ
    ownership_ranges[:-1,1] = ownership_ranges[1:,0]
    ownership_ranges[-1,1] = sizes[0][-1]

    my_rows, my_cols, my_vals = [], [], []
    for i in range (pool_size):
        idces = np.argwhere((rows >= ownership_ranges[i,0]) & \
                            (rows < ownership_ranges[i,1])).reshape(-1)
        rows_i, cols_i, vals_i = rows[idces], cols[idces], vals[idces]
        if i != pool_rank:
            comm.send(pickle.dumps([rows_i, cols_i, vals_i]),dest=i)
            rows_i, cols_i, vals_i = pickle.loads(comm.recv(source=i))
        my_rows.extend(rows_i)
        my_cols.extend(cols_i)
        my_vals.extend(vals_i)

    my_rows = np.asarray(my_rows,dtype=np.int64) - ownership_ranges[pool_rank,0]
    my_cols = np.asarray(my_cols,dtype=np.int64)
    my_vals = np.asarray(my_vals,dtype=np.complex128)

    _, rows_counts = np.unique(my_rows,return_counts=True)
    my_rows_ptr = np.concatenate(([0],np.cumsum(rows_counts)))

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