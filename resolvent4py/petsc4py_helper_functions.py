import numpy as np
import scipy as sp
import pickle
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

def petscprint(
    comm,
    arg
):
    r"""
        Print to terminal

        :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
        :param string: argument to be fed into print()
        :type string: any
    """
    if comm == MPI.COMM_SELF:
        print(arg)
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(arg)

def compute_local_size(
    Nglob
):
    r"""
        Compute local size given the global size

        :param Nglob: global size
        :type Nglob: int

        :return: local size
        :rtype: int
    """
    
    size, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
    Nloc = Nglob//size + 1 if np.mod(Nglob,size) > rank else Nglob//size

    return Nloc

def convert_coo_to_csr(
    comm,
    arrays,
    sizes
):
    r"""
        Convert arrays = [row indices, col indices, values] for COO matrix 
        assembly to [row pointers, col indices, values] for CSR matrix assembly. 
        (petsc4py currently does not support coo matrix assembly, hence the need 
        to convert.)
        
        :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
        :param arrays: a list of numpy arrays (e.g., arrays = [rows,cols,vals])
        :param sizes: matrix size
        :type sizes: `MatSizeSpec <matSizeSpec_>`_

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

def mat_solve_hermitian_transpose(
        ksp,
        X
):
    r"""
        Solve :math:`A^{-1}X = Y`, where :math:`X` is a dense PETSc matrix.
        
        :param ksp: a KPS solver structure
        :type ksp: PETSc.KSP
        :param X: a dense PETSc matrix
        :type X: PETSc.Mat.Type.DENSE

        :return: a matrix of the same type and size as :math:`X`
        :rtype: PETSc.Mat.Type.DENSE
    """

    Y = SLEPc.BV().createFromMat(X)
    Y.setType('mat')
    n = Y.getSizes()[-1]
    for i in range (n):
        y = Y.getColumn(i)
        y.conjugate()
        ksp.solveTranspose(y,y)
        y.conjugate()
        Y.restoreColumn(i,y)
    
    Z = Y.getMat().copy()
    Y.destroy()

    return Z

def compute_dense_inverse(
    comm,
    M
):
    r"""
        Compute :math:`M^{-1}`
        
        :param ksp: a KPS solver structure
        :type ksp: PETSc.KSP
        :param X: a dense PETSc matrix
        :type X: PETSc.Mat.Type.DENSE

        :return: the inverse of :math:`M`
        :rtype: PETSc.Mat.Type.DENSE
    """

    
    # Gather data in M on all processors
    sizes = M.getSizes()
    M_array = M.getDenseArray().reshape(-1)
    size_vec_local = len(M_array)
    size_vec_global = sum(comm.allgather(size_vec_local))
    M_vec = PETSc.Vec().createWithArray(M_array,\
                                        (size_vec_local,size_vec_global),
                                        None,comm=comm)
    scatter, M_vec_seq = PETSc.Scatter().toAll(M_vec)
    scatter.scatter(M_vec,M_vec_seq,addv=PETSc.InsertMode.INSERT)

    # Compute the inverse using scipy and scatter back to all
    M_array = M_vec_seq.getArray().reshape((sizes[0][-1],sizes[0][-1]))
    M_vec_seq.setValues(np.arange(M_vec_seq.getSize()),\
                        sp.linalg.inv(M_array).reshape(-1))
    M_vec_seq.assemble()
    scatter.scatter(M_vec_seq,M_vec,addv=PETSc.InsertMode.INSERT,\
                    mode=PETSc.ScatterMode.REVERSE)
    
    M_array = M_vec.getArray().reshape((sizes[0][0],sizes[-1][-1]))
    X = PETSc.Mat().createDense(sizes,None,M_array,comm=comm)

    return X



