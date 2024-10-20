from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import pickle

from .petsc4py_helper_functions import convert_coo_to_csr

def read_vector(comm, filename):
    r"""
        Read PETSc Vec from file

        :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
        :param filename: name of the file that holds the vector
        :type filename: str
        
        :return: vector with the data read from the file
        :rtype: PETSc.Vec
    """
    viewer = PETSc.Viewer().createBinary(filename,"r",comm=comm)
    vec = PETSc.Vec().create(comm)
    vec.load(viewer)
    viewer.destroy()
    return vec

def read_coo_matrix(comm, filenames, sizes):
    fname_rows, fname_cols, fname_vals = filenames
    rows = np.asarray(read_vector(comm,fname_rows).getArray().real,\
                      dtype=np.int64)
    cols = np.asarray(read_vector(comm,fname_cols).getArray().real,\
                      dtype=np.int64)
    vals = read_vector(comm,fname_vals).getArray()
    rows_ptr, cols, vals = convert_coo_to_csr(comm,[rows,cols,vals],sizes)
    M = PETSc.Mat().createAIJ(sizes, comm=comm)
    M.setPreallocationCSR((rows_ptr,cols))
    M.setValuesCSR(rows_ptr,cols,vals,True)
    M.assemble(False)
    return M

def read_dense_matrix(comm, filename, sizes):
    viewer = PETSc.Viewer().createBinary(filename,"r",comm=comm)
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.load(viewer)
    viewer.destroy()
    return M

def write_to_file(comm, filename, object):
    viewer = PETSc.Viewer().createBinary(filename, "w", comm=comm)
    object.view(viewer)
    viewer.destroy()