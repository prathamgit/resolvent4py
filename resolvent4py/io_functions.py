from . import PETSc
from . import SLEPc
from . import np

from .linalg import convert_coo_to_csr
from .linalg import compute_local_size

def read_vector(comm, filename, sizes=None):
    r"""
        Read PETSc Vec from file

        :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
        :param filename: name of the file that holds the vector
        :type filename: str
        :param sizes: [optional] (local row size, global row size)
        :type sizes: `VecSizeSpec`_
        
        :return: vector with the data read from the file
        :rtype: PETSc.Vec
    """
    viewer = PETSc.Viewer().createMPIIO(filename, "r", comm=comm)
    vec = PETSc.Vec().create(comm)
    vec.setSizes(sizes) if sizes != None else None
    vec.load(viewer)
    viewer.destroy()
    return vec

def read_coo_matrix(comm, filenames, sizes):
    r"""
        Read COO matrix from file

        :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
        :param filenames: 3-tuple with the filenames of the row indices, 
            column indices and values
        
        :return: sparse matrix of AIJ type
        :rtype: PETSc.Mat
    """
    fname_rows, fname_cols, fname_vals = filenames
    rowsvec = read_vector(comm,fname_rows)
    rows = np.asarray(rowsvec.getArray().real,dtype=np.int32)
    colsvec = read_vector(comm,fname_cols)
    cols = np.asarray(colsvec.getArray().real,dtype=np.int32)
    valsvec = read_vector(comm,fname_vals)
    vals = valsvec.getArray()
    idces = np.argwhere(np.abs(vals) <= 1e-16)
    rows = np.delete(rows, idces)
    cols = np.delete(cols, idces)
    vals = np.delete(vals, idces)
    rows_ptr, cols, vals = convert_coo_to_csr(comm,[rows,cols,vals],sizes)
    M = PETSc.Mat().createAIJ(sizes, comm=comm)
    M.setPreallocationCSR((rows_ptr,cols))
    M.setValuesCSR(rows_ptr,cols,vals,True)
    M.assemble(False)
    rowsvec.destroy()
    colsvec.destroy()
    valsvec.destroy()
    return M

def read_dense_matrix(comm, filename, sizes):
    viewer = PETSc.Viewer().createMPIIO(filename,"r",comm=comm)
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.load(viewer)
    viewer.destroy()
    return M

def read_bv(comm, filename, sizes):
    ncols = sizes[-1]
    sizes_mat = (sizes[0], (compute_local_size(ncols), ncols))
    Mm = read_dense_matrix(comm, filename, sizes_mat)
    M = SLEPc.BV().createFromMat(Mm)
    M.setType('mat')
    Mm.destroy()
    return M


def write_to_file(comm, filename, object):
    viewer = PETSc.Viewer().createBinary(filename, "w", comm=comm)
    object.view(viewer)
    viewer.destroy()

def write_bv_to_file(comm, filename, object):
    mat = object.getMat()
    viewer = PETSc.Viewer().createBinary(filename, "w", comm=comm)
    mat.view(viewer)
    viewer.destroy()
    object.restoreMat(mat)