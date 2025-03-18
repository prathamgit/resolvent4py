from . import np
from . import MPI
from . import PETSc
from . import tracemalloc

numpy_to_mpi_dtype = {
        np.dtype(np.int32): MPI.INT,
        np.dtype(np.int64): MPI.INT64_T,
        np.dtype(np.float64): MPI.DOUBLE,
        np.dtype(np.complex64): MPI.COMPLEX,
        np.dtype(np.complex128): MPI.DOUBLE_COMPLEX
    }

def get_mpi_type(dtype):
    r"""
        Get the corresponding MPI type for a given numpy data type.

        :param dtype: dtype (e.g., :code:`np.dtype(np.int32)`)
        :rtype: MPI data type (e.g., :code:`MPI.INT`)
    """
    mpi_dtype = numpy_to_mpi_dtype.get(dtype)
    if mpi_dtype is None:
        raise ValueError(f"No MPI type found for NumPy dtype {dtype}")
    return mpi_dtype

def petscprint(comm, arg):
    r"""
        Print to terminal

        :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
        :param arg: argument to be fed into print()
        :type arg: any
    """
    if comm == MPI.COMM_SELF:
        print(arg)
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(arg)

def get_memory_usage():
    r"""
        Compute the used memory (in Mb)
    """
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    total_memory = sum(stat.size for stat in top_stats) / (1024 ** 2)
    values = MPI.COMM_WORLD.allgather(total_memory)
    value = sum(values)
    return value

def copy_mat_from_bv(bv):
    r"""
        Extract a PETSc Mat from a SLEPc BV. This function returns a copy
        of the the data underlying the BV structure.
        :rtype: PETSc.Mat.Type.DENSE
    """
    bv_mat = bv.getMat()
    mat = bv_mat.copy()
    bv.restoreMat(bv_mat)
    return mat

def create_dense_matrix(comm, sizes):
    r"""
        Create dense matrix

        :param comm: MPI communicator
        :param sizes: `MatSizeSpec`_
        
        :rtype: PETSc.Mat.Type.DENSE
    """
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.setUp()
    return M