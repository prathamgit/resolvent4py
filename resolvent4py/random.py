from . import np
from . import sp
from . import PETSc
from .comms import scatter_array_from_root_to_all
from .linalg import convert_coo_to_csr

def generate_random_petsc_sparse_matrix(comm, sizes, complex=None):
    r"""
        :param comm: MPI communicator
        :param sizes: matrix sizes
        :type sizes: `MatSizeSpec`_
        :param complex: [optional] set to :code:`True` if you desire a 
            complex-valued matrix
        :type complex: Bool

        :return: a sparse PETSc matrix
        :rtype: PETSc.Mat.Type.AIJ
    """
    rank = comm.Get_rank()
    nrows, ncols = sizes[0][-1], sizes[-1][-1]
    # Generate random matrix on root
    arrays = [None, None, None]
    if rank == 0:
        dtype = np.complex128 if complex == True else np.float64
        A = sp.sparse.random(nrows, ncols, density=0.01, \
                                format='csr', dtype=dtype)
        if nrows == ncols:  # add identity to make A invertible
            A += sp.sparse.identity(nrows, dtype=dtype, format='csr')
        A = A.tocoo()
        arrays = [A.row, A.col, A.data]
    # Scatter to all other processors and assemble
    recv_bufs = [scatter_array_from_root_to_all(comm, a) for a in arrays]
    row_ptrs, cols, data = convert_coo_to_csr(comm, recv_bufs, sizes)
    A = PETSc.Mat().create(comm=comm)
    A.setSizes(sizes)
    A.setUp()
    A.setPreallocationCSR((row_ptrs,cols,data))
    A.setValuesCSR(row_ptrs,cols,data)
    A.assemble(None)
    return A

def generate_random_petsc_vector(comm, sizes, complex=False):
    r"""
        :param comm: MPI communicator
        :param sizes: vector sizes
        :type sizes: `LayoutSizeSpec`_
        :param complex: [optional] set to :code:`True` if you desire a 
            complex-valued vector
        :type complex: Bool

        :return: a PETSc vector
        :rtype: PETSc.Vec.Type.STANDARD
    """
    array = None
    if comm.Get_rank() == 0:
        vec = np.random.randn(sizes[-1])
        array = vec + 1j*np.random.randn(sizes[-1]) if complex else vec
    array = scatter_array_from_root_to_all(comm, array, sizes[0])
    vec = PETSc.Vec().createWithArray(array, sizes, None, comm=comm)
    return vec