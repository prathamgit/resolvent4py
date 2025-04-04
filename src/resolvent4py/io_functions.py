from . import PETSc
from . import SLEPc
from . import MPI
from . import np
from . import typing

from .linalg import convert_coo_to_csr
from .linalg import compute_local_size


def read_vector(
    comm: MPI.Comm,
    filename: str,
    sizes: typing.Optional[typing.Tuple[int, int]] = None,
) -> PETSc.Vec:
    r"""
    Read PETSc vector from file

    :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
    :type comm: MPI.Comm
    :param filename: name of the file that holds the vector
    :type filename: str
    :param sizes: :code:`(local size, global size)`
    :type sizes: Optional[Tuple[int, int]]

    :rtype: PETSc.Vec
    """
    viewer = PETSc.Viewer().createMPIIO(filename, "r", comm=comm)
    vec = PETSc.Vec().create(comm)
    vec.setSizes(sizes) if sizes != None else None
    vec.load(viewer)
    viewer.destroy()
    return vec


def read_coo_matrix(
    comm: MPI.Comm,
    filenames: typing.Tuple[str, str, str],
    sizes: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]],
) -> PETSc.Mat:
    r"""
    Read COO matrix from file

    :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
    :type comm: MPI.Comm
    :param filenames: names of the files that hold the rows, columns and values
        of the sparse matrix (e.g., :code:`(rows, cols, values)`)
    :type filenames: Tuple[str, str, str]
    :param sizes: :code:`((local rows, global rows), (local cols, global cols))`
    :type sizes: Tuple[Tuple[int, int], Tuple[int, int]]

    :rtype: PETSc.Mat
    """
    # Read COO vectors
    fname_rows, fname_cols, fname_vals = filenames
    rowsvec = read_vector(comm, fname_rows)
    colsvec = read_vector(comm, fname_cols)
    valsvec = read_vector(comm, fname_vals)
    rows = np.asarray(rowsvec.getArray().real, dtype=np.int32)
    cols = np.asarray(colsvec.getArray().real, dtype=np.int32)
    vals = valsvec.getArray()
    # Delete zeros for efficiencies
    idces = np.argwhere(np.abs(vals) <= 1e-16)
    rows = np.delete(rows, idces)
    cols = np.delete(cols, idces)
    vals = np.delete(vals, idces)
    # Convert COO to CSR and create the sparse matrix
    rows_ptr, cols, vals = convert_coo_to_csr(comm, [rows, cols, vals], sizes)
    M = PETSc.Mat().createAIJ(sizes, comm=comm)
    M.setPreallocationCSR((rows_ptr, cols))
    M.setValuesCSR(rows_ptr, cols, vals, True)
    M.assemble(False)
    rowsvec.destroy()
    colsvec.destroy()
    valsvec.destroy()
    return M


def read_dense_matrix(
    comm: MPI.Comm,
    filename: str,
    sizes: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]],
) -> PETSc.Mat:
    r"""
    Read dense PETSc matrix from file.

    :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
    :type comm: MPI.Comm
    :param filename: name of the file that holds the matrix
    :type filename: str
    :param sizes: :code:`((local rows, global rows), (local cols, global cols))`
    :type sizes: Tuple[Tuple[int, int], Tuple[int, int]]

    :rtype: PETSc.Mat
    """
    viewer = PETSc.Viewer().createMPIIO(filename, "r", comm=comm)
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.load(viewer)
    viewer.destroy()
    return M


def read_bv(
    comm: MPI.Comm,
    filename: str,
    sizes: typing.Tuple[typing.Tuple[int, int], int],
) -> SLEPc.BV:
    r"""
    Read dense matrix from file and store as a SLEPc BV

    :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
    :type comm: MPI.Comm
    :param filename: name of the file that holds the matrix
    :type filename: str
    :param sizes: :code:`((local rows, global rows), global columns)`
    :type sizes: Tuple[Tuple[int, int], int]

    :rtype: SLEPc.BV
    """
    ncols = sizes[-1]
    sizes_mat = (sizes[0], (compute_local_size(ncols), ncols))
    Mm = read_dense_matrix(comm, filename, sizes_mat)
    M = SLEPc.BV().createFromMat(Mm)
    M.setType("mat")
    Mm.destroy()
    return M


def write_to_file(
    comm: MPI.Comm,
    filename: str,
    petsc_object: typing.Union[PETSc.Mat, PETSc.Vec],
) -> None:
    r"""
    Write PETSc object to file.

    :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
    :type comm: MPI.Comm
    :param filename: name of the file to store the object
    :type filename: str
    :param petsc_object: any PETSc matrix or vector
    :type petsc_object: Union[PETSc.Mat, PETSc.Vec]
    """
    viewer = PETSc.Viewer().createBinary(filename, "w", comm=comm)
    petsc_object.view(viewer)
    viewer.destroy()


def write_bv_to_file(
    comm: MPI.Comm, filename: str, object: typing.Union[PETSc.Mat, PETSc.Vec]
) -> None:
    r"""
    Write SLEPc BV to file.

    :param comm: MPI communicator (MPI.COMM_WORLD or MPI.COMM_SELF)
    :type comm: MPI.Comm
    :param filename: name of the file to store the object
    :type filename: str
    :param object: SLEPc BV
    :type object: SLEPc.BV
    """
    mat = object.getMat()
    viewer = PETSc.Viewer().createBinary(filename, "w", comm=comm)
    mat.view(viewer)
    viewer.destroy()
    object.restoreMat(mat)
