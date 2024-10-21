import numpy as np
import scipy as sp

from mpi4py import MPI
from petsc4py import PETSc
from .miscellaneous import get_mpi_type

def sequential_to_distributed_matrix(Mat_seq, Mat_dist):
    r"""
        Populate a distributed dense matrix from a sequential dense matrix

        :param Mat_seq: a sequential dense matrix
        :type Mat_seq: PETSc.Mat.Type.DENSE
        :param Mat_dist: a distributed dense matrix
        :type Mat_dist: PETSc.Mat.Type.DENSE

        :rtype: PETSc.Mat.Type.DENSE
    """
    array = Mat_seq.getDenseArray()
    r0, r1 = Mat_dist.getOwnershipRange()
    rows = np.arange(r0,r1)
    cols = np.arange(0,Mat_seq.getSizes()[-1][-1])
    Mat_dist.setValues(rows,cols,array[r0:r1,].reshape(-1))
    Mat_dist.assemble(None)

def distributed_to_sequential_matrix(comm, Mat_dist):
    r"""
        Populate a sequential dense matrix from a distributed dense matrix

        :param comm: MPI communicator
        :param Mat_dist: a distributed dense matrix
        :type Mat_dist: PETSc.Mat.Type.DENSE

        :rtype: PETSc.Mat.Type.DENSE
    """
    array = Mat_dist.getDenseArray().copy().reshape(-1)
    counts = comm.allgather(len(array))
    disps = np.concatenate(([0],np.cumsum(counts[:-1])))
    recvbuf = np.zeros(np.sum(counts), dtype=array.dtype)
    comm.Allgatherv(array, (recvbuf, counts, disps, get_mpi_type(array.dtype)))
    sizes = Mat_dist.getSizes()
    nr, nc = sizes[0][-1], sizes[-1][-1]
    recvbuf = recvbuf.reshape((nr,nc))
    Mat_seq = PETSc.Mat().createDense((nr,nc), None, recvbuf, MPI.COMM_SELF)
    return Mat_seq

def distributed_to_sequential_matrix(comm, Mat_dist):
    r"""
        Populate a sequential dense matrix from a distributed dense matrix

        :param comm: MPI communicator
        :param Mat_dist: a distributed dense matrix
        :type Mat_dist: PETSc.Mat

        :rtype: PETSc.Mat
    """
    array = Mat_dist.getDenseArray().reshape(-1)
    counts = comm.allgather(len(array))
    disps = np.concatenate(([0],np.cumsum(counts[:-1])))
    recvbuf = np.zeros(np.sum(counts), dtype=array.dtype)
    comm.Allgatherv(array, (recvbuf, counts, disps, get_mpi_type(array.dtype)))
    sizes = Mat_dist.getSizes()
    nr, nc = sizes[0][-1], sizes[-1][-1]
    recvbuf = recvbuf.reshape((nr,nc))
    Mat_seq = PETSc.Mat().createDense((nr,nc), None, recvbuf, MPI.COMM_SELF)
    return Mat_seq

def scatter_array_from_root_to_all(comm, array, locsize=None):
    r"""
        Scatter numpy array from root to all other processors

        :param array: numpy array to be scattered from root to the rest of
            the MPI pool
        :type array: numpy.array
        :param locsize: local size owned by each processor. If :code:`None`
            :code:`locsize` is computed using the same logic as in the
            :code:`resolvent4py.linalg.compute_local_size()` routine

        :return: scattered array
        :rtype: numpy.array
    """
    size, rank = comm.Get_size(), comm.Get_rank()
    counts, displs = None, None
    if locsize == None:
        if rank == 0:
            n = len(array)
            counts = np.asarray([n//size + 1 if np.mod(n,size) > j \
                                else n//size for j in range (size)])
            displs = np.concatenate(([0],np.cumsum(counts[:-1])))
            count = counts[0]
            dtype = array.dtype
            for j in range (1,size):
                comm.send(counts[rank], dest=j, tag=0)
                comm.send(dtype, dest=j, tag=1)
        else:
            count = comm.recv(source=0, tag=0)
            dtype = comm.recv(source=0, tag=1)
    else:
        count = locsize
        counts = comm.gather(locsize, root=0)
        if rank == 0:
            dtype = array.dtype
            displs = np.concatenate(([0],np.cumsum(counts[:-1])))
            for j in range (1,size):
                comm.send(dtype, dest=j, tag=1)
        else:
            dtype = comm.recv(source=0, tag=1)
    recvbuf = np.empty(count, dtype=dtype)
    comm.Scatterv([array, counts, displs, get_mpi_type(dtype)], recvbuf, root=0)
    return recvbuf