from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

from .linalg import convert_coo_to_csr
from .miscellaneous import petscprint
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

# def assemble_sparse_matrix(comm,arrays,sizes):

#     # arrays = [rows,cols,data], where rows, etc. are numpy arrays
#     # sizes = [Nrows,Ncols,number of nnz per row]

#     rank, size = comm.Get_rank(), comm.Get_size()
#     Nr, Nc, nnzpr = sizes
#     Nrloc = compute_local_size(Nr)
#     Ncloc = compute_local_size(Nc)

#     """
#         Gather indices and values to root and convert (row indices, col indices)
#         to CSR (row pointers, col indices)
#     """
#     rows, cols, data, displ = None, None, None, None
#     counts = np.asarray(comm.gather(len(arrays[0]),root=0))
#     if rank == 0:
#         nnztot = np.sum(counts)
#         displ = np.concatenate(([0],np.cumsum(counts[:-1])))
#         rows = np.empty(nnztot,dtype=np.int32)
#         cols = np.empty(nnztot,dtype=np.int32)
#         data = np.empty(nnztot,dtype=np.complex128)

#     comm.Gatherv(arrays[0],[rows,counts,displ,MPI.INT],root=0)
#     comm.Gatherv(arrays[1],[cols,counts,displ,MPI.INT],root=0)
#     comm.Gatherv(arrays[2],[data,counts,displ,MPI.DOUBLE_COMPLEX],root=0)

#     """
#         Order rows in ascending order and distribute row
#         pointers among other processes
#     """
#     Nloc_lst = np.asarray(comm.gather(Nrloc,root=0))
#     Nloc_displ = np.concatenate(([0],np.cumsum(Nloc_lst[:-1]))) if rank == 0 else None
#     if rank == 0:
#         idces = np.argsort(rows)
#         rows, cols, data = rows[idces], cols[idces], data[idces]

#         ni = 0
#         rowsptr = np.zeros(Nr+1,dtype=np.int32)
#         for i in range (Nr):
#             ni += np.count_nonzero(rows[ni:ni+nnzpr] == i)
#             rowsptr[i+1] = ni

#         if rowsptr[-1] != nnztot: raise ValueError ("Total number of nonzeros does not match!")
        
#         for i in range (size):
            
#             ndisp, nloc = Nloc_displ[i], Nloc_lst[i]
#             rowsptr_i = rowsptr[ndisp:ndisp+nloc+1] - rowsptr[ndisp]
#             cols_i = cols[rowsptr[ndisp]:rowsptr[ndisp]+rowsptr_i[-1]]
#             vals_i = data[rowsptr[ndisp]:rowsptr[ndisp]+rowsptr_i[-1]]
#             sendbufs = [rowsptr_i,cols_i,vals_i]

#             if i == 0:
#                 my_rowsptr, my_cols, my_data = sendbufs
#             else:
#                 for j in range (len(sendbufs)):
#                     comm.Send(np.asarray([len(sendbufs[j])],dtype=np.int32),dest=i,tag=0)
#                     comm.Send(sendbufs[j],dest=i,tag=1)

#     else:

#         recvbufs, dtypes = [], [np.int32,np.int32,np.complex128]
#         for j in range (len(dtypes)):
#             bufsz = np.empty(1,dtype=np.int32)
#             comm.Recv(bufsz,source=0,tag=0)
#             recvbuf = np.empty(bufsz[0],dtype=dtypes[j])
#             comm.Recv(recvbuf,source=0,tag=1)
#             recvbufs.append(recvbuf)

#         my_rowsptr, my_cols, my_data = recvbufs
    
#     """
#         Assemble matrix
#     """
#     M = PETSc.Mat().createAIJ([[Nrloc,Nr],[Ncloc,Nc]],comm=MPI.COMM_WORLD)
#     M.setPreallocationCSR((my_rowsptr,my_cols))
#     M.setValuesCSR(my_rowsptr,my_cols,my_data,True)
#     M.assemble(False)

#     return M

def read_dense_matrix(comm, filename, sizes):
    viewer = PETSc.Viewer().createMPIIO(filename,"r",comm=comm)
    M = PETSc.Mat().createDense(sizes, comm=comm)
    M.load(viewer)
    viewer.destroy()
    return M

def write_to_file(comm, filename, object):
    viewer = PETSc.Viewer().createBinary(filename, "w", comm=comm)
    object.view(viewer)
    viewer.destroy()