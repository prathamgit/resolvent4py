import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

def commprint(comm,content):
    """
        Print to terminal from rank 0
    """
    if comm.Get_rank() == 0: print(content)

def compute_local_size(comm,Nglob):
    """
        Compute local size given global size
    """
    size, rank = comm.Get_size(), comm.Get_rank()
    Nloc = Nglob//size + 1 if np.mod(Nglob,size) > rank else Nglob//size

    return Nloc

def read_vector(comm,fname,*argv):
    """
        Read vector from file. You can pass an existing
        vector with the appropriate MPI structure as an
        input to the function
    """
    viewer = PETSc.Viewer().createBinary(fname,"r",comm=comm)
    vec = argv[0] if len(argv) > 0 else PETSc.Vec().create(comm)
    vec.load(viewer)
    viewer.destroy()

    return vec

def read_dense_matrix(comm,fname,sizes):
    """
        Read matrix from file
    """
    viewer = PETSc.Viewer().createBinary(fname,"r",comm=comm)
    Mat = PETSc.Mat().createDense(sizes,comm=comm)
    Mat.load(viewer)
    viewer.destroy()

    return Mat

def save_mat(comm,fname,Mat):
    """
        Save dense matrix to file
    """
    viewer = PETSc.Viewer().createBinary(fname,"w",comm=comm)
    Mat.view(viewer)
    viewer.destroy()

def save_vector(comm,fname,vec):
    """
        Save vector to file. You can pass an existing
        vector with the appropriate MPI structure as an
        input to the function
    """
    viewer = PETSc.Viewer().createBinary(fname,"w",comm=comm)
    vec.view(viewer)
    viewer.destroy()


def read_vectors_into_bv(comm,fname_root,idces,sizes):
    """
        Read len(idces) vectors into a SLEPc BV structure
    """
    bv = SLEPc.BV().create(comm=comm)
    bv.setSizes(sizes,len(idces))
    bv.setFromOptions()

    vec = bv.createVec()
    for (k,idx) in enumerate(idces):
        vec = read_vector(comm,fname_root%idx,vec)
        bv.insertVec(k,vec)
    vec.destroy()

    return bv

def read_bv(comm,fname,sizes,activeColumns):
    """
        Read BV from file (do not destroy Mat since bv is a pointer to the latter)
    """

    Mat = read_dense_matrix(comm,fname,sizes)
    bv = SLEPc.BV().createFromMat(Mat)
    bv.setFromOptions()
    bv.setActiveColumns(0,activeColumns)

    return bv

def stack_vectors(comm,veclst):
    """
        Take a list of vectors and stack them on top of each other
        (same as concatenate along axis = 0 in numpy)
    """

    nblocks = len(veclst)
    sizes = veclst[0].getSizes()
    Nglob = nblocks*sizes[-1]

    vec = PETSc.Vec().create(comm=comm)
    vec.setSizes((compute_local_size(comm,Nglob),Nglob))
    vec.setUp()

    for (k,v) in enumerate(veclst):
        j0, _ = v.getOwnershipRange()
        idces = np.arange(sizes[0]) + (j0 + k*sizes[1])
        vec.setValues(idces,v.getArray(),None)

    vec.assemble()
    return vec

def unstack_vectors(comm,vec,vecl,nblocks):
    """
        Take a list of vectors and stack them on top of each other
        (same as concatenate along axis = 0 in numpy)
    """

    _, N = vecl.getSizes()
    veclst = [vecl.copy() for _ in range (nblocks)]

    idces_to = np.arange(N)
    for i in range (nblocks):
        
        idces_from = np.arange(i*N,(i+1)*N) 
        is_from = PETSc.IS().createGeneral(idces_from,comm=comm)  
        is_to = PETSc.IS().createGeneral(idces_to,comm=comm)
        scatter = PETSc.Scatter().create(vec,is_from,veclst[i],is_to)
        scatter.scatter(vec,veclst[i],addv=PETSc.InsertMode.INSERT_VALUES)

    return veclst

def assemble_sparse_matrix(comm,arrays,sizes):

    # arrays = [rows,cols,data], where rows, etc. are numpy arrays
    # sizes = [Nrows,Ncols,number of nnz per row]

    rank, size = comm.Get_rank(), comm.Get_size()
    Nr, Nc, nnzpr = sizes
    Nrloc = compute_local_size(comm,Nr)
    Ncloc = compute_local_size(comm,Nc)

    """
        Gather indices and values to root and convert (row indices, col indices)
        to CSR (row pointers, col indices)
    """
    rows, cols, data, displ = None, None, None, None
    counts = np.asarray(comm.gather(len(arrays[0]),root=0))
    if rank == 0:
        nnztot = np.sum(counts)
        displ = np.concatenate(([0],np.cumsum(counts[:-1])))
        rows = np.empty(nnztot,dtype=np.int32)
        cols = np.empty(nnztot,dtype=np.int32)
        data = np.empty(nnztot,dtype=np.complex128)

    comm.Gatherv(arrays[0],[rows,counts,displ,MPI.INT],root=0)
    comm.Gatherv(arrays[1],[cols,counts,displ,MPI.INT],root=0)
    comm.Gatherv(arrays[2],[data,counts,displ,MPI.DOUBLE_COMPLEX],root=0)

    """
        Order rows in ascending order and distribute row
        pointers among other processes
    """
    Nloc_lst = np.asarray(comm.gather(Nrloc,root=0))
    Nloc_displ = np.concatenate(([0],np.cumsum(Nloc_lst[:-1]))) if rank == 0 else None
    if rank == 0:
        idces = np.argsort(rows)
        rows, cols, data = rows[idces], cols[idces], data[idces]

        ni = 0
        rowsptr = np.zeros(Nr+1,dtype=np.int32)
        for i in range (Nr):
            ni += np.count_nonzero(rows[ni:ni+nnzpr] == i)
            rowsptr[i+1] = ni

        if rowsptr[-1] != nnztot: raise ValueError ("Total number of nonzeros does not match!")
        
        for i in range (size):
            
            ndisp, nloc = Nloc_displ[i], Nloc_lst[i]
            rowsptr_i = rowsptr[ndisp:ndisp+nloc+1] - rowsptr[ndisp]
            cols_i = cols[rowsptr[ndisp]:rowsptr[ndisp]+rowsptr_i[-1]]
            vals_i = data[rowsptr[ndisp]:rowsptr[ndisp]+rowsptr_i[-1]]
            sendbufs = [rowsptr_i,cols_i,vals_i]

            if i == 0:
                my_rowsptr, my_cols, my_data = sendbufs
            else:
                for j in range (len(sendbufs)):
                    comm.Send(np.asarray([len(sendbufs[j])],dtype=np.int32),dest=i,tag=0)
                    comm.Send(sendbufs[j],dest=i,tag=1)

    else:

        recvbufs, dtypes = [], [np.int32,np.int32,np.complex128]
        for j in range (len(dtypes)):
            bufsz = np.empty(1,dtype=np.int32)
            comm.Recv(bufsz,source=0,tag=0)
            recvbuf = np.empty(bufsz[0],dtype=dtypes[j])
            comm.Recv(recvbuf,source=0,tag=1)
            recvbufs.append(recvbuf)

        my_rowsptr, my_cols, my_data = recvbufs
    
    """
        Assemble matrix
    """
    M = PETSc.Mat().createAIJ([[Nrloc,Nr],[Ncloc,Nc]],comm=MPI.COMM_WORLD)
    M.setPreallocationCSR((my_rowsptr,my_cols))
    M.setValuesCSR(my_rowsptr,my_cols,my_data,True)
    M.assemble(False)

    return M

def assemble_dense_matrix(comm,arrays,sizes):
    """
        arrays = [rows,cols,data], where rows, etc. are numpy arrays
        sizes = [Nrows,Ncols]
    """

    Nrloc = compute_local_size(comm,sizes[0])
    Ncloc = compute_local_size(comm,sizes[-1])
    M = PETSc.Mat().createDense([[Nrloc,sizes[0]],[Ncloc,sizes[-1]]],comm=comm)
    for i in range (len(arrays[0])):
        M.setValue(arrays[0][i],arrays[1][i],arrays[2][i],False)
    M.assemble(False)

    return M

def create_ksp(J):

    ksp = PETSc.KSP().create()
    ksp.setOperators(J)
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')
    pc.setUp()
    ksp.setUp()

    return ksp

def extract_coo_arrays_from_bv(BV):

    _, ncols = BV.getActiveColumns()
    rows, cols, data = [], [], []
    for i in range (ncols):
        vec = BV.getColumn(i)
        ri, _ = vec.getOwnershipRange()
        array = vec.getArray()
        rows.extend(ri + np.arange(len(array),dtype=np.int32))
        cols.extend(i*np.ones(len(array),dtype=np.int32))
        data.extend(array)
        BV.restoreColumn(i,vec)

    return np.asarray(rows,dtype=np.int32), np.asarray(cols,dtype=np.int32), np.asarray(data)



####################################################

def load_petscvec_return_nparray(fname):

    comm = MPI.COMM_WORLD
    viewer = PETSc.Viewer().createBinary(fname,"r",comm=comm)
    vec = PETSc.Vec().create(comm)
    vec.load(viewer)
    array = vec.getArray()
    vec.destroy()
    viewer.destroy()

    return array.reshape(-1)

def load_bv(sizes,activeCols,fname,conj):

    comm = MPI.COMM_WORLD
    viewer = PETSc.Viewer().createBinary(fname,"r",comm=comm)
    Mat = PETSc.Mat().createDense(sizes,comm=comm)
    Mat.load(viewer)
    viewer.destroy()
    if conj == True: Mat.conjugate(None)

    bv = SLEPc.BV().createFromMat(Mat)
    bv.setFromOptions()
    bvcopy = bv.duplicate()
    bv.copy(bvcopy)
    bvcopy.setActiveColumns(0,activeCols)
    
    bv.destroy()
    Mat.destroy()
    
    return bvcopy

def create_seq_dense_from_array(Marray):

    nr, nc = Marray.shape
    Mat = PETSc.Mat().createDense([nr,nc],comm=MPI.COMM_SELF)
    for i in range (nr):
        for j in range (nc):
            Mat.setValue(i,j,Marray[i,j])
    
    Mat.assemble()

    return Mat
