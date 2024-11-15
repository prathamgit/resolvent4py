from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc
from ..linalg import enforce_complex_conjugacy
from ..miscellaneous import petscprint

def arnoldi_iteration(lin_op, lin_op_action, krylov_dim):
    r"""
        This function uses the Arnoldi iteration algorithm to compute an 
        orthonormal basis and the corresponding Hessenberg matrix 
        for the range of the linear operator specified by
        :code:`lin_op` and :code:`lin_op_action`.

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: one of :code:`lin_op.apply`, :code:`lin_op.solve`,
            :code:`lin_op.apply_hermitian_transpose` or 
            :code:`lin_op.solve_hermitian_transpose`
        :param krylov_dim: dimension of the Krylov subspace
        :type kyrlov_dim: int

        :return: a 2-tuple with an orthonormal basis for the Krylov subspace
            and the Hessenberg matrix
        :rtype: (SLEPc BV with :code:`krylov_dim` columns, \
            numpy.ndarray of size :code:`krylov_dim x krylov_dim`)
    """
    comm = lin_op.get_comm()
    sizes = lin_op.get_dimensions()[0]
    nblocks = lin_op.get_nblocks()
    block_cc = lin_op._block_cc
    
    # Initialize the BV structure and the Hessenberg matrix
    Q = SLEPc.BV().create(comm=comm)
    Q.setSizes(sizes, krylov_dim)
    Q.setFromOptions()
    H = np.zeros((krylov_dim, krylov_dim),dtype=np.complex128)
    # Draw the first vector at random
    q = Q.createVec()
    qa = np.random.randn(sizes[0]) + 1j*np.random.randn(sizes[0])
    qa = qa.real if lin_op._real == True else qa
    q.setArray(qa)
    enforce_complex_conjugacy(comm, q, nblocks) if block_cc == True else None
    q.scale(1./q.norm())
    Q.insertVec(0,q)
    # Perform Arnoldi iteration
    for k in range(1,krylov_dim+1):
        v = lin_op_action(q)
        # string = "Arnoldi iteration (%d/%d) - ||Aq|| "%(k, krylov_dim) + \
        #             "= %1.15e"%(v.norm())
        # petscprint(comm, string)
        for j in range (k):
            qj = Q.getColumn(j)
            H[j,k-1] = v.dot(qj)
            v.axpy(-H[j,k-1],qj)
            Q.restoreColumn(j,qj)
        if k < krylov_dim:
            H[k,k-1] = v.norm()
            v.scale(1./H[k,k-1])
            Q.insertVec(k,v)
        q = v.copy()
        v.destroy()
    return (Q, H)


def eig(lin_op, lin_op_action, krylov_dim, n_evals, process_evals=None):
    r"""
        Compute the eigendecomposition of the linear operator :math:`L` 
        specified by :code:`lin_op` and :code:`lin_op_action`. Example:
        to compute the eigenvalues of :math:`L` closest to the origin, 
        set :code:`lin_op_action = lin_op.solve` and 
        :code:`process_evals = lambda x: 1./x`.

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: one of :code:`lin_op.apply`, :code:`lin_op.solve`,
            :code:`lin_op.apply_hermitian_transpose` or 
            :code:`lin_op.solve_hermitian_transpose`
        :param krylov_dim: dimension of the Arnoldi Krylov subspace
        :type krylov_dim: int
        :param n_evals: number of eigenvalues to return
        :type n_evals: int
        :param process_evals: function to extract the desired eigenvalues (see
            example above).
        :type process_evals: Callable

        :return: a 2-tuple with the :code:`n_evals` desired eigenvalues
            and corresponding eigenvectors
        :rtype: (numpy.ndarray of size :code:`n_evals x n_evals`, 
            SLEPc.BV with :code:`n_evals` columns)
    """
    Q, H = arnoldi_iteration(lin_op, lin_op_action, krylov_dim)
    evals, evecs = sp.linalg.eig(H)
    idces = np.flipud(np.argsort(np.abs(evals)))[:n_evals]
    evals = evals[idces]
    evecs = evecs[:,idces]
    evecs_ = PETSc.Mat().createDense(evecs.shape,None,evecs,comm=MPI.COMM_SELF)
    Q.multInPlace(evecs_,0,n_evals)
    Q.setActiveColumns(0,n_evals)
    Q.resize(n_evals,copy=True)
    process_evals = (lambda x: x) if process_evals == None else process_evals
    evals = process_evals(evals)
    return (np.diag(evals), Q)

def right_and_left_eig(lin_op, lin_op_action, krylov_dim, n_evals, \
                       process_evals=None):
    r"""
        Compute the right and left eigendecomposition of the linear operator 
        :math:`L` specified by :code:`lin_op` and :code:`lin_op_action`.
        Example: to compute the eigenvalues of :math:`L` closest to the origin, 
        set :code:`lin_op_action = lin_op.solve` and 
        :code:`process_evals = lambda x: 1./x`.

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param lin_op_action: one of :code:`lin_op.apply`, :code:`lin_op.solve`,
            :code:`lin_op.apply_hermitian_transpose` or 
            :code:`lin_op.solve_hermitian_transpose`
        :param krylov_dim: dimension of the Arnoldi Krylov subspace
        :type krylov_dim: int
        :param n_evals: number of eigenvalues to return
        :type n_evals: int
        :param process_evals: function to extract the desired eigenvalues (see
            example above).
        :type process_evals: Callable

        :return: a 3-tuple with the desired right eigenvectors :math:`V`, the 
            corresponding eigenvalues :math:`D` and left eigenvectors :math:`W`
            normalized so that :math:`W^* V = I`.
        :rtype: (SLEPc.BV with :code:`n_evals` columns, 
            numpy.ndarray of size :code:`n_evals x n_evals`, 
            SLEPc.BV with :code:`n_evals` columns)
    """
    
    if lin_op_action == lin_op.solve:
        lin_op_action_adj = lin_op.solve_hermitian_transpose
    elif lin_op_action == lin_op.apply:
        lin_op_action_adj = lin_op.apply_hermitian_transpose
    else:
        raise ValueError (
            f"lin_op_action must be one of lin_op.solve or lin_op.apply. "
            f"See documentation for additional information."
        )
    # Compute the right and left eigendecompositions
    Dfwd, Qfwd = eig(lin_op, lin_op_action, krylov_dim, n_evals)
    Dadj, Qadj = eig(lin_op, lin_op_action_adj, krylov_dim, n_evals)
    # Match the right and left eigenvalues/vectors
    Dfwdd = np.diag(Dfwd)
    Dadjd = np.diag(Dadj)
    idces = [np.argmin(np.abs(Dfwdd - val.conj())) for val in Dadjd]
    Qadj_ = Qadj.copy()
    for j in range (len(idces)):
        q_ = Qadj_.getColumn(idces[j])
        Qadj.insertVec(j, q_)
        Qadj_.restoreColumn(idces[j], q_)
    Qadj_.destroy()
    # Biorthogonalize the eigenvectors
    M = Qfwd.dot(Qadj)
    evals, evecs = sp.linalg.eig(M.getDenseArray())
    idces = np.argwhere(np.abs(evals) < 1e-10).reshape(-1)
    evals[idces] += 1e-10
    Minv = evecs@np.diag(1./evals)@sp.linalg.inv(evecs)
    Minv = PETSc.Mat().createDense(Minv.shape, None, Minv, MPI.COMM_SELF)
    M.destroy()
    Qfwd.multInPlace(Minv, 0, n_evals)
    process_evals = (lambda x: x) if process_evals == None else process_evals
    Dfwd = np.diag(process_evals(Dfwdd))
    return (Qfwd, Dfwd, Qadj)


def check_eig_convergence(lin_op_action, D, V):
    r"""
        Check convergence of the eigenpairs by measuring 
        :math:`\lVert L v - \lambda v\rVert`.

        :param lin_op_action: one of :code:`L.apply`, :code:`L.solve`, 
            :code:`L.apply_hermitian_transpose` or 
            :code:`L.solve_hermitian_transpose`
        :param D: diagonal 2D numpy array with the eigenvalues
        :type D: numpy.ndarray
        :param V: corresponding eigenvectors
        :type V: `BV`_
            
        :return: None
    """
    petscprint(MPI.COMM_WORLD, " ")
    petscprint(MPI.COMM_WORLD, "Executing eigenpair convergence check...")
    w = V.createVec()
    for j in range (D.shape[-1]):
        v = V.getColumn(j)
        e = v.copy()
        e.scale(D[j,j])
        w = lin_op_action(v, w)
        e.axpy(-1.0, w)
        error = e.norm()
        V.restoreColumn(j, v)
        e.destroy()
        str = "Error for eigenpair %d = %1.15e"%(j+1, error)
        petscprint(MPI.COMM_WORLD, str)
    w.destroy()
    petscprint(MPI.COMM_WORLD, "Executing eigenpair convergence check...")
    petscprint(MPI.COMM_WORLD, " ")
        
