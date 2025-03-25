from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc
from .. import typing

from ..io_functions import read_coo_matrix
from ..io_functions import read_bv
from ..linalg import bv_real
from ..linalg import bv_imag
from ..linalg import compute_local_size
from ..miscellaneous import petscprint
from ..miscellaneous import create_dense_matrix
from ..comms import distributed_to_sequential_matrix


from ..linear_operators import MatrixLinearOperator


def sample_gramian_factors(
        L_generators: typing.Callable,
        frequencies: np.ndarray,
        weights: np.ndarray,
        B: SLEPc.BV,
        C: SLEPc.BV,
    ) -> typing.Tuple[PETSc.Mat, PETSc.Mat]:
    r"""
    Compute the Gramian factors as outlined in section 2 of [Dergham2011]_.
    In particular, we efficiently approximate the Reachability and Observability
    Gramians 

    .. math::

        G_R = \frac{1}{2\pi}\int_{-\infty}^{\infty}R(\omega)BB^*R(\omega)^*\
        \,d\omega, \quad G_O = \frac{1}{2\pi}\int_{-\infty}^{\infty}\
        R(\omega)^*CC^*R(\omega)\,d\omega

    using quadrature over a discrete set of frequencies :math:`\Omega`. Here,
    :math:`R(\omega) = \left(i\omega I - A\right)^{-1}`.
    The quadrature points and appropriate quadrature weights are supplied by
    the user.

    :param L_generators: tuple of functions that define the linear operator \
        :math:`R(\omega)^{-1}`, the action of :math:`R(\omega)` on matrices,
        and destructors to avoid memory leaks. (See examples.)
    :type L_generators: Tuple[[Callable, Callable, ...]]
    :param frequencies: quadrature points
    :type frequencies: numpy.ndarray
    :param weights: quadrature weights
    :type weights: numpy.ndarray
    :param B: input matrix
    :type B: SLEPc.BV
    :param C: output matrix
    :type C: SLEPc.BV

    References
    ----------
    .. [Dergham2011] Dergham et al., *Model reduction for fluids using \
        frequential snapshots*, Physics of Fluids, 2011
    """
    nb = B.getSizes()[-1]
    nc = C.getSizes()[-1]
    idces = np.argsort(frequencies).reshape(-1)
    frequencies = frequencies[idces]
    weights = weights[idces]
    min_f = frequencies[0]
    if min_f >= 0.0:
        split = True
        nf = 2*(len(frequencies)-1) + 1 if min_f == 0.0 else 2*len(frequencies)
    else:
        split = False
        nf = len(frequencies)
    
    L, action, destroyers = L_generators[0](frequencies[0])
    action_ht = L.apply_hermitian_transpose_mat if action == L.apply_mat \
        else L.solve_hermitian_transpose_mat
    
    LB = L.create_left_bv(nb)
    LHTC = L.create_right_bv(nc)
    Xarray = np.zeros((LB.getSizes()[0], nb*nf), dtype=np.complex128)
    Yarray = np.zerso((LHTC.getSizes()[0], nc*nf), dtype=np.complex128)
    for k in range (len(frequencies)):
        if k > 0:
            L, action, destroyers = L_generators[k](frequencies[k])
            action_ht = L.apply_hermitian_transpose_mat if \
                action == L.apply_mat else L.solve_hermitian_transpose_mat
        LB = action(B, LB)
        LC = action_ht(C, LC)
        LB.scale(np.sqrt(weights[k]))
        LC.scale(np.sqrt(weights[k]))
        LBMat = LB.getMat()
        LBMat_ = LBMat.getDenseArray().copy()
        LB.restoreMat(LBMat)
        LCMat = LC.getMat()
        LCMat_ = LCMat.getDenseArray().copy()
        LC.restoreMat(LCMat)
        if not split:
            Xarray[:,k*nb:(k+1)*nb] = LBMat_
            Yarray[:,k*nc:(k+1)*nc] = LCMat_
        else:
            if k == 0:
                Xarray[:,0:nb] = LBMat_.real
                Yarray[:,0:nc] = LCMat_.real
            else:
                k0 = nb + 2*(k-1)*nb
                k1 = k0 + 2*nb
                Xarray[:,k0:k1] = np.concatenate((LBMat_.real, LBMat_.imag), -1)
                k0 = nc + 2*(k-1)*nc
                k1 = k0 + 2*nc
                Yarray[:,k0:k1] = np.concatenate((LCMat_.real, LCMat_.imag), -1)
        for destroy in destroyers: destroy()
    LB.destroy()
    X = PETSc.Mat().createDense(LB.getSizes(), None, Xarray, MPI.COMM_WORLD)
    Y = PETSc.Mat().createDense(LC.getSizes(), None, Yarray, MPI.COMM_WORLD)
    return (X, Y)


def compute_projection(
        X: SLEPc.BV,
        Y: SLEPc.BV,
        r: int
    ) -> typing.Tuple[SLEPc.BV, SLEPc.BV, np.ndarray]:

    comm = MPI.COMM_WORLD

    Y.hermitianTranspose()
    Z = Y.matMult(X)
    Y.hermitianTranspose()
    svd = SLEPc.SVD().create(MPI.COMM_WORLD)
    svd.setOperators(Z)
    svd.setProblemType(SLEPc.SVD.ProblemType.STANDARD)
    svd.setType(SLEPc.SVD.Type.CROSS)
    svd.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
    svd.setUp()
    svd.solve()
    Z.destroy()

    r = np.min([r, svd.getConverged()])
    rloc = compute_local_size(r)
    u = Z.createVecLeft()
    v = Z.createVecRight()
    U = create_dense_matrix(comm, (Y.getSizes()[-1], (rloc, r)))
    V = create_dense_matrix(comm, (X.getSizes()[-1], (rloc, r)))
    S = create_dense_matrix(comm, ((rloc, r), (rloc, r)))
    U_ = U.getDenseArray()
    V_ = V.getDenseArray()
    S_ = np.zeros((r, r), dtype=np.float64)
    for i in range (r):
        s = svd.getSingularTriplet(i, u, v)
        U_[:, i] = u.getArray().copy()
        V_[:, i] = v.getArray().copy()
        S_[i, i] = s
        S.setValue(i, i, 1./np.sqrt(s))
    S.assemble(None)
    Phi_mat = X.matMatMult(V, S)
    Psi_mat = Y.matMatMult(U, S)
    Phi_ = SLEPc.BV().createFromMat(Phi_mat)
    Phi_.setType('mat')
    Phi = Phi_.copy()
    Psi_ = SLEPc.BV().createFromMat(Psi_mat)
    Psi_.setType('mat')
    Psi = Psi_.copy()
    objs = [Phi_, Psi_, Phi_mat, Psi_mat, U, V, S]
    for obj in objs: obj.destroy()
    return (Phi, Psi, S_)

    
    









class BalancedTruncation:

    def __init__(self, comm, fnames_jacobian, jacobian_sizes, fname_B, \
                 B_sizes, fname_C, C_sizes, path_factors, \
                 fname_frequencies, fname_weights):
        
        self.comm = comm
        self.jacobian_sizes = jacobian_sizes
        A = read_coo_matrix(self.comm, fnames_jacobian, jacobian_sizes)
        self.A = MatrixLinearOperator(comm, A)
        self.B = read_bv(comm, fname_B, B_sizes)
        self.C = read_bv(comm, fname_C, C_sizes)

        self.freqs = np.load(fname_frequencies)
        self.weights = np.load(fname_weights)
        self.load_factors(path_factors)
    
    def load_factors(self, path_factors):
        
        sizes_B = self.B.getSizes()
        sizes_C = self.C.getSizes()
        self.Xf, self.Yf = [], []
        for i in range (len(self.freqs)):
            fname_x = path_factors + 'Xf_omega_%1.5f.dat'%self.freqs[i]
            fname_y = path_factors + 'Yf_omega_%1.5f.dat'%self.freqs[i]
            self.Xf.append(read_bv(self.comm, fname_x, sizes_B))
            self.Yf.append(read_bv(self.comm, fname_y, sizes_C))


    def compute_balancing_factors(self, rdim=None):

        if np.min(self.freqs) > 0.0:
            negative_freqs = False
            nf = 2*len(self.freqs)
        else:
            negative_freqs = True
            nf = len(self.freqs)

        nb = self.B.getSizes()[-1]
        nc = self.C.getSizes()[-1]
        M = np.zeros((nf*nc, nf*nb), dtype=np.complex128)
        if negative_freqs:
            for i in range (len(self.Yf)):
                wi = self.weights[i]
                Yi = self.Yf[i]
                idxi0, idxi1 = i*nc, (i+1)*nc
                for j in range (len(self.Xf)):
                    wj = self.weights[j]
                    Xj = self.Xf[j]
                    idxj0, idxj1 = j*nb, (j+1)*nb
                    f = np.sqrt(wi*wj)
                    Mat = Xj.dot(Yi)
                    M[idxi0:idxi1, idxj0:idxj1] = f*Mat.getDenseArray()
                    Mat.destroy()
        else:
            shift_i = [0, nc*nf//2]
            shift_j = [0, nb*nf//2]
            for i in range (len(self.Yf)):
                wi = self.weights[i]
                Yi = self.Yf[i]
                Yreal = bv_real(Yi)
                Yimag = bv_imag(Yi)
                for (ii, Y) in enumerate([Yreal, Yimag]):
                    idxi0 = i*nc + shift_i[ii]
                    idxi1 = (i+1)*nc + shift_i[ii]
                    for j in range (len(self.Xf)):
                        wj = self.weights[j]
                        Xj = self.Xf[j]
                        Xreal = bv_real(Xj)
                        Ximag = bv_imag(Xj)
                        for (jj, X) in enumerate([Xreal, Ximag]):
                            idxj0 = j*nb + shift_j[jj]
                            idxj1 = (j+1)*nb + shift_j[jj]
                            f = 2*np.sqrt(wi*wj)
                            Mat = X.dot(Y)
                            M[idxi0:idxi1, idxj0:idxj1] = f*Mat.getDenseArray()
                            Mat.destroy()
                        Xreal.destroy()
                        Ximag.destroy()
                Yreal.destroy()
                Yimag.destroy()

        U, S, VHT = sp.linalg.svd(M, full_matrices=False)
        V = VHT.conj().T
        idces = np.argwhere(S > 1e-11).reshape(-1)
        U = U[:, idces]
        S = S[idces]
        V = V[:, idces]
        USsqrt = U@np.diag(1./np.sqrt(S))
        VSsqrt = V@np.diag(1./np.sqrt(S))

        Phi = SLEPc.BV().create(comm=self.comm)
        Phi.setSizes(self.Xf[0].getSizes()[0], len(idces))
        Phi.setType('mat')
        Phi.scale(0.0)
        Psi = Phi.copy()
        if negative_freqs:
            for i in range (len(self.Xf)):
                wi = np.sqrt(self.weights[i])
                VSsqrt_i = wi*VSsqrt[i*nb:(i+1)*nb,]
                Vi = PETSc.Mat().createDense(VSsqrt_i.shape, None, \
                                             VSsqrt_i, MPI.COMM_SELF)
                Phi.mult(1.0, 1.0, self.Xf[i], Vi)
                USsqrt_i = wi*USsqrt[i*nc:(i+1)*nc,]
                Ui = PETSc.Mat().createDense(USsqrt_i.shape, None, \
                                            USsqrt_i, MPI.COMM_SELF)
                Psi.mult(1.0, 1.0, self.Yf[i], Ui)
                Vi.destroy()
                Ui.destroy()
        else:
            shift_i = [0, nc*nf//2]
            shift_j = [0, nb*nf//2]
            for i in range (len(self.Xf)):
                wi = np.sqrt(2*self.weights[i])
                Xreal = bv_real(self.Xf[i])
                Ximag = bv_imag(self.Xf[i])
                for (ii, X) in enumerate([Xreal, Ximag]):
                    VSsqrt_i = wi*VSsqrt[i*nb+shift_j[ii]:(i+1)*nb+shift_j[ii],]
                    Vi = PETSc.Mat().createDense(VSsqrt_i.shape, None, \
                                                 VSsqrt_i, MPI.COMM_SELF)
                    Phi.mult(1.0, 1.0, X, Vi)
                    Vi.destroy()
                Yreal = bv_real(self.Yf[i])
                Yimag = bv_imag(self.Yf[i])
                for (ii, Y) in enumerate([Yreal, Yimag]):
                    USsqrt_i = wi*USsqrt[i*nc+shift_i[ii]:(i+1)*nc+shift_i[ii],]
                    Ui = PETSc.Mat().createDense(USsqrt_i.shape, None, \
                                                USsqrt_i, MPI.COMM_SELF)
                    Psi.mult(1.0, 1.0, Y, Ui)
                    Ui.destroy()
                BVs = [Xreal, Ximag, Yreal, Yimag]
                for bv in BVs: bv.destroy()

        if rdim != None and rdim < Phi.getSizes()[-1]:
            Phi.resize(rdim, copy=True)
            Psi.resize(rdim, copy=True)
            S = S[:rdim]
        
        return (Phi, Psi, S)


    def compute_reduced_order_tensors(self, Phi, Psi):

        APhi = self.A.apply_mat(Phi)
        Arm = APhi.dot(Psi)
        Brm = self.B.dot(Psi)
        Crm = self.C.dot(Phi)
        Ar = Arm.getDenseArray().copy()
        Br = Brm.getDenseArray().copy()
        Cr = Crm.getDenseArray().copy()
        Arm.destroy()
        Brm.destroy()
        Crm.destroy()
        return (Ar, Br, Cr)












