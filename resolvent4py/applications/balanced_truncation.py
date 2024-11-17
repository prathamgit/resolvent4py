from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc

from ..io_functions import read_coo_matrix
from ..io_functions import read_bv
from ..linalg import bv_real
from ..linalg import bv_imag
from ..linalg import compute_local_size


from ..linear_operators import MatrixLinearOperator

class balanced_truncation:

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
            print(len(self.freqs), nf, nb, nc, M.shape)
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












