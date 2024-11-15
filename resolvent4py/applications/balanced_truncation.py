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
from ..linear_operators import LowRankLinearOperator
from ..linear_operators import LowRankUpdatedLinearOperator

class balanced_truncation:

    def __init__(self, comm, fnames_jacobian, jacobian_sizes, fname_B, \
                 B_sizes, fname_C, C_sizes, path_factors, \
                 fname_frequencies, fname_weights):
        
        self.comm = comm
        self.jacobian_sizes = jacobian_sizes
        self.A = read_coo_matrix(self.comm, fnames_jacobian, jacobian_sizes)
        self.B = read_bv(comm, fname_B, B_sizes)
        self.C = read_bv(comm, fname_C, C_sizes)

        self.freqs = np.load(fname_frequencies)
        self.weights = np.load(fname_weights)
        self.load_factors(path_factors)
    
    def load_factors(self, path_factors):
        
        nb = self.B.getSizes()[-1]
        nc = self.C.getSizes()[-1]
        self.Xf, self.Yf = [], []
        for i in range (len(self.freqs)):
            fname_x = path_factors + 'Xf_omega_%1.5f.dat'%self.freqs[i]
            fname_y = path_factors + 'Yf_omega_%1.5f.dat'%self.freqs[i]
            self.Xf.append(read_bv(self.comm, fname_x, nb))
            self.Yf.append(read_bv(self.comm, fname_y, nc))            


    def solve(self, rdim):

        if np.max(self.freqs) > 0.0:
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
                    idxj0, idxj1 = i*nb, (i+1)*nb
                    f = np.sqrt(wi*wj)
                    Mat = Xj.dot(Yi)
                    M[idxi0:idxi1, idxj0:idxj1] = f*Mat.getDenseArray()
                    Mat.destroy()
        else:
            Xreal = self.Xf[0].duplicate()
            Ximag = self.Xf[0].duplicate()
            Yreal = self.Yf[0].duplicate()
            Yimag = self.Yf[0].duplicate()
            shift = [0, nb*nf//2]
            for i in range (len(self.Yf)):
                wi = self.weights[i]
                Yi = self.Yf[i]
                Yreal = bv_real(Yi, Yreal)
                Yimag = bv_imag(Yi, Yimag)
                for (ii, Y) in enumerate([Yreal, Yimag]):
                    idxi0, idxi1 = i*nc + shift[ii], (i+1)*nc + shift[ii]
                    for j in range (len(self.Xf)):
                        wi = self.weights[j]
                        Xj = self.Xf[j]
                        Xreal = bv_real(Xj, Xreal)
                        Ximag = bv_imag(Xj, Ximag)
                        for (jj, X) in enumerate([Xreal, Ximag]):
                            idxj0 = j*nb + shift[jj]
                            idxj1 = (j+1)*nb + shift[jj]
                            f = 2*np.sqrt(wi*wj)
                            Mat = X.dot(Y)
                            M[idxi0:idxi1, idxj0:idxj1] = f*Mat.getDenseArray()
                            Mat.destroy()


        U, S, VHT = sp.linalg.svd(M, full_matrices=False)
        V = VHT.conj().T
        idces = np.argwhere(S > 1e-11).reshape(-1)
        U = U[:, idces]
        S = S[idces]
        V = V[:, idces]
        USsqrt = U@np.diag(1./np.sqrt(S))
        VSsqrt = V@np.diag(1./np.sqrt(S))

        Phi = SLEPc.BV().create(comm=self.comm)
        Phi.setSizes((self.Xf[0].getSizes()[0], len(idces)))
        Phi.setType('mat')
        Phi.scale(0.0)
        Psi = Phi.copy()
        if negative_freqs == False:
            for i in range (len(self.Xf)):
                VSsqrt_i = VSsqrt[i*nb:(i+1)*nb,]
                Vi = PETSc.Mat().createDense(VSsqrt_i.shape, None, \
                                            VSsqrt_i, MPI.COMM_SELF)
                Phi.mult(1.0, 1.0, self.Xf[i], Vi)
                USsqrt_i = USsqrt[i*nc:(i+1)*nc,]
                Ui = PETSc.Mat().createDense(USsqrt_i.shape, None, \
                                            USsqrt_i, MPI.COMM_SELF)
                Psi.mult(1.0, 1.0, self.Yf[i], Ui)
                Vi.destroy()
                Ui.destroy()
        else:
            shift = [0, nb*nf//2]
            for i in range (len(self.Xf)):
                Xreal = bv_real(self.Xf[i], Xreal)
                Ximag = bv_imag(self.Xf[i], Ximag)
                for (ii, X) in enumerate([Xreal, Ximag]):
                    VSsqrt_i = VSsqrt[i*nb + shift[ii]:(i+1)*nb + shift[ii],]
                    Vi = PETSc.Mat().createDense(VSsqrt_i.shape, None, \
                                                VSsqrt_i, MPI.COMM_SELF)
                    Phi.mult(1.0, 1.0, X, Vi)
                    Vi.destroy()
                Yreal = bv_real(self.Yf[i], Yreal)
                Yimag = bv_imag(self.Yf[i], Yimag)
                for (ii, Y) in enumerate([Yreal, Yimag]):
                    USsqrt_i = USsqrt[i*nc + shift[ii]:(i+1)*nc + shift[ii],]
                    Ui = PETSc.Mat().createDense(USsqrt_i.shape, None, \
                                                USsqrt_i, MPI.COMM_SELF)
                    Psi.mult(1.0, 1.0, Y, Ui)
                    Ui.destroy()

        if rdim != None and rdim < Phi.getSizes()[-1]:
            Phi.resize(rdim, copy=True)
            Psi.resize(rdim, copy=True)
            S = S[:rdim]
        
        return (Phi, Psi, S)















