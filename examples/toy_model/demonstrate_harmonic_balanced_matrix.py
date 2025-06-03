import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)


comm = MPI.COMM_WORLD

save_path = "data/"
bflow_freqs = np.load(save_path + "bflow_freqs.npy")
nfb = len(bflow_freqs) - 1
fnames_lst = [
    (
        save_path + "rows_%02d.dat" % j,
        save_path + "cols_%02d.dat" % j,
        save_path + "vals_%02d.dat" % j,
    )
    for j in range(nfb + 1)
]

nfp = nfb
perts_freqs = np.arange(-nfp, nfp + 1) * bflow_freqs[1]

N = 3 * len(perts_freqs)
Nl = res4py.compute_local_size(N)
n = 3
nl = res4py.compute_local_size(n)
A = res4py.read_harmonic_balanced_matrix(
    comm,
    fnames_lst,
    True,
    ((nl, n), (nl, n)),
    ((Nl, N), (Nl, N)),
)
H = res4py.assemble_harmonic_resolvent_generator(comm, A, perts_freqs)
ksp = res4py.create_mumps_solver(comm, H)
# ksp = res4py.create_gmres_bjacobi_solver(
#     comm,
#     H,
#     len(perts_freqs),
#     1e-10,
#     1e-10,
# )
# res4py.check_gmres_bjacobi_solver(comm, H, ksp)
res4py.check_lu_factorization(comm, H, ksp)

Linop = res4py.linear_operators.MatrixLinearOperator(
    comm, H, ksp, (2 * nfp + 1)
)
print(Linop._block_cc)
D, V = res4py.linalg.eig(Linop, Linop.solve, N - 3, 20, lambda x: 1 / x)
D = np.diag(D)

if comm.Get_rank() == 0:
    plt.figure()
    plt.plot(D.real, D.imag, "ko")
    plt.plot(0, 0, "rx")
    ax = plt.gca()
    ax.axhline(y=bflow_freqs[1] / 2, color="r", alpha=0.5)
    ax.axhline(y=-bflow_freqs[1] / 2, color="r", alpha=0.5)
    ax.set_xlabel(r"$\mathrm{Real}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Imag}(\lambda)$")
    plt.tight_layout()
    plt.show()

# x = PETSc.Vec().createWithArray(np.ones(Nl), (Nl, N), None, comm)
# y = x.duplicate()
# H.mult(x, y)

# x = res4py.generate_random_petsc_vector(comm, (Nl, N), True)
# y = x.duplicate()
# ksp.solve(x, y)

# x2 = x.duplicate()
# H.mult(y, x2)
# x2.axpy(-1.0, x)
# res4py.petscprint(comm, x2.norm())

# fnames_lst = [(save_path + "Aj_%02d.dat" % j) for j in range(len(bflow_freqs))]
# bv = res4py.read_harmonic_balanced_bv(
#     comm, fnames_lst, True, ((nl, n), 3), ((Nl, N), N)
# )

# # bv.view()
# L = bv.getMat().getDenseArray()
# A.convert(PETSc.Mat.Type.DENSE)
# A_ = A.getDenseArray()
# print(np.linalg.norm(A_ - L))
