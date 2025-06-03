import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


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

nfp = nfb + 3
perts_freqs = np.arange(-nfp, nfp + 1) * bflow_freqs[1]
nblocks = 2 * nfp + 1

# ------------------------------------------------------------------------------
# -------- Read data from file and assemble harmonic resolvent generator -------
# ------------------------------------------------------------------------------
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
# Perturb the generator to avoid numerical singularities
Id = res4py.create_AIJ_identity(comm, H.getSizes())
Id.scale(1e-7)
H.axpy(-1.0, Id)
Id.destroy()
ksp = res4py.create_mumps_solver(comm, H)
res4py.check_lu_factorization(comm, H, ksp)

Hop = res4py.linear_operators.MatrixLinearOperator(comm, H, ksp, nblocks)

# ------------------------------------------------------------------------------
# -------- Read base-flow time-derivative and define projection operators ------
# -------- to remove the phase-shift direction ---------------------------------
# ------------------------------------------------------------------------------
fnames_lst = [(save_path + "dQ_%02d.dat" % j) for j in range(len(bflow_freqs))]
dQ = res4py.read_harmonic_balanced_vector(
    comm, fnames_lst, True, (nl, n), (Nl, N)
)
dQ.scale(1 / dQ.norm())
w = Hop.solve_hermitian_transpose(dQ)
w.scale(1 / w.norm())

Phi = SLEPc.BV().create(comm)
Phi.setSizes(dQ.getSizes(), 1)
Phi.setType('mat')
Psi = Phi.copy()
Phi.insertVec(0, dQ)
Psi.insertVec(0, w)

Pd = res4py.linear_operators.ProjectionOperator(comm, Psi, Psi, True, nblocks)
Pr = res4py.linear_operators.ProjectionOperator(comm, Phi, Phi, True, nblocks)

lops = [Pr, Hop, Pd]
lops_actions = [Pr.apply, Hop.solve, Pd.apply]
Linop = res4py.linear_operators.ProductLinearOperator(comm, lops, lops_actions, nblocks)


_, S, _ = res4py.linalg.randomized_svd(Linop, Linop.apply_mat, 30, 3, 10)
S = np.diag(S)
_, S2, _ = res4py.linalg.randomized_svd(Hop, Hop.solve_mat, 30, 3, 11)
S2 = np.diag(S2)

res_path = 'results/'
os.makedirs(res_path) if not os.path.exists(res_path) else None

if comm.Get_rank() == 0:

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(S) + 1), S.real, "ko", label=r"$P_r H P_d$")
    ax.set_xlabel(r"Index $j$ for $P_r H P_d$")
    ax.set_ylabel(r"$\sigma_j$")
    ax2 = ax.twiny()
    ax2.plot(np.arange(2, len(S2) + 1), S2[1:].real, "rx", label=r"$H$")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_xticks(np.arange(1, len(S)+1))
    ax2.set_xticks(np.arange(2, len(S2)+1))
    ax2.set_xlabel(r"Index $j$ for $H$")
    plt.tight_layout()
    plt.savefig(res_path + 'singular_values.png', dpi=100)
    


P = res4py.linear_operators.ProjectionOperator(comm, Phi, Psi, True, nblocks)
lops = [P, Hop, P]
lops_actions = [P.apply, Hop.solve, P.apply]
Linop = res4py.linear_operators.ProductLinearOperator(comm, lops, lops_actions, nblocks)

D, _ = res4py.linalg.eig(Linop, Linop.apply, N - 3, 30, lambda x: 1 / x)
D = np.diag(D)

if comm.Get_rank() == 0:
    omega = bflow_freqs[1]
    idces = np.argwhere((D.imag > -omega/2) & (D.imag <= omega/2)).reshape(-1)
    
    plt.figure()
    plt.plot(D.real, D.imag, "ko")
    plt.plot(D[idces].real, D[idces].imag, 'go')
    plt.plot(0, 0, "rx")
    ax = plt.gca()
    ax.axhline(y=omega / 2, color="r", alpha=0.5)
    ax.axhline(y=-omega / 2, color="r", alpha=0.5)
    ax.set_xlabel(r"$\mathrm{Real}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Imag}(\lambda)$")
    plt.tight_layout()
    plt.savefig(res_path + 'floquet_exponents.png', dpi=100)