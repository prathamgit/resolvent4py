r"""
Resolvent Analysis Demonstration via Time Stepping
==================================================

Given the linear dynamics :math:`d_t q = Aq`, we perform resolvent analysis
by computing the singular value decomposition (SVD) of the resolvent operator

.. math::

    R(i\omega) = \left(i\omega I - A\right)^{-1}

with :math:`\omega = 0.648` the natural frequency of the linearized CGL
equation. This script demonstrates the following:

- Resolvent analysis using time-stepping via
  :func:`~resolvent4py.linalg.resolvent_analysis_time_stepping.resolvent_analysis_rsvd_dt`

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
import scipy as sp
from petsc4py import PETSc

import cgl


def save_bv_list(bv_list, prefix, save_path):
    save_dir = pathlib.Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, bv in enumerate(bv_list):
        nv = bv.getSizes()[1]
        for j in range(nv):
            vec = bv.getColumn(j)
            fname = save_dir / f"{prefix}_freq{i:02d}_mode{j:02d}.petsc"
            viewer = PETSc.Viewer().createBinary(
                str(fname), "w", comm=vec.comm
            )
            vec.view(viewer)
            viewer.destroy()
            bv.restoreColumn(j, vec)

def ensure_structural_diagonal(mat, value_if_empty=0.0):
    r0, _ = mat.getOwnershipRange()
    diag = mat.getDiagonal()
    holes = diag.getArray() == 0
    diag.destroy()

    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)
    for local_i, hole in enumerate(holes):
        if hole:
            global_i = r0 + local_i
            mat.setValue(
                global_i,
                global_i,
                value_if_empty,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )

    mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    mat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)


def shift_matrix_by_matrix(A, G, alpha):
    A.axpy(-alpha, G)
    A.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    A.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)

comm = PETSc.COMM_WORLD
rank = comm.getRank()
save_path = "results/"

# Read the A matrix from file
res4py.petscprint(comm, "Reading matrix from file...")
load_path = "data/"
N = 2000
Nl = res4py.compute_local_size(N)
sizes = ((Nl, N), (Nl, N))
names = [
    load_path + "rows.dat",
    load_path + "cols.dat",
    load_path + "vals.dat",
]
A = res4py.read_coo_matrix(comm, names, sizes)

s = 0.0206

ksp = res4py.create_gmres_bjacobi_solver(comm, A, nblocks=comm.Get_size())
res4py.petscprint(comm, "A ksp")
L = res4py.linear_operators.MatrixLinearOperator(comm, A, ksp)
res4py.petscprint(comm, "A operator")

# Compute the svd
res4py.petscprint(comm, "Running randomized SVD...")
n_periods = 20
n_rand = 5
n_loops = 3
n_svals = 1

U, S, V = res4py.linalg.resolvent_analysis_rsvd_dt(
    L,
    0.02,
    s,
    10,
    n_periods,
    n_rand,
    n_loops,
    n_svals
)
Sa = np.diag(Sa)

save_bv_list(U, "U", save_path)
save_bv_list(V, "V", save_path)

if rank == 0:
    for i in range(len(S)):
        print(S[i][0, 0])

idx = 0
bvs = [Ua, Ut, Va, Vt]
arrays = []
for bv in bvs:
    vec = bv.getColumn(idx)
    vecseq = res4py.distributed_to_sequential_vector(vec)
    bv.restoreColumn(idx, vec)
    arrays.append(vecseq.getArray().copy())
    vecseq.destroy()

if comm.getRank() == 0:
    save_path = "results/"
    os.makedirs(save_path) if not os.path.exists(save_path) else None

    l = 30 * 2
    x = np.linspace(-l / 2, l / 2, num=N, endpoint=True)
    nu = 1.0 * (2 + 0.4 * 1j)
    gamma = 1 - 1j
    mu0 = 0.38
    mu2 = -0.01
    sigma = 0.4
    system = cgl.CGL(x, nu, gamma, mu0, mu2, sigma)

# 225442.64114943202
# 255609.02196194816
# 10853.866858048343
# 5437.6451349571835
# 3237.275242550277
# 2305.4072426501807
# 1811.7329001981327
# 1538.666348648664
# 1345.8019333393772
# 1229.5436710878569
# 1155.113762320117
