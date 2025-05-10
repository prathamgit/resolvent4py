import os

import matplotlib.pyplot as plt
import numpy as np
import resolvent4py as res4py
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc

import pathlib
import cgl

def save_bv_list(bv_list, prefix, save_path):
    save_dir = pathlib.Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, bv in enumerate(bv_list):
        nv = bv.getSizes()[1]
        for j in range(nv):
            vec = bv.getColumn(j)
            fname = save_dir / f"{prefix}_freq{i:02d}_mode{j:02d}.petsc"
            viewer = PETSc.Viewer().createBinary(str(fname), "w", comm=vec.comm)
            vec.view(viewer)
            viewer.destroy()
            bv.restoreColumn(j, vec)

def ensure_structural_diagonal(mat, value_if_empty=0):
    r0, r1 = mat.getOwnershipRange()
    diag    = mat.getDiagonal()
    missing = (diag.getArray() == 0)
    diag.destroy()
    if not missing.any():
        return
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)
    for local_i, hole in enumerate(missing):
        if hole:
            global_i = r0 + local_i
            mat.setValue(global_i, global_i, value_if_empty,
                         addv=PETSc.InsertMode.INSERT_VALUES)
    mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    mat.assemblyEnd  (PETSc.Mat.AssemblyType.FINAL)

def shift_matrix_by_matrix(A, G, alpha):
    A.axpy(-alpha, G)
    A.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    A.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": False,
    }
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
save_path = "results/" 

# Read the A matrix from file
res4py.petscprint(comm, "Reading matrix from file...")
load_path = "data/"
N = 73084
Nl = res4py.compute_local_size(N)
sizes = ((Nl, N), (Nl, N))
names = [
    load_path + "rows.dat",
    load_path + "cols.dat",
    load_path + "vals.dat",
]
A = res4py.read_coo_matrix(comm, names, sizes)
names = [
    load_path + "rowsG.dat",
    load_path + "colsG.dat",
    load_path + "valsG.dat",
]
G = res4py.read_coo_matrix(comm, names, sizes)
# ensure_structural_diagonal(A)
# ensure_structural_diagonal(G)
# shift_matrix_by_matrix(A,G,1)

# Compute the SVD of the resolvent operator R = inv(1j*omega*I - A) using
# the randomized SVD algorithm
s = 0.0183

ksp = res4py.create_gmres_bjacobi_solver(comm, A, nblocks=comm.Get_size())
L = res4py.MatrixLinearOperator(comm, A, ksp)
ksp2 = res4py.create_gmres_bjacobi_solver(comm, G, nblocks=comm.Get_size())
L_mass = res4py.MatrixLinearOperator(comm, G, ksp2)

# Compute the svd
res4py.petscprint(comm, "Running randomized SVD...")
n_periods = 40
n_timesteps = 12800
n_rand = 10
n_loops = 5
n_svals = 3

U, S, V = res4py.linalg.randomized_time_stepping_svd(L, L_mass, np.array([-2*s, -s, 0, s]), n_periods, n_timesteps, n_rand, n_loops, n_svals)

if rank == 0:
    save_bv_list(U, "U", save_path)
    save_bv_list(V, "V", save_path)

# S.assemble()

# if comm.rank == 0:
#     pathlib.Path(save_path).mkdir(exist_ok=True)
#     s_fname = os.path.join(save_path, "S.petsc")
#     viewer = PETSc.Viewer().createBinary(s_fname, "w", comm=comm)
#     S.view(viewer)
#     viewer.destroy()

# S.destroy()
# for bv in U: bv.destroy()
# for bv in V: bv.destroy()
# s_local=(10, 73084)
# 325.6156641578527
# u_tilde=(10, 3)
# (0.996982690886429+0.07733680658739268j)
# (-0.9981321870309865+0.059603665428424366j)
# s=(3,)
# vh=(3, 73084)
# (0.025157718154200863+0.011752109604101437j)
# (-0.0269402382166987-0.012570437097104757j)
# [106.85947509 146.4574703  210.23818948]
# s_local=(10, 73084)
# 680.9238949985848
# u_tilde=(10, 3)
# (0.9994169495200661-0.03349989664725789j)
# (-0.9999996095404631+0.0008357266221214785j)
# s=(3,)
# vh=(3, 73084)
# (0.06102598773881543+0.007186024872089215j)
# (-0.056041989580329625-0.010288256274822901j)
# [120.33645409 287.29204463 592.6511671 ]
#
#
