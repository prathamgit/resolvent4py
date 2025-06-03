import numpy as np
import scipy as sp
import os

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)

from petsc4py import PETSc
from mpi4py import MPI
import resolvent4py as res4py


def evaluate_dynamics(t, q, params):
    mu, alpha, beta = params
    x, y, z = q
    rhs = np.zeros(3)
    rhs[0] = mu * x - y - alpha * x * z - beta * x * y
    rhs[1] = x + mu * y - alpha * y * z + beta * x**2
    rhs[2] = -alpha * z + alpha * (x**2 + y**2)
    return rhs


def evaluate_jacobian(t, Q, params):
    eps = 1e-5
    A = np.zeros((3, 3))
    for i in range(3):
        veci = np.zeros(3)
        veci[i] = 1.0
        A[:, i] = (
            evaluate_dynamics(t, Q + eps * veci, params)
            - evaluate_dynamics(t, Q - eps * veci, params)
        ) / (2 * eps)
    return A


# -------------------------------------------------------------------
# --------- Define system -------------------------------------------
# -------------------------------------------------------------------

n = 3
mu = 1 / 5
alpha = 1 / 5
beta = 1 / 5
params = [mu, alpha, beta]

omega = np.sqrt(1 - beta**2 * mu / alpha)
T = 2 * np.pi / omega

# -------------------------------------------------------------------
# --------- Compute base flow ---------------------------------------
# -------------------------------------------------------------------
dt = T / 300
time = np.arange(0, 200 * T, dt)
Q = sp.integrate.solve_ivp(
    evaluate_dynamics,
    [0, time[-1]],
    0.1 * np.ones(3),
    "RK45",
    t_eval=time,
    args=(params,),
    rtol=1e-12,
    atol=1e-12,
).y

idx0 = np.argmin(np.abs(time - 198 * T))
idxf = np.argmin(np.abs(time - 199 * T))
time = time[idx0:idxf] - time[idx0]
Q = Q[:, idx0:idxf]

# -------------------------------------------------------------------
# --------- Compute A(t) = A(t + T) ---------------------------------
# -------------------------------------------------------------------
nf = 10
As = [
    evaluate_jacobian(time[i], Q[:, i], params).reshape(-1, 1)
    for i in range(len(time))
]
As = np.concatenate(As, axis=-1)
Ashat = (1 / len(time)) * np.fft.rfft(As, axis=-1)[:, : (nf + 1)]
freqs = omega * np.arange(nf + 1)


# -------------------------------------------------------------------
# --------- Save the Fourier coefficients of A(t) -------------------
# -------------------------------------------------------------------
save_path = "data/"
os.makedirs(save_path) if not os.path.exists(save_path) else None

for j in range(Ashat.shape[-1]):
    Aj = Ashat[:, j].reshape((3, 3))
    Aj = sp.sparse.coo_matrix(Aj)
    rows = Aj.row
    cols = Aj.col
    data = Aj.data

    idces = np.argwhere(np.abs(data) >= 1e-16)
    rows = rows[idces]
    cols = cols[idces]
    data = data[idces]

    arrays = [rows, cols, data]
    fnames = ["rows_%02d.dat" % j, "cols_%02d.dat" % j, "vals_%02d.dat" % j]
    for i, array in enumerate(arrays):
        fname = save_path + fnames[i]
        vec = PETSc.Vec().createWithArray(
            array, len(array), None, MPI.COMM_SELF
        )
        res4py.write_to_file(MPI.COMM_WORLD, fname, vec)

    Aj = Aj.todense()
    Aj_mat = PETSc.Mat().createDense((3, 3), None, Aj, MPI.COMM_SELF)
    res4py.write_to_file(MPI.COMM_WORLD, save_path + "Aj_%02d.dat" % j, Aj_mat)

np.save(save_path + "bflow_freqs.npy", freqs)
