from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc
from math import ceil
import copy

from ..linalg import enforce_complex_conjugacy
from ..mat_helpers import create_dense_matrix

np.seterr(all='raise')

def estimate_dt_max(lin_op, scheme="CN"):
    A = lin_op.A
    comm = lin_op.get_comm()

    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    try:
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    except AttributeError:
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    eps.setDimensions(nev=1)
    eps.setTolerances(1e-7)
    eps.setFromOptions()
    eps.solve()

    if eps.getConverged() == 0:
        raise RuntimeError(
            "Eigenvalue solver failed while estimating Δt stability limit."
        )

    eig = eps.getEigenpair(0)
    if isinstance(eig, tuple):
        lam_max = complex(eig[0], eig[1])
    else:
        lam_max = eig
    rho = abs(lam_max)
    if rho == 0.0:
        return np.inf

    scheme = scheme.upper()
    if scheme == "FE":
        dt_max = 2.0 / rho
    elif scheme == "RK4":
        dt_max = 2.785 / rho
    elif scheme == "CN":
        dt_max = 2.0 / rho
    else:
        raise ValueError(f"Unknown scheme '{scheme}'")

    return dt_max

def construct_dft_mats(n_omega, n_timesteps, n):
    dft_mat = create_dense_matrix(MPI.COMM_SELF, (n_omega, n_omega // 2))
    i_dft_mat = create_dense_matrix(MPI.COMM_SELF, (2 * n_timesteps, n_omega // 2))
    j = np.linspace(0, 2 * n_timesteps - 1, 2 * n_timesteps)
    alpha = -np.pi * 1j / n_timesteps 
    for i in range(n_omega // 2):
        col = i_dft_mat.getDenseColumnVec(i)
        col_array = col.getArray()
        col_array[:] = np.conj(np.exp(alpha * i * j)) / n_timesteps
        i_dft_mat.restoreDenseColumnVec(i, col)
    j = np.linspace(0, n_omega - 1, n_omega)
    alpha = -2 * np.pi * 1j / n_omega
    for i in range(n_omega // 2):
        col = dft_mat.getDenseColumnVec(i)
        col_array = col.getArray()
        col_array[:] = np.exp(alpha * i * j)
        dft_mat.restoreDenseColumnVec(i, col)
    i_dft_mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    i_dft_mat.assemblyEnd  (PETSc.Mat.AssemblyType.FINAL)
    dft_mat.assemblyBegin  (PETSc.Mat.AssemblyType.FINAL)
    dft_mat.assemblyEnd    (PETSc.Mat.AssemblyType.FINAL)
    return dft_mat, i_dft_mat

def randomized_time_stepping_svd(lin_op, lin_op_mass, omega, n_periods, n_timesteps, n_rand, n_loops, n_svals):
    r"""
        Compute the SVD of the linear operator :math:`iwG - A`
        specified by :code:`lin_op` using a time-stepping randomized SVD algorithm

        :param lin_op: any child class of the :code:`LinearOperator` class
        :param n_rand: number of random vectors to use
        :type n_rand: int
        :param n_loops: number of randomized svd power iterations
        :type n_loops: int
        :param n_svals: number of singular triplets to return
        :type n_svals: int 

        :return: :math:`(U,\,\Sigma,\, V)` a 3-tuple with the leading
            :code:`n_svals` singular values and corresponding left and \
            right singular vectors
        :rtype: (SLEPc.BV with :code:`n_svals` columns,
            numpy.ndarray of size :code:`n_svals x n_svals`,
            SLEPc.BV with :code:`n_svals` columns)
    """

    n_omega = len(omega)
    t_s = 2 * np.pi / np.min(np.abs(omega[omega != 0]))

    delta_t = t_s / n_omega
    dt = delta_t / ceil(delta_t / (t_s / n_timesteps))

    comm = lin_op.get_comm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        try:
            dt_max = estimate_dt_max(lin_op, scheme="CN")
            if dt > 0.5 * dt_max:
                print(
                    f"dt = {dt:.3e} is large compared with "
                    f"the Crank-Nicolson accuracy guard (≈ {dt_max:.3e})."
                )
            else:
                print(
                    f"dt = {dt:.3e} is reasonable compared with "
                    f"the Crank-Nicolson accuracy guard (≈ {dt_max:.3e})."
                )
        except RuntimeError as err:
            print(f"Eigenvalue probe failed: {err}")

    t_ratio = int(delta_t / dt)

    omega = omega[omega >= 0]

    # Assemble random BV
    N = lin_op.get_dimensions()[0][1]
    Nl = lin_op.get_dimensions()[0][0]
    if rank == 0:
        print(f"N = {N}, k = {n_rand}, Nl = {Nl}")
        print(f"ts = {t_s}")
        print(f"deltat = {delta_t}")
        print(f"dt = {dt}")
    X = []
    for k in range(n_rand):
        X_k = SLEPc.BV().create(comm=lin_op._comm)
        X_k.setSizes(N, n_omega // 2)
        X_k.setType('vecs')
        X_k.setRandomNormal()
        X_k.orthogonalize(None)
        X.append(X_k)

    # Assemble DFT matrices
    dft_mat, i_dft_mat = construct_dft_mats(n_omega, n_timesteps, N)

    lhs = lin_op_mass.duplicate(True)
    lhs.axpy(-dt/2, lin_op)

    rhs_1 = lin_op_mass.duplicate(True)
    rhs_1.axpy(dt/2, lin_op)

    ksp = PETSc.KSP().create(comm=lin_op._comm)
    ksp.setOperators(lhs.A)
    ksp.setType('gmres')

    pc = ksp.getPC()
    pc.setType('asm')

    ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=500)
    ksp.setUp()
    ksp.setErrorIfNotConverged(True)

    subksps = pc.getASMSubKSP()

    for sub in subksps:
        sub.getPC().setType('ilu')
        sub.getPC().setFactorLevels(2)

    lin_op_mass.hermitian_transpose()
    lin_op.hermitian_transpose()

    lhs_2 = lin_op_mass.duplicate(True)
    lhs_2.axpy(-dt/2, lin_op)

    rhs_2 = lin_op_mass.duplicate(True)
    rhs_2.axpy(dt/2, lin_op)

    ksp_2 = PETSc.KSP().create(comm=lin_op._comm)
    ksp_2.setOperators(lhs_2.A)
    ksp_2.setType('gmres')

    pc_2 = ksp_2.getPC()
    pc_2.setType('asm')

    ksp_2.setTolerances(rtol=1e-10, atol=1e-10, max_it=500)
    ksp_2.setUp()
    ksp_2.setErrorIfNotConverged(True)

    subksps = pc_2.getASMSubKSP()

    for sub in subksps:
        sub.getPC().setType('ilu')
        sub.getPC().setFactorLevels(2)

    lin_op_mass.hermitian_transpose()
    lin_op.hermitian_transpose()

    # Forcing sampling function
    def sample_forcing(f, forcing_mat, idx):
        coeff = i_dft_mat.getDenseArray()[idx, :].copy()
        for i in range(n_rand): 
            f_col = f.getColumn(i);  
            col_array = f_col.getArray()
            col_array[:] = 0
            for l, c in enumerate(coeff):
                v = forcing_mat[i].getColumn(l)
                f_col.axpy(c, v)
                forcing_mat[i].restoreColumn(l, v)
            f.restoreColumn(i, f_col)

    # Timestepping functions
    def direct_action(forcing_mat, rhs_1, ksp):
        q_temp = SLEPc.BV().create(comm=lin_op._comm)
        q_temp.setSizes(N, n_rand)
        q_temp.setType('vecs')
        q_temp.scale(0.0)
        Y_hat = []
        temp_rhs = SLEPc.BV().create(comm=lin_op._comm)
        temp_rhs.setSizes(N, n_rand)
        temp_rhs.setType('vecs')
        f = SLEPc.BV().create(comm=lin_op._comm)
        f.setSizes(N, n_rand)
        f.setType('vecs')
        f_next = SLEPc.BV().create(comm=lin_op._comm)
        f_next.setSizes(N, n_rand)
        f_next.setType('vecs')
        f_sum = SLEPc.BV().create(comm=lin_op._comm)
        f_sum.setSizes(N, n_rand)
        f_sum.setType('vecs')
        for period in range(n_periods):
            for i in range(n_timesteps):
                if rank == 0:
                    print(f"timestep = {i}")
                sample_forcing(f, forcing_mat, (2*(i-1)) % (2*n_timesteps))
                sample_forcing(f_next, forcing_mat, (2*(i-1)+2) % (2*n_timesteps))
                temp_rhs.scale(0.0)
                rhs_1.apply_mat(q_temp, temp_rhs)
                f_sum.scale(0.0)
                for k in range(n_rand):
                    v_f = f.getColumn(k)
                    v_fn = f_next.getColumn(k)
                    v_sum = f_sum.getColumn(k)
                    v_sum.axpy(1.0, v_f)
                    v_sum.axpy(1.0, v_fn)
                    v_sum.scale(dt / 2)
                    v_rhs = temp_rhs.getColumn(k)
                    v_rhs.axpy(1.0, v_sum)
                    f.restoreColumn(k, v_f)
                    f_next.restoreColumn(k, v_fn)
                    f_sum.restoreColumn(k, v_sum)
                    sol_col = q_temp.getColumn(k)
                    ksp.solve(v_rhs, sol_col)
                    temp_rhs.restoreColumn(k, v_rhs)
                    q_temp.restoreColumn(k, sol_col)
                if period == n_periods - 1 and i % t_ratio == 0:
                    Y_new = SLEPc.BV().create(comm=lin_op._comm)
                    Y_new.setSizes(N, n_rand)
                    Y_new.setType('vecs')
                    q_temp.copy(Y_new)
                    Y_hat.append(Y_new)
        temp_rhs.destroy()
        f.destroy()
        f_next.destroy()
        f_sum.destroy()
        Y_temp = []
        for i in range(n_rand):
            Y_temp_i = SLEPc.BV().create(comm=lin_op._comm)
            Y_temp_i.setSizes(N, n_omega//2)
            Y_temp_i.setType('vecs')
            local_cols = []
            for y in Y_hat:
                col = y.getColumn(i)
                local_cols.append(col.getArray().copy())
                y.restoreColumn(i, col)
            dft_coeffs_local = np.column_stack(local_cols) @ dft_mat.getDenseArray().copy()
            for j in range(n_omega//2):
                col = Y_temp_i.getColumn(j)
                col_array = col.getArray()
                col_array[:] = dft_coeffs_local[:, j] * t_ratio
                Y_temp_i.restoreColumn(j, col)
            Y_temp.append(Y_temp_i)
        q_temp.destroy()
        return Y_temp

    def adjoint_action(forcing_mat, rhs_2, ksp_2):
        z_temp = SLEPc.BV().create(comm=lin_op._comm)
        z_temp.setSizes(N, n_rand)
        z_temp.setType('vecs')
        z_temp.scale(0.0)
        S_hat = []
        temp_rhs = SLEPc.BV().create(comm=lin_op._comm)
        temp_rhs.setSizes(N, n_rand)
        temp_rhs.setType('vecs')
        s = SLEPc.BV().create(comm=lin_op._comm)
        s.setSizes(N, n_rand)
        s.setType('vecs')
        s_next = SLEPc.BV().create(comm=lin_op._comm)
        s_next.setSizes(N, n_rand)
        s_next.setType('vecs')
        s_sum = SLEPc.BV().create(comm=lin_op._comm)
        s_sum.setSizes(N, n_rand)
        s_sum.setType('vecs')
        for period in range(n_periods):
            for i in range(n_timesteps):
                if rank == 0:
                    print(f"timestep = {i}")
                sample_forcing(s, forcing_mat, (2*(n_timesteps - i)) % (2*n_timesteps))
                sample_forcing(s_next, forcing_mat, (2*(n_timesteps - i) - 2) % (2*n_timesteps))
                temp_rhs.scale(0.0)
                rhs_2.apply_mat(z_temp, temp_rhs)
                s_sum.scale(0.0)
                for k in range(n_rand):
                    sol_col = z_temp.getColumn(k)
                    v_rhs = temp_rhs.getColumn(k)
                    v_s = s.getColumn(k)
                    v_sn = s_next.getColumn(k)
                    v_sum = s_sum.getColumn(k)
                    v_sum.axpy(1.0, v_s)
                    v_sum.axpy(1.0, v_sn)
                    v_sum.scale(dt / 2)
                    v_rhs.axpy(1.0, v_sum)
                    ksp_2.solve(v_rhs, sol_col)
                    s.restoreColumn(k, v_s)
                    s_next.restoreColumn(k, v_sn)
                    s_sum.restoreColumn(k, v_sum)
                    temp_rhs.restoreColumn(k, v_rhs)
                    z_temp.restoreColumn(k, sol_col)
                    
                if period == n_periods - 1 and i % t_ratio == 0:
                    S_new = SLEPc.BV().create(comm=lin_op._comm)
                    S_new.setSizes(N, n_rand)
                    S_new.setType('vecs')
                    z_temp.copy(S_new)
                    S_hat.append(S_new)
        temp_rhs.destroy()
        s.destroy()
        s_next.destroy()
        s_sum.destroy()
        S_hat.reverse()
        S_temp = []
        for i in range(n_rand):
            S_temp_i = SLEPc.BV().create(comm=lin_op._comm)
            S_temp_i.setSizes(N, n_omega//2)
            S_temp_i.setType('vecs')
            local_cols = []
            for s in S_hat:
                col = s.getColumn(i)
                local_cols.append(col.getArray().copy())
                s.restoreColumn(i, col)
            dft_coeffs_local = np.column_stack(local_cols) @ dft_mat.getDenseArray().copy()
            for j in range(n_omega//2):
                col = S_temp_i.getColumn(j)
                col_array = col.getArray()
                col_array[:] = dft_coeffs_local[:, j] * t_ratio
                S_temp_i.restoreColumn(j, col)
            S_temp.append(S_temp_i)
        z_temp.destroy()
        return S_temp

    def permute_mat_to_Nw_list(mat):
        mat_hat_mod = []
        for ww in range(n_omega // 2):
            mat_tt = SLEPc.BV().create(comm=lin_op._comm)
            mat_tt.setSizes(N, n_rand)
            mat_tt.setType('vecs')
            for k in range(n_rand):
                curr = mat[k]
                col = curr.getColumn(ww)
                col_tt = mat_tt.getColumn(k)
                col_arr = col.getArray()
                col_arr_tt = col_tt.getArray()
                col_arr_tt[:] = col_arr
                mat_tt.restoreColumn(k, col_tt)
                curr.restoreColumn(ww, col)
            mat_hat_mod.append(mat_tt)
        for k in range(n_rand):
            mat[k].destroy()
        return mat_hat_mod
    
    def permute_mat_to_k_list(mat):
        mat_hat_mod = []
        for k in range(n_rand):
            mat_tt = SLEPc.BV().create(comm=lin_op._comm)
            mat_tt.setSizes(N, n_omega // 2)
            mat_tt.setType('vecs')
            for ww in range(n_omega // 2):
                curr = mat[ww]
                col = curr.getColumn(k)
                col_tt = mat_tt.getColumn(ww)
                col_arr = col.getArray()
                col_arr_tt = col_tt.getArray()
                col_arr_tt[:] = col_arr
                mat_tt.restoreColumn(ww, col_tt)
                curr.restoreColumn(k, col)
            mat_hat_mod.append(mat_tt)
        for ww in range(n_omega // 2):
            mat[ww].destroy()
        return mat_hat_mod

    def orthogonalize_timestepped_mat(Z):
        for i in range(n_omega // 2):
            Z_i = Z[i]
            Z_i.orthogonalize(None)
            Z[i] = Z_i
        return Z

    Y_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(direct_action(X, rhs_1, ksp))))
    for q in range(n_loops):
        S_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(adjoint_action(Y_hat, rhs_2, ksp_2))))
        Y_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(direct_action(S_hat, rhs_1, ksp))))
    S_hat = orthogonalize_timestepped_mat(permute_mat_to_Nw_list(adjoint_action(Y_hat, rhs_2, ksp_2)))

    Y_hat = permute_mat_to_Nw_list(Y_hat)
    S = PETSc.Mat().createDense([n_omega // 2, n_svals], comm=MPI.COMM_SELF)
    S.setUp()
    U = []
    V = []
    for i in range(n_omega//2):
        S_temp = S_hat[i].getMat()
        S_local = S_temp.getDenseArray().copy().T
        S_hat[i].restoreMat(S_temp)
        u_tilde, s, vh = sp.linalg.svd(S_local, full_matrices=False)
        r = min(len(s), n_svals)
        row = np.zeros(n_svals, dtype=s.dtype)
        row[:r] = s[:r]
        S[i, :] = row
        factor = u_tilde[:, :r] @ np.diag(s[:r])
        Q = Y_hat[i].getMat()
        U_arr   = Q.getDenseArray().copy() @ factor
        Y_hat[i].restoreMat(Q)
        U_i = SLEPc.BV().create(comm=lin_op._comm)
        U_i.setSizes(N, n_svals)
        U_i.setType('vecs')
        for j in range(n_svals):
            col = U_i.getColumn(j)
            if j < r:
                col_array = col.getArray()
                col_array[:] = U_arr[:, j]
            else:
                col_array = col.getArray()
                col_array[:] = 0
            U_i.restoreColumn(j, col)
        V_i = SLEPc.BV().create(comm=lin_op._comm)
        V_i.setSizes(N, n_svals)
        V_i.setType('vecs')
        v = vh.conj().T
        for j in range(n_svals):
            col = V_i.getColumn(j)
            if j < r:
                col_array = col.getArray()
                col_array[:] = v[:N, j]
            else:
                col_array = col.getArray()
                col_array[:] = 0
            V_i.restoreColumn(j, col)
        U.append(U_i)
        V.append(V_i)

    return U, S, V
