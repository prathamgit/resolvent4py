from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc
from math import ceil
import copy

from ..linear_operators import MatrixLinearOperator, ProductLinearOperator
from ..linalg import enforce_complex_conjugacy
from ..mat_helpers import create_dense_matrix, create_AIJ_identity
from ..ksp_helpers import create_mumps_solver, create_gmres_bjacobi_solver

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
    dft_mat = create_dense_matrix(MPI.COMM_WORLD, (n_omega,   n_omega // 2))
    i_dft_mat = create_dense_matrix(MPI.COMM_SELF, (2*n_timesteps, n_omega // 2))
    r0, r1 = i_dft_mat.getOwnershipRange()
    j_local = np.arange(r0, r1)
    alpha   = -np.pi * 1j / n_timesteps
    for i in range(n_omega // 2):
        col = i_dft_mat.getDenseColumnVec(i)
        col_arr = col.getArray()
        col_arr[:] = np.conj(np.exp(alpha * i * j_local)) / n_timesteps
        i_dft_mat.restoreDenseColumnVec(i, col)
    r0, r1 = dft_mat.getOwnershipRange()
    j_local = np.arange(r0, r1)
    alpha   = -2 * np.pi * 1j / n_omega
    for i in range(n_omega // 2):
        col       = dft_mat.getDenseColumnVec(i)
        col_arr   = col.getArray()
        col_arr[:] = np.exp(alpha * i * j_local)
        dft_mat.restoreDenseColumnVec(i, col)
    for M in (i_dft_mat, dft_mat):
        M.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        M.assemblyEnd  (PETSc.Mat.AssemblyType.FINAL)
    return dft_mat, i_dft_mat.transpose()

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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
    print(f"N = {N}, k = {n_rand}, Nl = {Nl}")
    print(f"ts = {t_s}")
    print(f"deltat = {delta_t}")
    print(f"dt = {dt}")
    print(f"t_ratio = {t_ratio}")
    X = []
    for k in range(n_rand):
        comm.barrier()
        X_k = SLEPc.BV().create(comm=MPI.COMM_WORLD)
        X_k.setSizes(N, n_omega // 2)
        X_k.setType('vecs')
        X_k.setRandomNormal()
        X_k.orthogonalize(None)
        X.append(X_k)

    id_mat = create_dense_matrix(MPI.COMM_SELF, (n_rand, n_rand))
    for i in range(n_rand):
        id_mat.setValue(i, i, 1.0)
    id_mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    id_mat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

    # Assemble DFT matrices
    dft_mat, i_dft_mat = construct_dft_mats(n_omega, n_timesteps, N)

    # rhs_mat = lin_op_mass.A.copy()
    # rhs_mat.axpy(dt/2, lin_op.A)
    # ksp = create_gmres_bjacobi_solver(comm, rhs_mat, nblocks=comm.Get_size())
    # ksp = create_mumps_solver(comm, rhs_mat)
    # rhs_1 = MatrixLinearOperator(comm, rhs_mat, ksp)
    
    # lhs_mat = lin_op_mass.A.copy()
    # lhs_mat.axpy(-dt, lin_op.A)
    # ksp2 = create_gmres_bjacobi_solver(comm, lhs_mat, nblocks=comm.Get_size())
    # ksp2 = create_mumps_solver(comm, lhs_mat)
    # ksp2 = PETSc.KSP().create(comm=MPI.COMM_WORLD)
    # ksp2.setOperators(lhs_mat)
    # ksp2.setType('fgmres')

    # pc = ksp.getPC()
    # pc = ksp2.getPC()
    # pc.setType('asm')
    # pc.setReusePreconditioner(True)

    # ksp2.setTolerances(rtol=1e-10, atol=1e-10, max_it=500)
    # ksp2.setInitialGuessNonzero(False)
    # ksp2.setUp()
    # ksp2.setErrorIfNotConverged(True)

    # subksps = pc.getASMSubKSP()

    # for sub in subksps:
    #     sub.getPC().setType('ilu')
    #     sub.getPC().setFactorLevels(0)
    # lhs = MatrixLinearOperator(comm, lhs_mat, ksp2)
    # direct_step = ProductLinearOperator(comm, [lhs, rhs_1], [lhs.solve, rhs_1.apply])

    # lin_op_mass.hermitian_transpose()
    # lin_op.hermitian_transpose()

    # rhs_mat_2 = lin_op_mass.A.copy()
    # rhs_mat_2.axpy(dt/2, lin_op.A)
    # ksp3 = create_gmres_bjacobi_solver(comm, rhs_mat_2, nblocks=comm.Get_size())
    # ksp3 = create_mumps_solver(comm, rhs_mat_2)
    # rhs_2 = MatrixLinearOperator(comm, rhs_mat_2, ksp3)
    
    # lhs_mat_2 = lin_op_mass.A.copy()
    # lhs_mat_2.axpy(-dt, lin_op.A)
    # ksp4 = create_gmres_bjacobi_solver(comm, lhs_mat_2, nblocks=comm.Get_size())
    # ksp4 = create_mumps_solver(comm, lhs_mat_2)
    # ksp4 = PETSc.KSP().create(comm=MPI.COMM_WORLD)
    # ksp4.setOperators(lhs_mat_2)
    # ksp4.setType('fgmres')

    # pc2 = ksp.getPC()
    # pc2 = ksp4.getPC()
    # pc2.setType('asm')
    # pc2.setReusePreconditioner(True)

    # ksp4.setTolerances(rtol=1e-10, atol=1e-10, max_it=500)
    # ksp4.setInitialGuessNonzero(False)
    # ksp4.setUp()
    # ksp4.setErrorIfNotConverged(True)

    # subksps2 = pc2.getASMSubKSP()

    # for sub in subksps2:
    #     sub.getPC().setType('ilu')
    #     sub.getPC().setFactorLevels(0)
    # lhs_2 = MatrixLinearOperator(comm, lhs_mat_2, ksp4)
    # adjoint_step = ProductLinearOperator(comm, [lhs_2, rhs_2], [lhs_2.solve, rhs_2.apply])

    # lin_op_mass.hermitian_transpose()
    # lin_op.hermitian_transpose()

    q_temp = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    q_temp.setSizes(N, n_rand)
    q_temp.setType('vecs')

    k1 = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    k1.setSizes(N, n_rand)
    k1.setType('vecs')

    k2 = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    k2.setSizes(N, n_rand)
    k2.setType('vecs')

    k3 = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    k3.setSizes(N, n_rand)
    k3.setType('vecs')

    k4 = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    k4.setSizes(N, n_rand)
    k4.setType('vecs')

    temp_rhs = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    temp_rhs.setSizes(N, n_rand)
    temp_rhs.setType('vecs')

    f = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    f.setSizes(N, n_rand)
    f.setType('vecs')

    f_next = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    f_next.setSizes(N, n_rand)
    f_next.setType('vecs')

    f_sum = SLEPc.BV().create(comm=MPI.COMM_WORLD)
    f_sum.setSizes(N, n_rand)
    f_sum.setType('vecs')

    # Forcing sampling function
    def sample_forcing(f, forcing_mat, idx):
        idft_col = i_dft_mat.getColumnVector(idx)
        for i in range(n_rand):
            f_col = f.getColumn(i)
            forcing_mat[i].multVec(1, 0, f_col, idft_col.getArray())
            f.restoreColumn(i, f_col)

    # Timestepping functions
    def direct_action(forcing_mat, rhs_1):
        print("direct_action")
        q_temp.scale(0.0)
        temp_rhs.scale(0.0)
        f.scale(0.0)
        f_next.scale(0.0)
        f_sum.scale(0.0)
        Y_hat = []
        for period in range(n_periods):
            print(f"period = {period}")
            for i in range(n_timesteps):
                print(f"timestep = {i}")
                # sample_forcing(f, forcing_mat, (2*(i-1)) % (2*n_timesteps))
                # sample_forcing(f_next, forcing_mat, (2*(i-1)+2) % (2*n_timesteps))
                # f_sum.mult(1.0, 0.0, f, id_mat)
                # f_sum.mult(dt/2, dt/2, f_next, id_mat)

                # for k in range(n_rand):
                #     v_q = q_temp.getColumn(k)
                #     v_rhs = temp_rhs.getColumn(k)
                #     v_f = f_sum.getColumn(k)
                #     direct_step.apply(v_q)
                #     lhs.solve(v_f, v_q)
                #     q_temp.restoreColumn(k, v_q)
                #     temp_rhs.restoreColumn(k, v_rhs)
                #     f_sum.restoreColumn(k, v_f)
                # direct_step.apply_mat(q_temp, temp_rhs)
                # lhs.solve_mat(f_sum, q_temp)
                # q_temp.mult(1.0, 1.0, temp_rhs, id_mat)

                # sample_forcing(f_next, forcing_mat, (2*(i-1)+2) % (2*n_timesteps))
                # f_next.scale(dt)

                # lin_op_mass.apply_mat(q_temp, temp_rhs)
                # temp_rhs.mult(1.0, 1.0, f_next, id_mat)

                # for k in range(n_rand):
                #     v_q = q_temp.getColumn(k)
                #     v_rhs = temp_rhs.getColumn(k)
                #     lhs.solve(v_rhs, v_q)
                #     temp_rhs.restoreColumn(k, v_rhs)
                #     q_temp.restoreColumn(k, v_q)
                # lhs.solve_mat(temp_rhs, q_temp)

                sample_forcing(f, forcing_mat, (2*(n_timesteps - i)) % (2*n_timesteps))
                sample_forcing(f_next, forcing_mat, (2*(n_timesteps - i) - 2) % (2*n_timesteps))
                f_sum.mult(1.0, 0.0, f, id_mat)
                f_sum.mult(0.5, 0.5, f_next, id_mat)

                lin_op.apply_mat(q_temp, k1)
                k1.mult(1.0, 1.0, f, id_mat)

                q_temp.mult(dt/2, 1.0, k1, id_mat)
                lin_op.apply_mat(q_temp, k2)
                q_temp.mult(-dt/2, 1.0, k1, id_mat)
                k2.mult(1.0, 1.0, f_sum, id_mat)

                q_temp.mult(dt/2, 1.0, k2, id_mat)
                lin_op.apply_mat(q_temp, k3)
                q_temp.mult(-dt/2, 1.0, k2, id_mat)
                k3.mult(1.0, 1.0, f_sum, id_mat)

                q_temp.mult(dt, 1.0, k3, id_mat)
                lin_op.apply_mat(q_temp, k4)
                q_temp.mult(-dt, 1.0,k3, id_mat)
                k4.mult(1.0, 1.0, f_next, id_mat)

                q_temp.mult(dt/6, 1.0, k1, id_mat)
                q_temp.mult(dt/3, 1.0, k2, id_mat)
                q_temp.mult(dt/3, 1.0, k3, id_mat)
                q_temp.mult(dt/6, 1.0, k4, id_mat)

                assert not np.isnan(q_temp.norm()) and not q_temp.norm() >= 10e10, "NaNs already present before solve"

                if period == n_periods - 1 and i % t_ratio == 0:
                    Y_new = SLEPc.BV().create(comm=MPI.COMM_WORLD)
                    Y_new.setSizes(N, n_rand)
                    Y_new.setType('vecs')
                    q_temp.copy(Y_new)
                    Y_hat.append(Y_new)
        Y_temp = []
        Y_hat = permute_mat_to_k_list_full(Y_hat)
        for i in range(n_rand):
            Y_temp_i = SLEPc.BV().create(comm=MPI.COMM_WORLD)
            Y_temp_i.setSizes(N, n_omega//2)
            Y_temp_i.setType('vecs')
            Y_temp_mat = Y_temp_i.getMat()
            
            Y_curr = Y_hat[i].getMat()
            Y_curr.matMult(dft_mat, Y_temp_mat)
            Y_hat[i].restoreMat(Y_curr)
            Y_temp_i.restoreMat(Y_temp_mat)
            Y_temp.append(Y_temp_i)
        return Y_temp

    def adjoint_action(forcing_mat, rhs_2):
        print("adjoint_action")
        q_temp.scale(0.0)
        temp_rhs.scale(0.0)
        f.scale(0.0)
        f_next.scale(0.0)
        f_sum.scale(0.0)
        S_hat = []
        for period in range(n_periods):
            print(f"period = {period}")
            for i in range(n_timesteps):
                print(f"timestep = {i}")
                # sample_forcing(f, forcing_mat, (2*(n_timesteps - i)) % (2*n_timesteps))
                # sample_forcing(f_next, forcing_mat, (2*(n_timesteps - i) - 2) % (2*n_timesteps))
                # f_sum.mult(1.0, 0.0, f, id_mat)
                # f_sum.mult(dt/2, dt/2, f_next, id_mat)

                # for k in range(n_rand):
                #     v_q = q_temp.getColumn(k)
                #     v_rhs = temp_rhs.getColumn(k)
                #     v_f = f_sum.getColumn(k)
                #     direct_step.apply(v_q)
                #     lhs.solve(v_f, v_q)
                #     q_temp.restoreColumn(k, v_q)
                #     temp_rhs.restoreColumn(k, v_rhs)
                #     f_sum.restoreColumn(k, v_f)

                # adjoint_step.apply_mat(q_temp, temp_rhs)
                # lhs_2.solve_mat(f_sum, q_temp)
                # q_temp.mult(1.0, 1.0, temp_rhs, id_mat)

                # sample_forcing(f_next, forcing_mat, (2*(n_timesteps - i) - 2) % (2*n_timesteps))
                # f_next.scale(dt)

                # lin_op_mass.apply_mat(q_temp, temp_rhs)
                # temp_rhs.mult(1.0, 1.0, f_next, id_mat)

                # for k in range(n_rand):
                #     v_q = q_temp.getColumn(k)
                #     v_rhs = temp_rhs.getColumn(k)
                #     lhs_2.solve(v_rhs, v_q)
                #     temp_rhs.restoreColumn(k, v_rhs)
                #     q_temp.restoreColumn(k, v_q)
                # lhs_2.solve_mat(temp_rhs, q_temp)

                sample_forcing(f, forcing_mat, (2*(n_timesteps - i)) % (2*n_timesteps))
                sample_forcing(f_next, forcing_mat, (2*(n_timesteps - i) - 2) % (2*n_timesteps))
                f_sum.mult(1.0, 0.0, f, id_mat)
                f_sum.mult(0.5, 0.5, f_next, id_mat)

                lin_op.apply_mat(q_temp, k1)
                k1.mult(1.0, 1.0, f, id_mat)

                q_temp.mult(dt/2, 1.0, k1, id_mat)
                lin_op.apply_mat(q_temp, k2)
                q_temp.mult(-dt/2, 1.0, k1, id_mat)
                k2.mult(1.0, 1.0, f_sum, id_mat)

                q_temp.mult(dt/2, 1.0, k2, id_mat)
                lin_op.apply_mat(q_temp, k3)
                q_temp.mult(-dt/2, 1.0, k2, id_mat)
                k3.mult(1.0, 1.0, f_sum, id_mat)

                q_temp.mult(dt, 1.0, k3, id_mat)
                lin_op.apply_mat(q_temp, k4)
                q_temp.mult(-dt, 1.0,k3, id_mat)
                k4.mult(1.0, 1.0, f_next, id_mat)

                q_temp.mult(dt/6, 1.0, k1, id_mat)
                q_temp.mult(dt/3, 1.0, k2, id_mat)
                q_temp.mult(dt/3, 1.0, k3, id_mat)
                q_temp.mult(dt/6, 1.0, k4, id_mat)

                assert not np.isnan(q_temp.norm()), "NaNs already present before solve"
                    
                if period == n_periods - 1 and i % t_ratio == 0:
                    S_new = SLEPc.BV().create(comm=MPI.COMM_WORLD)
                    S_new.setSizes(N, n_rand)
                    S_new.setType('vecs')
                    q_temp.copy(S_new)
                    S_hat.append(S_new)
        S_hat.reverse()
        S_temp = []
        S_hat = permute_mat_to_k_list_full(S_hat)
        for i in range(n_rand):
            S_temp_i = SLEPc.BV().create(comm=MPI.COMM_WORLD)
            S_temp_i.setSizes(N, n_omega//2)
            S_temp_i.setType('vecs')
            S_temp_mat = S_temp_i.getMat()
            
            S_curr = S_hat[i].getMat()
            S_curr.matMult(dft_mat, S_temp_mat)
            S_hat[i].restoreMat(S_curr)
            S_temp_i.restoreMat(S_temp_mat)
            S_temp.append(S_temp_i)
        return S_temp

    def permute_mat_to_Nw_list(mat):
        mat_hat_mod = []
        for ww in range(n_omega // 2):
            mat_tt = SLEPc.BV().create(comm=MPI.COMM_WORLD)
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
            mat_tt = SLEPc.BV().create(comm=MPI.COMM_WORLD)
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
    
    def permute_mat_to_k_list_full(mat):
        mat_hat_mod = []
        for k in range(n_rand):
            mat_tt = SLEPc.BV().create(comm=MPI.COMM_WORLD)
            mat_tt.setSizes(N, n_omega)
            mat_tt.setType('vecs')
            for ww in range(n_omega):
                curr = mat[ww]
                col = curr.getColumn(k)
                col_tt = mat_tt.getColumn(ww)
                col_arr = col.getArray()
                col_arr_tt = col_tt.getArray()
                col_arr_tt[:] = col_arr
                mat_tt.restoreColumn(ww, col_tt)
                curr.restoreColumn(k, col)
            mat_hat_mod.append(mat_tt)
        for ww in range(n_omega):
            mat[ww].destroy()
        return mat_hat_mod

    def orthogonalize_timestepped_mat(Z):
        for i in range(n_omega // 2):
            Z_i = Z[i]
            Z_i.orthogonalize(None)
            Z[i] = Z_i
        return Z

    rhs_1 = None
    rhs_2 = None

    Y_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(direct_action(X, rhs_1))))
    for q in range(n_loops):
        S_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(adjoint_action(Y_hat, rhs_2))))
        Y_hat = permute_mat_to_k_list(orthogonalize_timestepped_mat(permute_mat_to_Nw_list(direct_action(S_hat, rhs_1))))
    S_hat = permute_mat_to_Nw_list(adjoint_action(Y_hat, rhs_2))

    Y_hat = permute_mat_to_Nw_list(Y_hat)

    if rank == 0:
        S = PETSc.Mat().createDense([n_omega // 2, n_svals], comm=MPI.COMM_SELF)
        S.setUp()
    else:
        S = 0
    U = []
    V = []
    for i in range(n_omega//2):
        Y_local = Y_hat[i].getMat().getDenseArray()
        S_local = S_hat[i].getMat().getDenseArray()

        Y_all = comm.gather(Y_local, root=0)
        S_all = comm.gather(S_local, root=0)

        if rank == 0:
            Y_full = np.vstack(Y_all)
            S_local = np.vstack(S_all).T
            
            u_tilde, s, vh = sp.sparse.linalg.svds(S_local, k=n_svals)
            S[i, :] = s
            factor = u_tilde @ np.diag(s)
            U_arr = Y_full @ factor
            U_i = SLEPc.BV().create(comm=MPI.COMM_SELF)
            U_i.setSizes(N, n_svals)
            U_i.setType('vecs')
            for j in range(n_svals):
                col = U_i.getColumn(j)
                col_array = col.getArray()
                col_array[:] = U_arr[:, j]
                U_i.restoreColumn(j, col)
            V_i = SLEPc.BV().create(comm=MPI.COMM_SELF)
            V_i.setSizes(N, n_svals)
            V_i.setType('vecs')
            v = vh.conj().T
            for j in range(n_svals):
                col = V_i.getColumn(j)
                col_array = col.getArray()
                col_array[:] = v[:, j]
                V_i.restoreColumn(j, col)
            U.append(U_i)
            V.append(V_i)
            
            print(s)

    temp_rhs.destroy()
    f.destroy()
    f_next.destroy()
    f_sum.destroy()
    q_temp.destroy()

    return U, S, V
