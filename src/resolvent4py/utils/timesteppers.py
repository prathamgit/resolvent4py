import numpy as np
from mpi4py import MPI
import petsc4py.PETSc as PETSc
import slepc4py.SLEPc as SLEPc
from typing import Union, Tuple, Optional, Literal
from ..utils.matrix import create_dense_matrix

TimeSteppingScheme = Literal['RK4', 'BE', 'CN', 'rk4', 'be', 'cn']

_METHOD_HANDLERS = {}
_METHOD_SETUP = {}
_METHOD_SETUP_CACHE = {}

def register_setup(name: str):
    def decorator(handler_func):
        _METHOD_SETUP[name.upper()] = handler_func
        return handler_func
    return decorator

@register_setup('RK4')
def _setup_rk4(lin_op, u0, dt):
    _, num_vecs = u0.getSizes()
    
    k1 = u0.duplicate()
    k2 = u0.duplicate()
    k3 = u0.duplicate()
    k4 = u0.duplicate()
    temp = u0.duplicate()
    u_next = u0.duplicate()
    f_sum = u0.duplicate()

    Q_ident = create_dense_matrix(MPI.COMM_SELF, (num_vecs, num_vecs))
    for i in range(num_vecs):
        Q_ident.setValue(i, i, 1.0)
    Q_ident.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    Q_ident.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

    return (k1, k2, k3, k4, temp, u_next, f_sum, Q_ident)

@register_setup('BE')
def _setup_be(lin_op, u0, dt):
    from ..utils.ksp import create_gmres_bjacobi_solver
    comm = u0.comm
    _, num_vecs = u0.getSizes()

    Q_ident = create_dense_matrix(MPI.COMM_SELF, (num_vecs, num_vecs))
    for i in range(num_vecs):
        Q_ident.setValue(i, i, 1.0)
    Q_ident.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    Q_ident.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
    
    A = lin_op.A.duplicate(copy=True)
    A.scale(-dt)

    I_mat = PETSc.Mat().createAIJ(A.getSizes(), comm=comm, nnz=1) 
    I_mat.setDiagonal(PETSc.Vec().createWithArray([1.0]*A.getLocalSize()[0], comm=PETSc.COMM_SELF))
    I_mat.assemble()

    A.axpy(1.0, I_mat, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
    I_mat.destroy()

    rhs = u0.duplicate()
    u_next = u0.duplicate()
    ksp = create_gmres_bjacobi_solver(comm, A, nblocks=comm.Get_size())
    
    return (rhs, u_next, Q_ident, ksp)

@register_setup('CN')
def _setup_cn(lin_op, u0, dt):
    from ..utils.ksp import create_gmres_bjacobi_solver
    comm = u0.comm
    _, num_vecs = u0.getSizes()

    Q_ident = create_dense_matrix(MPI.COMM_SELF, (num_vecs, num_vecs))
    for i in range(num_vecs):
        Q_ident.setValue(i, i, 1.0)
    Q_ident.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
    Q_ident.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

    A_lhs = lin_op.A.duplicate(copy=True)
    A_lhs.scale(-0.5*dt)
    I_mat = PETSc.Mat().createAIJ(A_lhs.getSizes(), comm=comm, nnz=1)
    I_mat.setDiagonal(PETSc.Vec().createWithArray([1.0]*A_lhs.getLocalSize()[0], comm=PETSc.COMM_SELF))
    I_mat.assemble()
    A_lhs.axpy(1.0, I_mat, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
    
    A_rhs_op_part = lin_op.A.duplicate(copy=True)
    A_rhs_op_part.scale(0.5*dt)
    A_rhs_op_part.axpy(1.0, I_mat, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN) 
    I_mat.destroy()

    rhs = u0.duplicate()
    temp_op_eval_out = u0.duplicate()
    u_next = u0.duplicate()
    ksp = create_gmres_bjacobi_solver(comm, A_lhs, nblocks=comm.Get_size())

    return (rhs, temp_op_eval_out, u_next, Q_ident, ksp)


def register_method(name: str):
    """Decorator to register a time-stepping method."""
    def decorator(handler_func):
        _METHOD_HANDLERS[name.upper()] = handler_func
        return handler_func
    return decorator

@register_method('RK4')
def _handle_rk4(lin_op, u0, dt, timestepping_intermediates, f):
    """Handler for Runge-Kutta 4th order."""
    if f is not None and not isinstance(f, tuple):
        f = (f, f)
    return runge_kutta4(lin_op, u0, dt, timestepping_intermediates, f)

@register_method('BE')
def _handle_be(lin_op, u0, dt, timestepping_intermediates, f):
    """Handler for Backward Euler."""
    if f is not None and isinstance(f, tuple):
        if len(f) > 1:
            raise ValueError("For Backward Euler, forcing term should be a single vector, not a tuple")
        f = f[0]
    return backward_euler(lin_op, u0, dt, timestepping_intermediates, f)

@register_method('CN')
def _handle_cn(lin_op, u0, dt, timestepping_intermediates, f):
    """Handler for Crank-Nicolson."""
    if f is not None and not isinstance(f, tuple):
        f = (f, f)
    return crank_nicolson(lin_op, u0, dt, timestepping_intermediates, f)

def get_available_methods() -> Tuple[str, ...]:
    """Get available time-stepping method names."""
    return tuple(_METHOD_HANDLERS.keys())

def setup(lin_op: 'LinearOperator', u0: SLEPc.BV, dt: float, method: TimeSteppingScheme = 'RK4'):
    method_upper = method.upper()
    handler = _METHOD_SETUP.get(method_upper)

    if handler is None:
        available = ', '.join(f"'{m}'" for m in get_available_methods())
        raise ValueError(f"Unknown time-stepping method: '{method}'. "
                        f"Available methods are: {available}")

    timestepping_intermediates = handler(lin_op, u0, dt)
    _METHOD_SETUP_CACHE[method_upper] = timestepping_intermediates

def timestep(lin_op: 'LinearOperator', u0: SLEPc.BV, dt: float, 
            f: Optional[Union[SLEPc.BV, Tuple[SLEPc.BV, SLEPc.BV]]] = None,
            method: TimeSteppingScheme = 'RK4') -> SLEPc.BV:
    """Perform one time step using a specified method (RK4, BE, CN)."""
    method_upper = method.upper()
    handler = _METHOD_HANDLERS.get(method_upper)
    timestepping_intermediates = _METHOD_SETUP_CACHE.get(method_upper)
    
    if handler is None:
        available = ', '.join(f"'{m}'" for m in get_available_methods())
        raise ValueError(f"Unknown time-stepping method: '{method}'. "
                        f"Available methods are: {available}")
    
    return handler(lin_op, u0, dt, timestepping_intermediates, f)

def runge_kutta4(lin_op: 'LinearOperator', u0: SLEPc.BV, dt: float, timestepping_intermediates: Tuple[any, ...], f: Optional[Tuple[SLEPc.BV, SLEPc.BV]] = None) -> SLEPc.BV:
    """Perform one time step using 4th-order Runge-Kutta."""
    (k1, k2, k3, k4, temp, u_next, f_sum, Q_ident) = timestepping_intermediates

    f_sum.mult(1.0, 0.0, f[0], Q_ident)
    f_sum.mult(0.5, 0.5, f[1], Q_ident)

    lin_op.apply_mat(u0, k1)
    if f is not None and f[0] is not None:
        k1.mult(1.0, 1.0, f[0], Q_ident)
    
    temp.mult(1.0, 0.0, u0, Q_ident)
    temp.mult(0.5*dt, 1.0, k1, Q_ident)
    lin_op.apply_mat(temp, k2)
    if f_sum is not None:
        k2.mult(1.0, 1.0, f_sum, Q_ident)
    
    temp.mult(1.0, 0.0, u0, Q_ident)
    temp.mult(0.5*dt, 1.0, k2, Q_ident)
    lin_op.apply_mat(temp, k3)
    if f_sum is not None:
        k3.mult(1.0, 1.0, f_sum, Q_ident)
    
    temp.mult(1.0, 0.0, u0, Q_ident)
    temp.mult(dt, 1.0, k3, Q_ident)
    lin_op.apply_mat(temp, k4)
    if f is not None and f[1] is not None:
        k4.mult(1.0, 1.0, f[1], Q_ident)
    
    u_next.mult(1.0, 0.0, u0, Q_ident)
    u_next.mult(dt/6.0, 1.0, k1, Q_ident)
    u_next.mult(dt/3.0, 1.0, k2, Q_ident)
    u_next.mult(dt/3.0, 1.0, k3, Q_ident)
    u_next.mult(dt/6.0, 1.0, k4, Q_ident)
    
    assert not np.isnan(u_next.norm()) , "NaNs already present after timestep"

    return u_next

def backward_euler(lin_op: 'LinearOperator', u0: SLEPc.BV, dt: float, timestepping_intermediates: Tuple[any, ...], f: Optional[SLEPc.BV] = None) -> SLEPc.BV:
    """Perform one time step using Backward Euler."""
    (rhs, u_next, Q_ident, ksp) = timestepping_intermediates

    rhs.copy(u0)
    if f is not None:
        rhs.mult(dt, 1.0, f, Q_ident)
    
    u0_mat = u0.getMat()
    u_next_mat = u_next.getMat()
    ksp.matSolve(u0_mat, u_next_mat)
    u0.restoreMat(u0_mat)
    u_next.restoreMat(u_next_mat)

    assert not np.isnan(u_next.norm()) , "NaNs already present after timestep"

    return u_next

def crank_nicolson(lin_op: 'LinearOperator', u0: SLEPc.BV, dt: float, timestepping_intermediates: Tuple[any, ...], f: Optional[Tuple[SLEPc.BV, SLEPc.BV]] = None) -> SLEPc.BV:
    """Perform one time step using Crank-Nicolson."""
    (rhs, temp_op_eval_out, u_next, Q_ident, ksp) = timestepping_intermediates

    rhs.copy(u0)
    if f is not None and f[0] is not None:
        rhs.mult(1.0, 0.5 * dt, f[0], Q_ident)

    lin_op.apply_mat(u0, temp_op_eval_out)
    rhs.mult(1.0, 0.5 * dt, temp_op_eval_out, Q_ident)

    if f is not None and f[1] is not None:
        rhs.mult(1.0, 0.5 * dt, f[1], Q_ident)
    
    rhs_mat = rhs.getMat()
    u_next_mat = u_next.getMat()
    ksp.matSolve(rhs_mat, u_next_mat)
    u_next.restoreMat(u_next_mat)
    rhs.restoreMat(rhs_mat)

    assert not np.isnan(u_next.norm()) , "NaNs already present after timestep"

    return u_next
    
def estimate_dt_max(lin_op: 'LinearOperator', scheme: TimeSteppingScheme = 'CN') -> float:
    """Estimate max stable dt for a given scheme (FE, RK4, CN)."""
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
    elif scheme == "BE":
        dt_max = np.inf  # Backward Euler is A-stable
    elif scheme == "CN":
        dt_max = 2.0 / rho  # Technically CN is A-stable, but we provide a practical limit
    else:
        raise ValueError(f"Unknown scheme '{scheme}'")

    eps.destroy()
    return dt_max