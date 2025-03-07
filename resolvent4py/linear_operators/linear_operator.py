from .. import np
from .. import abc
from .. import PETSc
from .. import SLEPc
from ..error_handling_functions import raise_not_implemented_error
from ..random import generate_random_petsc_vector
from ..linalg import enforce_complex_conjugacy
from ..linalg import check_complex_conjugacy

class LinearOperator(metaclass=abc.ABCMeta):
    r"""
        Abstract base class for linear operators :math:`L`

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param name: name of the linear operator
        :type name: str
        :param dimensions: row and column sizes of the linear operator
        :type dimensions: `MatSizeSpec`_
        :param nblocks: number of blocks (if the linear operator has block \
            structure)
        :type nblocks: int
    """

    def __init__(self, comm, name, dimensions, nblocks):
        self._comm = comm
        self._name = name
        self._dimensions = dimensions
        self._nblocks = nblocks
        self._real = self.check_if_real_valued()
        self._block_cc = self.check_if_complex_conjugate_structure() if \
            self._nblocks != None else None
    
    def get_comm(self):
        r"""
            The MPI communicator

            :rtype: `MPICOMM`_
        """
        return self._comm
    
    def get_name(self):
        r"""
            The name of the linear operator

            :rtype: str
        """
        return self._name
    
    def get_dimensions(self):
        r"""
            The dimensions of the linear operator

            :rtype: `MatSizeSpec`_
        """
        return self._dimensions
    
    def get_nblocks(self):
        r"""
            The number of blocks of the linear operator
            
            :rtype: int
        """
        return self._nblocks
    
    def create_right_vector(self):
        r"""
            :return: a PETSc vector that :math:`L` can be multiplied against
            :rtype: `StandardVec`_
        """
        vec = PETSc.Vec().create(comm=self._comm)
        vec.setSizes(self._dimensions[-1])
        vec.setUp()
        return vec

    def create_right_bv(self, ncols: int):
        r"""
            :param ncols: number of columns in the BV
            :param type: int

            :return: a SLEPc BV that :math:`L` can be multiplied against
            :rtype: `BV`_
        """
        bv = SLEPc.BV().create(comm=self._comm)
        bv.setSizes(self._dimensions[-1], ncols)
        bv.setType('mat')
        return bv

    def create_left_vector(self):
        r"""
            :return: a PETSc vector where :math:`Lx` can be stored into
            :rtype: `StandardVec`_
        """
        vec = PETSc.Vec().create(comm=self._comm)
        vec.setSizes(self._dimensions[0])
        vec.setUp()
        return vec

    def create_left_bv(self, ncols: int):
        r"""
            :param ncols: number of columns in the BV
            :param type: int

            :return: a SLEPc BV where :math:`LX` can be stored into
            :rtype: `BV`_
        """
        bv = SLEPc.BV().create(comm=self._comm)
        bv.setSizes(self._dimensions[0], ncols)
        bv.setType('mat')
        return bv
    
    def check_if_real_valued(self):
        r"""
            :return: :code:`True` if the linear operator is real-valued, 
                :code:`False` otherwise
            :rtype: bool
        """
        sizes = self.get_dimensions()[-1]
        array = np.random.randn(sizes[0])
        x = PETSc.Vec().createWithArray(array, sizes, None, self._comm)
        Lx = self.apply(x)
        Lxai = Lx.getArray().imag
        norm = np.sqrt(sum(self._comm.allgather(np.linalg.norm(Lxai)**2)))
        result = True if norm <= 1e-14 else False
        x.destroy()
        Lx.destroy()
        return result
    
    def check_if_complex_conjugate_structure(self):
        r"""
            Given a vector

            .. math::
                x = \left(\ldots,x_{-1},x_{0},x_{1},\ldots\right)

            with vector-valued entries that satisfy \
            :math:`x_{-i} = \overline{x_i}`, check if the vector :math:`Lx` \
            satisfies :math:`(Lx)_{-i}=\overline{(Lx)_{i}}`. (Here, the \
            overline denote complex conjugation.)

            :return: :code:`True` if the linear operator has complex-conjugate
                structure, :code:`False` otherwise.
            :rtype: bool
        """
        x = generate_random_petsc_vector(self._comm, self._dimensions[-1])
        enforce_complex_conjugacy(self._comm, x, self._nblocks)
        cc_x = check_complex_conjugacy(self._comm, x, self._nblocks)
        if cc_x == False:
            raise ValueError(
                f"Error from {self.get_name()}.check_if_complex_conjugate"
                f"_structure(): complex conjugacy was not enforced "
                f"appropriately."
            )
        Lx = self.apply(x)
        result = check_complex_conjugacy(self._comm, Lx, self._nblocks)
        x.destroy()
        Lx.destroy()
        return result

    # Methods that must be implemented by subclasses
    @abc.abstractmethod
    def apply(self, x, y=None):
        r"""
            Compute :math:`y = Lx`

            :param x: a PETSc vector
            :type x: `StandardVec`_
            :param y: [optional] a PETSc vector to store the result
            :type y: `StandardVec`_

            :rtype: `StandardVec`_
        """
    
    @abc.abstractmethod
    def apply_mat(self, X, Y=None):
        r"""
            Compute :math:`Y = LX`

            :param X: a SLEPc BV
            :type X: `BV`_
            :param Y: [optional] a SLEPc BV to store the result
            :type Y: `BV`_

            :rtype: `BV`_
        """

    @abc.abstractmethod
    def destroy(self):
        r"""
            Destroy the PETSc objects associated with :math:`L`
        """
    
    # Methods that don't necessarily need to be implemented by subclasses
    @raise_not_implemented_error
    def apply_hermitian_transpose(self, x, y=None):
        r"""
            Compute :math:`y = L^*x`

            :param x: a PETSc vector
            :type x: `StandardVec`_
            :param y: [optional] a PETSc vector to store the result
            :type y: `StandardVec`_

            :rtype: `StandardVec`_
        """

    @raise_not_implemented_error
    def apply_hermitian_transpose_mat(self, X, Y=None):
        r"""
            Compute :math:`Y = L^*X`

            :param X: a SLEPc BV
            :type X: `BV`_
            :param Y: [optional] a SLEPc BV to store the result
            :type Y: `BV`_

            :rtype: `BV`_
        """

    @raise_not_implemented_error
    def solve(self, x, y=None):
        r"""
            Compute :math:`y = L^{-1}x`

            :param x: a PETSc vector
            :type x: `StandardVec`_
            :param y: [optional] a PETSc vector to store the result
            :type y: `StandardVec`_

            :rtype: `StandardVec`_
        """

    @raise_not_implemented_error
    def solve_mat(self, X, Y=None):
        r"""
            Compute :math:`Y = L^{-1}X`
            
            :param X: a SLEPc BV
            :type X: `BV`_
            :param Y: [optional] a SLEPc BV to store the result
            :type Y: `BV`_

            :rtype: `BV`_
        """
    
    @raise_not_implemented_error
    def solve_hermitian_transpose(self, x, y=None):
        r"""
            Compute :math:`y = L^{-*}x`

            :param x: a PETSc vector
            :type x: `StandardVec`_
            :param y: [optional] a PETSc vector to store the result
            :type y: `StandardVec`_

            :rtype: `StandardVec`_
        """

    @raise_not_implemented_error
    def solve_hermitian_transpose_mat(self, X, Y=None):
        r"""
            Compute :math:`Y = L^{-*}X`
            
            :param X: a SLEPc BV
            :type X: `BV`_
            :param Y: [optional] a SLEPc BV to store the result
            :type Y: `BV`_
            
            :rtype: `BV`_
        """
