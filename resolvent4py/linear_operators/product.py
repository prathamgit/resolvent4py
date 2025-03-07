from .linear_operator import LinearOperator
from .. import SLEPc

class ProductLinearOperator(LinearOperator):
    r"""
        Linear operator of the form

        .. math::

            L = L_{r}L_{r-1}\ldots L_{2}L_{1}

        where :math:`L_i` are themselves linear operators.
        For example, by setting :code:`linops = (L2, L1)` and 
        :code:`linops_actions = (L2.solve, L1.apply_hermitian_transpose)` 
        we define a linear operator

        .. math:

            L = L_2^{-1}L_1^*.


        :param comm: MPI communicator
        :param linops: tuple of linear operators. (See example above.)
        :param linops_actions: tuple of actions (one of :code:`apply`, 
            :code:`solve`, :code:`apply_hermitian_transpose` or 
            :code:`solve_hermitian_transpose`) to define the overall action
            of the product operator :math:`L`. (See example above.)
        :param nblocks: [optional] number of blocks (if the linear operator \
            has block structure). This must be an odd number.
        :type nblocks: int
    """
    def __init__(self, comm, linops, linops_actions, nblocks=None):
        
        linops.reverse()
        linops_actions.reverse()
        if len(linops) == 1:
            raise ValueError (
                f"ProductLinearOperator is the product of at least two "
                f"linear operators. Only 1 was provided at initialization."
            )
        if len(linops) != len(linops_actions):
            raise ValueError (
                f"len(linops) must be equal to len(linops_actions)."
            )
        
        self.linops = linops
        self.actions = linops_actions
        self.actions_hermitian_transpose = []
        self.actions_mat = []
        self.actions_hermitian_transpose_mat = []
        self.names = []
        for (i, linop) in enumerate(self.linops):
            name = 'L%03d'%i
            self.names.append(name)
            setattr(self, name, linop)
            if self.actions[i] == linop.apply:
                self.actions_hermitian_transpose.append(\
                    linop.apply_hermitian_transpose)
                self.actions_mat.append(linop.apply_mat)
                self.actions_hermitian_transpose_mat.append(\
                    linop.apply_hermitian_transpose_mat)
            elif self.actions[i] == linop.apply_hermitian_transpose:
                self.actions_hermitian_transpose.append(\
                    linop.apply)
                self.actions_mat.append(linop.apply_hermitian_transpose_mat)
                self.actions_hermitian_transpose_mat.append(\
                    linop.apply_mat)
            elif self.actions[i] == linop.solve:
                self.actions_hermitian_transpose.append(\
                    linop.solve_hermitian_transpose)
                self.actions_mat.append(linop.solve_mat)
                self.actions_hermitian_transpose_mat.append(\
                    linop.solve_hermitian_transpose_mat)
            elif self.actions[i] == linop.solve_hermitian_transpose:
                self.actions_hermitian_transpose.append(\
                    linop.solve)
                self.actions_mat.append(linop.solve_hermitian_transpose_mat)
                self.actions_hermitian_transpose_mat.append(\
                    linop.solve_mat)
            else:
                raise ValueError (
                    f"Action for the {i}th linear operator in the list must be "
                    f"one of apply, solve, apply_hermitian_transpose or "
                    f"solve_hermitian_transpose."
                )
        
        self.actions_hermitian_transpose.reverse()
        self.actions_hermitian_transpose_mat.reverse()
        
        self.nlops = len(self.names)
        self.create_intermediate_vectors()
        L1 = getattr(self, self.names[0])
        Ln = getattr(self, self.names[-1])
        dimr = Ln._dimensions[0] if self.actions[-1] == Ln.apply or \
            self.actions[-1] == Ln.solve else Ln._dimensions[-1]
        dimc = L1._dimensions[-1] if self.actions[0] == L1.apply or \
            self.actions[0] == L1.solve else Ln._dimensions[0]
        dims = (dimr, dimc)
        super().__init__(comm, 'ProductLinearOperator', dims, nblocks)
    
    def create_intermediate_vectors(self):
        self.intermediate_vecs = []
        for j in range (self.nlops - 1):
            L = getattr(self, self.names[j])
            if self.actions[j] == L.apply or self.actions[j] == L.solve:
                self.intermediate_vecs.append(L.create_left_vector())
            else:
                self.intermediate_vecs.append(L.create_right_vector())
        self.intermediate_vecs_hermitian_transpose = []

        self.names.reverse()
        self.actions.reverse()
        for j in range (self.nlops - 1):
            L = getattr(self, self.names[j])
            if self.actions[j] == L.apply or self.actions[j] == L.solve:
                self.intermediate_vecs_hermitian_transpose.append(\
                    L.create_right_vector())
            else:
                self.intermediate_vecs_hermitian_transpose.append(\
                    L.create_left_vector())
        self.names.reverse()
        self.actions.reverse()
                
    def create_intermediate_bvs(self, m):
        intermediate_bvs = []
        for j in range (self.nlops - 1):
            L = getattr(self, self.names[j])
            X = SLEPc.BV().create(self._comm)
            sz = L._dimensions[0] if self.actions[j] == L.apply or \
                self.actions[j] == L.solve else L._dimensions[-1]
            X.setSizes(sz, m)
            X.setType('mat')
            intermediate_bvs.append(X)
        return intermediate_bvs

    def create_intermediate_bvs_hermitian_transpose(self, m):
        self.names.reverse()
        self.actions.reverse()
        intermediate_bvs = []
        for j in range (self.nlops - 1):
            L = getattr(self, self.names[j])
            X = SLEPc.BV().create(self._comm)
            sz = L._dimensions[-1] if self.actions[j] == L.apply or \
                self.actions[j] == L.solve else L._dimensions[0]
            X.setSizes(sz, m)
            X.setType('mat')
            intermediate_bvs.append(X)
        self.names.reverse()
        self.actions.reverse()
        return intermediate_bvs

    def apply(self, x, y=None):
        y = self.create_left_vector() if y == None else y
        for j in range (self.nlops):
            if j == 0:
                yj = self.intermediate_vecs[j]
                yj = self.actions[j](x, yj)
                self.intermediate_vecs[j] = yj
            elif j > 0 and j < self.nlops - 1:
                yj = self.intermediate_vecs[j]
                yjm1 = self.intermediate_vecs[j-1]
                yj = self.actions[j](yjm1, yj)
                self.intermediate_vecs[j] = yj
            else:
                yjm1 = self.intermediate_vecs[j-1]
                y = self.actions[j](yjm1, y)
        return y

    def apply_hermitian_transpose(self, x, y=None):
        y = self.create_right_vector() if y == None else y
        for j in range (self.nlops):
            if j == 0:
                yj = self.intermediate_vecs_hermitian_transpose[j]
                yj = self.actions_hermitian_transpose[j](x, yj)
                self.intermediate_vecs_hermitian_transpose[j] = yj
            elif j > 0 and j < self.nlops - 1:
                yj = self.intermediate_vecs_hermitian_transpose[j]
                yjm1 = self.intermediate_vecs_hermitian_transpose[j-1]
                yj = self.actions_hermitian_transpose[j](yjm1, yj)
                self.intermediate_vecs_hermitian_transpose[j] = yj
            else:
                yjm1 = self.intermediate_vecs_hermitian_transpose[j-1]
                y = self.actions_hermitian_transpose[j](yjm1, y)
        return y
    
    def apply_mat(self, X, Y=None, Z=None):
        Y = self.create_left_bv(X.getSizes()[-1]) if Y == None else Y
        destroy = False
        if Z == None:
            Z = self.create_intermediate_bvs(X.getSizes()[-1])
            destroy = True
        for j in range (self.nlops):
            if j == 0:
                Z[j] = self.actions_mat[j](X, Z[j])
            elif j > 0 and j < self.nlops - 1:
                Z[j] = self.actions_mat[j](Z[j-1], Z[j])
            else:
                Y = self.actions_mat[j](Z[j-1], Y)
        if destroy:
            for Zj in Z: Zj.destroy()
        return Y
    
    def apply_hermitian_transpose_mat(self, X, Y=None, Z=None):
        Y = self.create_right_bv(X.getSizes()[-1]) if Y == None else Y
        destroy = False
        if Z == None:
            Z = self.create_intermediate_bvs_hermitian_transpose(\
                X.getSizes()[-1])
            destroy = True
        for j in range (self.nlops):
            if j == 0:
                Z[j] = self.actions_hermitian_transpose_mat[j](X, Z[j])
            elif j > 0 and j < self.nlops - 1:
                Z[j] = self.actions_hermitian_transpose_mat[j](Z[j-1], Z[j])
            else:
                Y = self.actions_hermitian_transpose_mat[j](Z[j-1], Y)
        if destroy:
            for Zj in Z: Zj.destroy()
        return Y
    
    def destroy(self):
        for name in self.names:
            getattr(self, name).destroy()