from .. import np


class InputOutputMatrices_Steady:

    def __init__(self, comm, compute_B, compute_C, compute_gradB):
        self.comm = comm
        self.compute_B = compute_B
        self.compute_C = compute_C
        self.compute_gradB = compute_gradB

    def compute_gradB(self, p):
        return self.compute_gradB(p)