from . import SLEPc
from . import typing


def bv_add(alpha: float, X: SLEPc.BV, Y: SLEPc.BV) -> None:
    r"""
    Compute :math:`X = X + \alpha Y

    :type alpha: float
    :type X: SLEPc.BV
    :type Y: SLEPc.BV

    """
    Xm = X.getMat()
    Ym = Y.getMat()
    Xm.axpy(alpha, Ym)
    X.restoreMat(Xm)
    Y.restoreMat(Ym)
    return X


def bv_conj(X: SLEPc.BV) -> None:
    r"""
    In-place conjugation of :code:`X`

    :type X: SLEPc.BV

    :rtype: SLEPc.BV
    """
    Xm = X.getMat()
    Xm.conjugate()
    X.restoreMat(Xm)


def bv_real(X: SLEPc.BV, inplace: typing.Optional[bool] = False) -> SLEPc.BV:
    r"""
    Returns the real part of the BV structure :math:`\text{Re}(X)`

    :type X: SLEPc.BV
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new SLEPc.BV structure
    :type inplace: Optional[bool], default is False

    :rtype: SLEPc.BV
    """
    Y = X if inplace else X.copy()
    Ym = Y.getMat()
    Ym.realPart()
    Y.restoreMat(Ym)
    return Y


def bv_imag(X: SLEPc.BV, inplace: typing.Optional[bool] = False) -> SLEPc.BV:
    r"""
    Returns the imaginary part of the BV structure :math:`\text{Im}(X)`

    :type X: SLEPc.BV
    :param inplace: in-place if :code:`True`, else the result is stored in a
        new SLEPc.BV structure
    :type inplace: Optional[bool], default is False

    :rtype: SLEPc.BV
    """
    Y = X if inplace else X.copy()
    Ym = Y.getMat()
    Ym.imagPart()
    Y.restoreMat(Ym)
    return Y
