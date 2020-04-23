import numpy as np


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    compute 0..max_s moments of the weight p(x) = 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
    """
    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    raise NotImplementedError


def runge(s0, s1, m, L):
    """
    estimate m-degree errors for s0 and s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    estimate accuracy degree
    s0, s1, s2: consecutive composite quads
    return: accuracy degree estimation
    """
    raise NotImplementedError


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    **kwargs passed to moments()
    """
    raise NotImplementedError


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """
    raise NotImplementedError


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n_intervals: number of intervals
    n_nodes: number of nodes on each interval
    """
    raise NotImplementedError


def integrate(func, x0, x1, tol):
    """
    integrate with error <= tol
    return: result, error estimation
    """
    raise NotImplementedError
