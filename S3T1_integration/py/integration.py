import numpy as np


def weight(degree, a, b, alpha=0, beta=0):
    """
    integrate (x**degree / (x-a) ** alpha / (b-x) ** beta) from a to b
    """
    assert alpha * beta == 0, \
        f'at least one of alpha ({alpha}) or beta ({beta}) should be 0'

    if alpha == 0 and beta != 0:
        raise NotImplementedError

    if alpha != 0 and beta == 0:
        raise NotImplementedError

    k = degree + 1
    return b ** k / k - a ** k / k


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
    m = - np.log(np.abs((s2 - s1) / (s1 - s0))) / np.log(L)
    return m


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    """
    n = len(xs)
    moments = []
    for i in range(n):
        moments.append(weight(i, x0, x1))
    nodes = []
    for i in range(n):
        for v in xs:
            nodes.append(v**i)
    w = np.reshape(np.array(nodes), (n, n))
    a = np.linalg.solve(w, moments)
    return sum(a * np.array([func(x) for x in xs]))


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """
    pass


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
