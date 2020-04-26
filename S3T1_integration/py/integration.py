import numpy as np


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    compute 0..max_s moments of the weight p(x) = 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
    """
    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
        m = [((xr - a) ** (1 - alpha) - (xl - a) ** (1 - alpha)) / (1 - alpha)]
        for i in range(1, max_s + 1):
            coef = np.poly([-a] * i)[::-1]
            m.append(sum([coef[j] / (j + 1 - alpha) * ((xr - a) ** (j + 1 - alpha) - (xl - a) ** (j + 1 - alpha))
                          for j in range(0, i + 1)]))
        return m
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'
        m = [((b - xr) ** (1 - beta) - (b - xl) ** (1 - beta)) / (1 - beta)]
        for i in range(1, max_s + 1):
            coef = np.poly([b]*i)[::-1]
            m.append(sum([coef[j] / (j + 1 - beta) * ((b - xr) ** (j + 1 - beta) - (b - xl) ** (j + 1 - beta))
                          for j in range(0, i + 1)]))
        return m
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
    print(s0, s1, s2)
    m = - np.log(np.abs((s2 - s1) / (s1 - s0))) / np.log(L)
    return m


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    **kwargs passed to moments()
    """
    n = len(xs)
    nodes = [[(v ** i) for v in xs] for i in range(n)]
    w = np.reshape(np.array(nodes), (n, n))
    a = np.linalg.solve(w, moments(n - 1, x0, x1, **kwargs))
    return sum(a * np.array([func(x) for x in xs]))


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """
    nyu = moments(2 * n - 1, x0, x1, **kwargs)
    w = np.array([[(nyu[j + s]) for j in range(n)] for s in range(n)])
    b = - np.array(nyu[n:])
    coef = np.append(np.linalg.solve(w, b), 1)
    xs = np.roots(coef[::-1])
    nodes = [[(v ** i) for v in xs] for i in range(n)]
    w = np.reshape(np.array(nodes), (n, n))
    a = np.linalg.solve(w, nyu[:n])
    return sum(a * np.array([func(x) for x in xs]))


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n_intervals: number of intervals
    n_nodes: number of nodes on each interval
    """
    segments = np.linspace(x0, x1, n_intervals + 1)
    return sum([quad(func, segments[i], segments[i + 1], np.linspace(segments[i], segments[i + 1], n_nodes), **kwargs)
                for i in range(n_intervals)])


def integrate(func, x0, x1, tol):
    """
    integrate with error <= tol
    return: result, error estimation
    """
    n_nodes = 3
    L = 2
    h_opt = x1 - x0
    error = tol + 1
    while error >= tol:
        h = [h_opt / L**i for i in range(3)]
        n_intervals = [int((x1 - x0) / step) + 1 for step in h]
        s_h = [composite_quad(func, x0, x1, interval, n_nodes) for interval in n_intervals]
        m = aitken(*s_h, L)
        error = runge(*s_h[1:], m, L)[0]
        h_opt = s_h[2] * (tol / error) ** (1/m)
    return s_h[2], error
