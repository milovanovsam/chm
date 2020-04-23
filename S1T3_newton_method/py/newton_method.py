import numpy as np
import sympy as sp
from sympy.abc import x, y


def solve_scalar(f: sp.Expr, x0, tol):
    """
    solve scalar equation y=f(x) starting with x0
    derivative can be obtained as f.diff()
    evaluate: float(f.subs(x, x0))
    return list of x, list of y
    """
    raise NotImplementedError
    return xs, ys


def solve_plane(f: sp.Matrix, x0, y0, tol):
    """
    solve SAE {f1(x,y) = 0, f2(x,y) = 0} starting with (x0,y0)
    jacobian can be obtained as f.jacobian([x, y])
    return list of x, list of y, list of np.linalg.norm(z)
    """
    raise NotImplementedError
    return xs, ys, zs
