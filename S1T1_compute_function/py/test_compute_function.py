import pytest
import numpy as np
import sympy as sp
from sympy.abc import x

from S1T1_compute_function.py import compute_function as cf


@pytest.mark.parametrize('func',
                         [
                             sp.sqrt,
                             sp.exp,
                             sp.cos,
                             sp.sin,
                             sp.atan,
                             sp.sinh,
                             sp.cosh,
                         ]
                         )
@pytest.mark.parametrize('tol', 10. ** np.array(range(-1, -10, -1)))
def test_elementaries(func, tol):
    for x0 in np.linspace(0, 2, 11):
        sp_func = func(x).func  # convert sp.sqrt() to sp.Pow()
        res = cf.calc_elem_func(sp_func, x0=x0, eps=tol)
        exact = func(x).subs({x: x0})
        error = abs(float(res-exact))

        if error > tol:
            print()
            print(f'Test for elementary {func}({x0:.1f}) with tolerance {tol:e}')
            print(f'exact: {float(exact):.6f}, computed: {float(res):.6f}, error = {error:e}')
        assert error <= tol


@pytest.mark.parametrize('function',
                         [
                             (x + 0.4) ** (1/2) + sp.sin(sp.cos(3 * x + 1)),
                             (1 + sp.atan(16.7 * x + 0.1)) ** 2 / sp.cos(7 * x + 0.3),
                             sp.exp(1 + x) * sp.cos(sp.sqrt(1 + x)),
                             sp.sinh(2 * x + 0.45) ** (1/2) / sp.atan(6 * x + 1),
                             sp.cosh(1 + (1 + x) ** (1/2)) * sp.cos(1 + x - x**2),
                         ]
                         )
@pytest.mark.parametrize('tol', 10. ** np.array([-1, -3, -5]))
def test_composites(function, tol):
    for x0 in np.linspace(0, 1, 11):
        res = cf.calc(function, x0=x0, eps=tol)
        exact = function.subs({x: x0})
        error = abs(float(res-exact))

        if error > tol:
            print()
            print(f'Test for {function}({x0:.1f}) with tolerance {tol:e}')
            print(f'exact: {float(exact):.6f}, computed: {float(res):.6f}, error = {error:e}')
        assert error <= tol
