import pytest
import numpy as np
import sympy as sp
from sympy.abc import x, y
from sympy.plotting.plot import List2DSeries
import matplotlib.pyplot as plt

from S1T3_newton_method.py.newton_method import solve_scalar, solve_plane


@pytest.mark.parametrize('f,x0',
                         [
                             (x - sp.sin(x) - 0.25,   3.0),
                             (2**x * (x-1)**2 - 2,   -2.5),
                             (sp.tan(.5*x+.2) - x**2, 2.0),
                         ]
                         )
def test_solve_scalar(f, x0):
    interval = -10, 10

    plt.figure()
    plt.suptitle(f)

    # plot reference line
    xs_ref = np.linspace(*interval, 1001)
    ys_ref = [f.subs(x, xx) for xx in xs_ref]
    plt.subplot(1, 2, 1)
    plt.plot(xs_ref, ys_ref, 'c-')
    plt.plot(interval, [0, 0], 'k-')

    # get iterations
    tol = 1e-9
    xs, ys = solve_scalar(f, x0, tol)
    assert abs(f.subs(x, xs[-1])) < tol

    # plot iterations
    plt.subplot(1, 2, 1)
    for i in range(len(xs)-1):
        plt.plot([xs[i],   xs[i+1]], [ys[i],   0], 'r:')
        plt.plot([xs[i+1], xs[i+1]], [0, ys[i+1]], 'g:')
    plt.plot(xs, ys, 'b*')

    plt.xlabel('x')
    plt.ylabel('y')

    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(-np.log10(np.abs(ys)), 'b.:')
    plt.xlabel('N step')
    plt.ylabel('Accuracy')
    plt.show()


@pytest.mark.parametrize('f,x0,y0',
                         [
                             (sp.Matrix([sp.sin(x + 1) - y - 1.2, 2 * x + sp.cos(y) - 2]), 0, 0),
                             (sp.Matrix([sp.cos(y + 0.5) - x - 2, sp.sin(x) - 2*y - 1]), 0, 0),
                             (sp.Matrix([sp.cos(x) + y - 1.2, 2*x - sp.sin(y - 0.5) - 2]), 0, 0),
                         ])
def test_solve_plane(f, x0, y0):
    # get iterations
    tol = 1e-9
    xs, ys, zs = solve_plane(f, x0, y0, tol)
    f_eval = f.subs({x: xs[-1], y: ys[-1]})
    assert np.linalg.norm([float(f_eval[0]),
                           float(f_eval[1])]
                          ) < tol

    # plot f() lines
    p = sp.plot(show=False, backend=sp.plotting.plot_backends['matplotlib'])
    p.extend(sp.plot_implicit(f[0], depth=1, line_color='k', show=False))
    p.extend(sp.plot_implicit(f[1], depth=1, line_color='k', show=False))

    # plot iterations
    l = List2DSeries(xs, ys)
    l.line_color = 'm'
    p.append(l)
    p.title = f'{f[0]}\n{f[1]}'
    p.xlabel = 'x'
    p.ylabel = 'y'
    p.show()

    # plot accuracy
    plt.figure()
    plt.plot(-np.log10(np.abs(zs)), 'b.:')
    plt.suptitle(p.title)
    plt.xlabel('N step')
    plt.ylabel('Accuracy')
    plt.show()
