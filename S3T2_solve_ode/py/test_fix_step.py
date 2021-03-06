import pytest
import numpy as np
import matplotlib.pyplot as plt

from utils.ode_collection import Harmonic
from utils.utils import get_log_error
from S3T2_solve_ode.py.solve_ode import fix_step_integration
from S3T2_solve_ode.py.one_step_methods import (
    ExplicitEulerMethod,
    RungeKuttaMethod,
)
import S3T2_solve_ode.py.coeffs_collection as collection
from S3T2_solve_ode.py.multistep_methods import adams, adams_coeffs


def test_one_step():
    """
    test Euler and RK methods
    """
    y0 = np.array([0., 1.])
    t0 = 0
    t1 = np.pi/2
    dt = 0.1

    f = Harmonic(y0, 1, 1)
    ts = np.arange(t0, t1+dt, dt)

    exact = f[ts].T
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(ts, [e[0] for e in exact], 'k', label='Exact')

    colors = 'rgbcmyk'
    for i, method in enumerate(
            [
                ExplicitEulerMethod(),
                RungeKuttaMethod(collection.rk4_coeffs),
                RungeKuttaMethod(collection.dopri_coeffs),
            ]
    ):
        f.clear_call_counter()
        _, y = fix_step_integration(method, f, y0, ts)
        print(f'len(Y): {len(y)}')
        print(f'Function calls: {f.get_call_counter()}')

        plt.subplot(1, 2, 1), plt.plot(ts, [_y[0] for _y in y], f'{colors[i]}.--', label=method.name)
        plt.subplot(1, 2, 2), plt.plot(ts, get_log_error(exact, y), f'{colors[i]}.--', label=method.name)

    plt.subplot(1, 2, 1), plt.xlabel('t'), plt.ylabel('y'), plt.legend()
    plt.subplot(1, 2, 2), plt.xlabel('t'), plt.ylabel('accuracy'), plt.legend()
    plt.suptitle('test_one_step')
    plt.show()


def test_multi_step():
    """
    test Adams method
    Q: compare the right plot for both cases and explain the difference
    A: Euler has p = 2, Rk4 has order = 5 (the best), because adams on interval gives acc = k+1, on step = k + 2
    """
    y0 = np.array([0., 1.])
    t0 = 0
    t1 = 1.
    dt = 0.1

    f = Harmonic(y0, 1, 1)
    ts = np.arange(t0, t1+dt, dt)
    exact = f[ts].T

    for one_step_method in [
        RungeKuttaMethod(collection.rk4_coeffs),
        ExplicitEulerMethod(),
    ]:
        plt.figure()
        plt.subplot(1, 2, 1), plt.plot(ts, [e[0] for e in exact], 'k', label='Exact')
        for p, c in adams_coeffs.items():
            f.clear_call_counter()
            t_adams, y_adams = adams(f, y0, ts, c,
                                     one_step_method=one_step_method)
            print(f'Function calls: {f.get_call_counter()}')

            err = get_log_error(exact, y_adams)

            label = f"Adams's order {p}"
            plt.subplot(1, 2, 1), plt.plot(t_adams, [y[0] for y in y_adams], '.--', label=label)
            plt.subplot(1, 2, 2), plt.plot(t_adams, err, '.--', label=label)

        plt.subplot(1, 2, 1), plt.xlabel('t'), plt.ylabel('y'), plt.legend()
        plt.subplot(1, 2, 2), plt.xlabel('t'), plt.ylabel('accuracy'), plt.legend()
        plt.suptitle(f'test_multi_step\none step method: {one_step_method.name}')
    plt.show()
