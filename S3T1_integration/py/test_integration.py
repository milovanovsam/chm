import numpy as np
import matplotlib.pyplot as plt

from utils.integrate_collection import Monome, Harmonic

from utils.utils import get_log_error
from S3T1_integration.py.integration import (quad,
                                             quad_gauss,
                                             composite_quad,
                                             integrate,
                                             aitken)


def test_quad_degree():
    """
    check quadrature degree
    """
    x0, x1 = 0, 1

    max_degree = 7

    for deg in range(1, max_degree):
        p = Monome(deg)
        y0 = p[x0, x1]

        max_node_count = range(1, max_degree + 1)

        Y = [[quad(p, x0, x1, np.linspace(x0, x1, node_count)) for node_count in max_node_count]]
        # Y = [quad(p, x0, x1, x0 + (x1-x0) * np.random.random(node_count)) for node_count in max_node_count]
        accuracy = get_log_error(Y, y0 * np.ones_like(Y))
        accuracy[np.isinf(accuracy)] = 17

        # check accuracy is good enough
        for node_count, acc in zip(max_node_count, accuracy):
            if node_count >= deg + 1:
                assert acc > 6
        print(accuracy)
        plt.plot(max_node_count, accuracy, '.:', label=f'x^{deg}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node_count')
    plt.suptitle(f'test quad')
    plt.show()


def test_quad_gauss_degree():
    """
    check gaussian quadrature degree
    """
    x0, x1 = 0, 1

    max_degree = 8

    for deg in range(2, max_degree):
        p = Monome(deg)
        y0 = p[x0, x1]

        max_node_count = range(2, 6)
        Y = [quad_gauss(p, x0, x1, node_count) for node_count in max_node_count]
        accuracy = get_log_error(Y, y0 * np.ones_like(Y))
        accuracy[np.isinf(accuracy)] = 17

        # check accuracy is good enough
        for node_count, acc in zip(max_node_count, accuracy):
            if 2 * node_count >= deg + 1:
                assert acc > 6

        plt.plot(max_node_count, accuracy, '.:', label=f'x^{deg}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node count')
    plt.suptitle(f'test quad gauss')
    plt.show()


def test_composite_quad():
    plt.figure()

    x0, x1 = 0, 1
    L = 2
    n_intervals = [L ** q for q in range(0, 8)]
    n_nodes = 3

    for i, degree in enumerate((5, 6)):
        p = Monome(degree)
        Y = [composite_quad(p, x0, x1, n_intervals=n, n_nodes=n_nodes) for n in n_intervals]
        accuracy = get_log_error(Y, p[x0, x1] * np.ones_like(Y))
        x = np.log10(n_intervals)

        # check convergence
        ind = np.isfinite(x) & np.isfinite(accuracy)
        k, b = np.polyfit(x[ind], accuracy[ind], 1)
        aitken_degree = aitken(*Y[0:6:2], L ** 2)

        plt.subplot(1, 2, i + 1)
        plt.plot(x, k * x + b, 'b:', label=f'{k:.2f}*x+{b:.2f}')
        plt.plot(x, aitken_degree * x + b, 'm:', label=f'aitken estimation')
        plt.plot(x, accuracy, 'kh', label=f'accuracy for x^{degree}')
        plt.legend()

        assert np.abs(aitken_degree - k) < 0.5, \
            f'Aitken estimation {aitken_degree:.2f} is too far from actual {k:.2f}'

    plt.show()


def test_integrate():
    p = Harmonic(1, 0)
    x0, x1 = 0, np.pi

    for tol in 10. ** np.arange(-9, -2):
        s, err = integrate(p, x0, x1, tol=tol)

        print(f'Check for tol {tol:.2e}: res = {s - err:.6f} .. {s:.6f} .. {s + err:.6f}')

        assert err >= 0, 'estimated error should be >= 0'
        assert np.abs(p[x0, x1] - s) <= err, 'actual error should be <= estimated error'
        assert np.abs(p[x0, x1] - s) <= tol, 'actual error should be <= tolerance'
