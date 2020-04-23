import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from S1T2_solve_linear_system.py.iteratives import richardson, jacobi, seidel


def test_iteratives():
    """
    Q: which is the best? why?
    """
    n = 5
    A = np.array([
        [n + 2, 1, 1],
        [1, n + 4, 1],
        [1, 1, n + 6],
    ], dtype='float64')

    b = np.array([n + 4, n + 6, n + 8], dtype='float64')

    xs, ys = richardson(A, b, 1e-6)
    plt.plot(range(len(ys)), -np.log10(np.abs(ys)), 'm.-', label='Richardson')

    xs, ys = jacobi(A, b, 1e-6)
    plt.plot(range(len(ys)), -np.log10(np.abs(ys)), 'b.-', label='Jacobi')

    xs, ys = seidel(A, b, 1e-6)
    plt.plot(range(len(ys)), -np.log10(np.abs(ys)), 'g.-', label='Gauss-Seidel')

    plt.suptitle('Test iterative methods')
    plt.ylabel('Acc')
    plt.xlabel('N iter')
    plt.legend()
    plt.show()
