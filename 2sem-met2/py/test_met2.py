import pytest
import numpy as np
import matplotlib.pyplot as plt
import met2

default_student = 0


@pytest.mark.parametrize("student", [default_student])
def test_met2(student):
    print(f'running met1 test for student #{student}')
    N = 5
    M = 10
    A = -1  # change to what you prefer
    B = 3   # change to what you prefer

    X_eq = np.linspace(A, B, N+1);
    X_cheb = 1/2 * ((B-A) * np.cos(np.pi * (2*np.arange(0, N+1) + 1) / (2*(N + 1))) + (B+A))
    X_dense = sorted([*np.linspace(A, B, M), *X_eq, *X_cheb])
    Y_dense = met2.func(X_dense)

    plt.figure(1)
    plt.plot(X_dense, Y_dense, 'k-')
    styles = ['b.:', 'm.:']

    Xs = [X_eq, X_cheb]
    for (i, X) in enumerate(Xs):
        Y0 = met2.func(X)  # here's your function (change met2.func())

        P = met2.interpol(X, Y0)    # here's your interpolation (change met2.interpol())
        assert len(P) == N+1
        Y2 = np.polyval(P, X_dense)

        plt.figure(1)
        plt.plot(X_dense, Y2, styles[i])
        plt.figure(2)
        plt.plot(X_dense, np.log10(np.abs(Y_dense - Y2)), styles[i])

    plt.figure(1)
    plt.title('Y(X)')
    plt.legend(['actual', 'interp-eq', 'interp-cheb'])

    plt.figure(2)
    plt.title('log error')
    plt.legend(['interp-eq', 'interp-cheb'])
    plt.show()

