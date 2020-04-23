import numpy as np
import numpy.linalg as la


def solve_upper_triang(A, b):
    """
    solves Ax=b for U-triangular A
    """
    A = np.array(A, ndmin=2)
    b = np.reshape(b, (-1, 1))
    n = len(b)

    Ab = np.concatenate((A, b), axis=1)

    for i in range(n)[::-1]:
        Ab[i, :] /= Ab[i, i]
        for j in range(i):
            Ab[j, :] -= Ab[i, :] * Ab[j, i]

    A, b = np.split(Ab, [n], axis=1)
    return b.reshape(-1)


def solve_lower_triang(A, b):
    """
    solves Ax=b for L-triangular A
    """
    raise NotImplementedError
    return x


def lu(A):
    """
    LU decomposition
    """
    raise NotImplementedError
    return l, u


def qr(A):
    """
    QR decomposition
    """
    raise NotImplementedError
    return q, r


def solve_lu(A, b):
    """
    solve Ax=b using LU decomposition
    """
    L, U = lu(A)
    b1 = solve_lower_triang(L, b)
    return solve_upper_triang(U, b1)


def solve_qr(A, b):
    """
    solve Ax=b using QR decomposition
    """
    Q, R = qr(A)
    b = np.reshape(b, (-1, 1))
    return solve_upper_triang(R, Q.T @ b)
