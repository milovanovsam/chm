import enum
import numpy as np

from S3T2_solve_ode.py.one_step_methods import OneStepMethod, ExplicitEulerMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, func, y_start, ts):
    """
    performs fix-step integration using one-step method
    ts: array of timestamps
    return: list of t's, list of y's
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(func, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def adaptive_step_integration(method: OneStepMethod, func, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    performs adaptive-step integration using one-step method
    t_span: (t0, t1)
    adapt_type: Runge or Embedded
    tolerances control the error:
        err <= atol
        err <= |y| * rtol
    return: list of t's, list of y's
    """
    y = y_start
    t, t_end = t_span
    p = method.p
    ts = [t]
    ys = [y]
    delta = (1/max(np.abs(t), np.abs(t_end))) ** (p + 1) + np.linalg.norm(func(t, y)) ** (p + 1)
    h_1 = (atol / delta) ** (1 / (p + 1))
    u_1 = y + h_1 * func(t, y)
    delta = (1 / max(np.abs(t), np.abs(t_end))) ** (p + 1) + np.linalg.norm(func(t + h_1, u_1)) ** (p + 1)
    h_2 = (atol / delta) ** (1 / (p + 1))
    h_opt = min(h_1, h_2)

    while t < t_end:
        if t + h_opt > t_end:
            h_opt = t_end - t
        if adapt_type == AdaptType.RUNGE:
            y_1 = method.step(func, t, y, h_opt)
            y_2 = method.step(func, t + h_opt / 2, method.step(func, t, y, h_opt / 2), h_opt / 2)
            err1 = (y_2 - y_1) / (1 - 2 ** (- p))
            y_help = y_1 + err1
            err2 = (y_1 - y_2) / (2 ** p - 1)
            err = max(np.linalg.norm(err1), np.linalg.norm(err2))
        else:
            y_help, err = method.embedded_step(func, t, y, h_opt)
        tol = rtol * np.linalg.norm(y_help) + atol
        # tol = atol
        if np.linalg.norm(err) <= tol:
            y = y_help
            ts.append(t + h_opt)
            t = t + h_opt
            ys.append(y)
        h_opt = h_opt * (tol / np.linalg.norm(err)) ** (1 / p) * 0.8
    return ts, ys
