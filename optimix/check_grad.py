from __future__ import division

from numpy import asarray
from numpy import zeros
from numpy import zeros_like
from numpy import finfo
from numpy import sqrt

_step = sqrt(finfo(float).eps)


def approx_fprime(xk, f, step=_step):
    f0 = f(xk)
    grad = [zeros_like(asarray(f0).ravel()) for i in range(len(xk))]
    d = zeros(len(grad))
    for k in range(len(xk)):
        d[k] = step
        grad[k][:] = asarray((f(xk + d) - f0) / step).ravel()
        d[k] = 0
    return grad


def check_grad(func, grad, x0, step=_step):
    g = grad(x0)
    g = [asarray(gi).ravel() for gi in g]
    fg = approx_fprime(x0, func, step)

    e = 0.
    for i in range(len(x0)):
        e += sum((fg[i] - g[i])**2)
    e /= len(x0)

    return e
