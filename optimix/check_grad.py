from __future__ import division

from numpy import asarray, finfo, sqrt

_step = sqrt(finfo(float).eps)

def approx_fprime(f, step=_step):
    f0 = f.value()
    grad = dict()
    for name in f.variables().names():
        f.set(name, f.get(name) + step)
        grad[name] = asarray((f.value() - f0) / step).ravel()
        f.set(name, f.get(name) - step)
    return grad

def check_grad(func, step=_step):
    g = func.gradient()
    g = {n:asarray(gi).ravel() for n, gi in iter(g.items())}
    fg = approx_fprime(func, step)

    e = 0.
    for name in set(g.keys()).union(fg.keys()):
        e += sum((fg[name] - g[name])**2)

    return e
