from __future__ import division


from numpy import asarray, atleast_1d, finfo, sqrt, squeeze, stack
from numpy.linalg import norm

_step = sqrt(finfo(float).eps)


def approx_fprime(f, step=_step):
    from collections import defaultdict

    f0 = f.value()
    grad = defaultdict(list)
    for name in f.variables().names():
        value = f.variables().get(name).value
        ndim = value.ndim
        value = atleast_1d(value).ravel()
        for i in range(len(value)):
            value[i] += step
            grad[name].append(asarray((f.value() - f0) / step))
            value[i] -= step
        grad[name] = stack(grad[name], axis=-1)
        if ndim == 0:
            grad[name] = squeeze(grad[name], axis=-1)
    return grad


def check_grad(func, step=_step):
    g = func.gradient()
    g = {n: asarray(gi) for n, gi in iter(g.items())}
    fg = approx_fprime(func, step)

    names = set(g.keys()).union(fg.keys())
    return sum(norm(fg[name] - g[name]) for name in names)
