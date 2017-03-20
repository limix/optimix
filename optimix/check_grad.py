from __future__ import division

from collections import defaultdict
from numpy import asarray, finfo, sqrt, stack, atleast_1d
from numpy.linalg import norm

_step = sqrt(finfo(float).eps)

def approx_fprime(f, step=_step):
    f0 = f.value()
    grad = defaultdict(list)
    for name in f.variables().names():
        value = atleast_1d(f.variables().get(name).value).ravel()
        import pdb; pdb.set_trace()
        for i in range(len(value)):
            value[i] += step
            # try:
                # f.variables().get(name).value[i] = v + step
            # except:
                # import pdb; pdb.set_trace()
            grad[name].append(asarray((f.value() - f0) / step))
            # f.variables().get(name).value[i] -= step
            value[i] -= step
        grad[name] = stack(grad[name], axis=0).T
    return grad

def check_grad(func, step=_step):
    g = func.gradient()
    g = {n:asarray(gi) for n, gi in iter(g.items())}
    fg = approx_fprime(func, step)

    names = set(g.keys()).union(fg.keys())
    return sum(norm(fg[name] - g[name]) for name in names)
