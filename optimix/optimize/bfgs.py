from numpy import asarray, concatenate
from scipy.optimize import fmin_l_bfgs_b

from ..exception import OptimixError


def do_flatten(x):
    if isinstance(x, (list, tuple)):
        return concatenate([asarray(xi).ravel() for xi in x])
    return concatenate(x)


class ProxyFunction(object):
    def __init__(self, function, progress, negative):
        self._function = function
        self._signal = -1 if negative else +1
        self.progress = progress

    def names(self):
        return sorted(self._function.variables().select(fixed=False).names())

    def value(self):
        return self._signal * self._function.value()

    def gradient(self):
        g = self._function.gradient()
        return {name: self._signal * g[name] for name in self.names()}

    def unflatten(self, x):
        variables = self._function.variables().select(fixed=False)
        d = dict()
        offset = 0
        for name in self.names():
            size = variables.get(name).size
            d[name] = x[offset:offset + size]
            offset += size
        return d

    def flatten(self, d):
        names = self.names()
        return do_flatten([d[name] for name in names])

    def __call__(self, x):
        x = asarray(x).ravel()
        self._function.variables().set(self.unflatten(x))
        v = self.value()
        g = self.flatten(self.gradient())
        return v, g

    def set_solution(self, x):
        self._function.variables().set(self.unflatten(x))

    def get_solution(self):
        v = self._function.variables().select(fixed=False)
        return concatenate([v.get(n).asarray().ravel() for n in self.names()])


def _minimize(proxy_function):
    x0 = proxy_function.get_solution()
    disp = 1 if proxy_function.progress else 0

    r = fmin_l_bfgs_b(proxy_function, x0, disp=disp)

    if r[2]['warnflag'] == 1:
        raise OptimixError("BFGS: too many function evaluations" +
                           " or too many iterations")
    elif r[2]['warnflag'] == 2:
        raise OptimixError("BFGS: %s" % r[2]['task'])

    proxy_function.set_solution(r[0])


def minimize(function, progress=True):
    return _minimize(ProxyFunction(function, progress, False))


def maximize(function, progress=True):
    return _minimize(ProxyFunction(function, progress, True))
