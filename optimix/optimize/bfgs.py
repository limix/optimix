from __future__ import division

from numpy import asarray, concatenate
from scipy.optimize import fmin_l_bfgs_b

from ..exception import OptimixError
from .exception import BadSolutionError


def do_flatten(x):
    if isinstance(x, (list, tuple)):
        return concatenate([asarray(xi).ravel() for xi in x])
    return concatenate(x)


class ProxyFunction(object):
    def __init__(self, function, progress, negative):
        self._function = function
        self._signal = -1 if negative else +1
        self.progress = progress
        self._solutions = []

    @property
    def solutions(self):
        return self._solutions

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
        self._solutions.append(x.copy())
        self._function.variables().set(self.unflatten(x))
        v = self.value()
        g = self.flatten(self.gradient())
        return v, g

    def set_solution(self, x):
        self._function.variables().set(self.unflatten(x))

    def get_solution(self):
        v = self._function.variables().select(fixed=False)
        return concatenate([v.get(n).asarray().ravel() for n in self.names()])


def _try_minimize(proxy_function, n):
    disp = 1 if proxy_function.progress else 0

    if n == 0:
        raise OptimixError("Too many bad solutions")

    try:
        x0 = proxy_function.get_solution()
        return fmin_l_bfgs_b(proxy_function, x0, disp=disp)

    except BadSolutionError:

        xs = proxy_function.solutions
        if len(xs) < 2:
            raise OptimixError("Bad solution at the first iteration.")

        proxy_function.set_solution(xs[-2] / 5 + xs[-1] / 5)

        if disp == 1:
            print("Optimix: Restarting L-BFGS-B due to bad solution.")
        return _try_minimize(proxy_function, n - 1)


def _minimize(proxy_function):

    r = _try_minimize(proxy_function, 5)

    if r[2]['warnflag'] == 1:
        raise OptimixError("L-BFGS-B: too many function evaluations" +
                           " or too many iterations")
    elif r[2]['warnflag'] == 2:
        raise OptimixError("L-BFGS-B: %s" % r[2]['task'])

    proxy_function.set_solution(r[0])


def minimize(function, progress=True):
    return _minimize(ProxyFunction(function, progress, False))


def maximize(function, progress=True):
    return _minimize(ProxyFunction(function, progress, True))
