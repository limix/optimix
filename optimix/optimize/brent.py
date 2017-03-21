"""Wrapper for Brent search."""
from numpy import asarray
from tqdm import tqdm

from brent_search import minimize as brent_minimize


class ProxyFunction(object):
    def __init__(self, function, progress, negative):
        self._function = function
        self._signal = -1 if negative else +1
        self._progress = tqdm(desc='Optimix', disable=not progress)
        self._iteration = 0

    def names(self):
        return sorted(self._function.variables().select(fixed=False).names())

    def value(self):
        return self._signal * self._function.value()

    def unflatten(self, x):
        variables = self._function.variables().select(fixed=False)
        d = dict()
        offset = 0
        for name in self.names():
            size = variables.get(name).size
            d[name] = x[offset:offset + size]
            offset += size
        return d

    def __call__(self, x):
        self._progress.update(1)

        self._iteration += 1

        x = asarray(x).ravel()
        self._function.variables().set(self.unflatten(x))

        return self.value()

    def set_solution(self, x):
        self._function.variables().set(self.unflatten(x))
        self._progress.close()


def _minimize(proxy_function):
    x = asarray(brent_minimize(proxy_function))
    proxy_function.set_solution(x[0:1])


def minimize(function, progress=True):
    return _minimize(ProxyFunction(function, progress, False))


def maximize(function, progress=False):
    return _minimize(ProxyFunction(function, progress, True))
