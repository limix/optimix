"""Wrapper for Brent search."""
from numpy import asarray

from brent_search import minimize as brent_minimize
from tqdm import tqdm


class ProxyFunction(object):
    def __init__(self, function, progress, negative):
        self._function = function
        self._signal = -1 if negative else +1
        self._progress = tqdm(desc='Optimix', disable=not progress)
        self._iteration = 0

    def value(self):
        return self._signal * self._function.value()

    def __call__(self, x):
        self._progress.update(1)

        self._iteration += 1

        x = asarray(x).ravel()
        self._function.variables().from_flat(x)

        return self.value()

    def set_solution(self, x):
        self._function.variables().from_flat(asarray(x).ravel())
        self._progress.close()


def _minimize(proxy_function):
    x = brent_minimize(proxy_function)
    proxy_function.set_solution(x)


def minimize(function, progress=False):
    t = function.variables().select(fixed=False)
    if len(t) == 0:
        return function.variables().flatten()[0]
    return _minimize(ProxyFunction(function, progress, False))


def maximize(function, progress=False):
    t = function.variables().select(fixed=False)
    if len(t) == 0:
        return function.variables().flatten()[0]
    return _minimize(ProxyFunction(function, progress, True))
