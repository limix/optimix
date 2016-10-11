import sys

from numpy import asarray

from scipy.optimize import brent

from ..util import as_data_function


class ProxyFunction(object):

    def __init__(self, function, purpose, progress, negative):
        self._function = as_data_function(function, purpose)
        self._signal = -1 if negative else +1
        self._progress = progress
        self._iteration = 0

    def value(self):
        return self._signal * self._function.value()

    def __call__(self, x):
        if self._progress:
            self._progress.update(self._iteration)

        self._iteration += 1

        x = asarray(x).ravel()
        self._function.variables().from_flat(x)

        return self.value()

    def set_solution(self, x):
        self._function.variables().from_flat(asarray(x).ravel())
        if self._progress:
            self._progress.finish()


def _minimize(proxy_function):
    x = brent(proxy_function)
    proxy_function.set_solution(x)


def minimize(function, purpose='learn', progress=None):
    return _minimize(ProxyFunction(function, purpose, progress, False))


def maximize(function, purpose='learn', progress=None):
    return _minimize(ProxyFunction(function, purpose, progress, True))
