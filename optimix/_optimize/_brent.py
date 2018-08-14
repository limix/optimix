from __future__ import division

from brent_search import minimize as brent_minimize
from numpy import asarray


def minimize(function, desc, verbose):
    r"""Minimize a scalar function using Brent's method.

    Parameters
    ----------
    function : object
        Objective function. It has to implement the
        :class:`optimix.function.Function` interface.
    verbose : bool
        ``True`` for verbose output; ``False`` otherwise.
    """
    _minimize(ProxyFunction(function, desc, verbose, False))


def maximize(function, desc, verbose):
    r"""Maximize a scalar function using Brent's method.

    Parameters
    ----------
    function : object
        Objective function. It has to implement the
        :class:`optimix.function.Function` interface.
    verbose : bool
        ``True`` for verbose output; ``False`` otherwise.
    """
    _minimize(ProxyFunction(function, desc, verbose, True))


class ProxyFunction(object):
    def __init__(self, function, desc, verbose, negative):
        from tqdm import tqdm

        self._function = function
        self._signal = -1 if negative else +1
        self._progress = tqdm(desc=desc, disable=not verbose)
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
            d[name] = x[offset : offset + size]
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
