import sys

from numpy import asarray

from scipy.optimize import brent

from ..negative import negative_function


def minimize(function, progress=None):

    def func(x):
        if progress:
            progress.update(func.i)
        func.i += 1
        x = asarray(x).ravel()
        function.variables().from_flat(x)
        return function.value()
    func.i = 0

    x = brent(func)
    function.variables().from_flat(asarray(x).ravel())
    if progress:
        progress.finish()


def maximize(function, progress=None):
    function = negative_function(function)
    return minimize(function, progress=progress)
