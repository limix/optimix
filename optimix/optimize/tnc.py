import numpy as np

from scipy.optimize import fmin_tnc

from ..negative import negative_function


def _do_flatten(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return np.concatenate([np.asarray(xi).ravel() for xi in x])
    return np.concatenate(x)


def minimize(function):

    def func(x):
        x = np.asarray(x).ravel()
        function.variables().from_flat(x)
        return function.value()

    def grad(x):
        x = np.asarray(x).ravel()
        function.variables().from_flat(x)
        return _do_flatten(function.gradient())

    x0 = function.variables().flatten()
    r = fmin_tnc(func, x0, fprime=grad, disp=0)
    return r[0]


def maximize(function):
    function = negative_function(function)
    return minimize(function)
