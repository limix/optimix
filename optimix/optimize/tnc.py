import numpy as np

from scipy.optimize import fmin_tnc

from ..negative import negative_function
from ..function import FunctionDataFeed


def _do_flatten(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return np.concatenate([np.asarray(xi).ravel() for xi in x])
    return np.concatenate(x)


def minimize(function, purpose='learn'):

    f = FunctionDataFeed(function, function._data[purpose])

    # def data(self, *args, **kwargs):
    #     fs = [f.data(*args, **kwargs) for f in self._functions]
    #     return FunctionReduceDataFeed(self, fs)

    def func(x):
        x = np.asarray(x).ravel()
        f.variables().from_flat(x)
        return f.value()

    def grad(x):
        x = np.asarray(x).ravel()
        f.variables().from_flat(x)
        return _do_flatten(f.gradient())

    x0 = f.variables().flatten()
    r = fmin_tnc(func, x0, fprime=grad, disp=0)
    return r[0]


def maximize(function):
    function = negative_function(function)
    return minimize(function)
