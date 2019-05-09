r"""
********
Function
********
"""
from __future__ import unicode_literals

import collections

from ._unicode import unicode_airlock
from ._variables import Variables, merge_variables
from ._optimize import maximize, minimize, maximize_scalar, minimize_scalar

FACTR = 1e5
PGTOL = 1e-7


class Function(object):
    r"""Base-class for object representing functions.

    Parameters
    ----------
    kwargs : dict
        Map of variable name to variable value.
    """

    def __init__(self, **kwargs):
        self._variables = Variables(kwargs)
        self._data = dict()
        self._name = kwargs.get("name", "unamed")

    def value(self, *args):
        r"""Evaluate the function at the ``args`` point.

        Parameters
        ----------
        args : tuple
            Point at the evaluation. The length of this :func:`tuple` is
            defined by the user.

        Returns
        -------
        float or array_like
            Function evaluated at ``args``.
        """
        raise NotImplementedError

    def gradient(self, *args):
        r"""Evaluate the gradient at the ``args`` point.

        Parameters
        ----------
        args : tuple
            Point at the gradient evaluation. The length of this :func:`tuple`
            is defined by the user.

        Returns
        -------
        dict
            Map between variables to their gradient values.
        """
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def feed(self, purpose="learn"):
        r"""Return a function with attached data."""
        purpose = unicode_airlock(purpose)
        f = FunctionDataFeed(self, self._data[purpose], self._name)
        return f

    def fix(self, var_name):
        r"""Set a variable fixed.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        self._variables[var_name].fix()

    def unfix(self, var_name):
        r"""Set a variable unfixed.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        self._variables[var_name].unfix()

    def isfixed(self, var_name):
        r"""Return whether a variable it is fixed or not.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        return self._variables[var_name].isfixed

    def variables(self):
        r"""Function variables."""
        return self._variables

    def set_nodata(self, purpose="learn"):
        r"""Disable data feeding.

        Parameters
        ----------
        purpose : str
            Name of the data source.
        """
        purpose = unicode_airlock(purpose)
        self._data[purpose] = tuple()

    def set_data(self, data, purpose="learn"):
        r"""Set a named data source.

        Parameters
        ----------
        purpose : str
            Name of the data source.
        """
        purpose = unicode_airlock(purpose)
        if not isinstance(data, collections.abc.Sequence):
            data = (data,)
        self._data[purpose] = data

    def unset_data(self, purpose="learn"):
        r"""Unset a named data source.

        Parameters
        ----------
        purpose : str
            Name of the data source.
        """
        purpose = unicode_airlock(purpose)
        del self._data[purpose]


class FunctionReduce(object):
    def __init__(self, functions, name="unamed"):
        self.functions = functions
        self.__name = name

    def operand(self, i):
        r"""Get the i-th function.

        Parameters
        ----------
        i : int
            Function index.

        Returns
        -------
        function
            The referred function.
        """
        return self.functions[i]

    def feed(self, purpose="learn"):
        r"""Return a function with attached data."""
        purpose = unicode_airlock(purpose)
        fs = [f.feed(purpose) for f in self.functions]
        f = FunctionReduceDataFeed(self, fs, self.__name)
        return f

    def variables(self):
        vars_list = [l.variables() for l in self.functions]
        vd = dict()
        for (i, vs) in enumerate(vars_list):
            vd["%s[%d]" % (self.__name, i)] = vs
        return merge_variables(vd)


class FunctionDataFeed(object):
    def __init__(self, target, data, name):
        self._target = target
        self.raw = data
        self._name = name

    @property
    def name(self):
        return self._name

    def value(self):
        return self._target.value(*self.raw)

    def gradient(self):
        return self._target.gradient(*self.raw)

    def variables(self):
        return self._target.variables()

    def maximize(self, verbose=True, factr=FACTR, pgtol=PGTOL):
        return maximize(self, verbose=verbose, factr=factr, pgtol=pgtol)

    def minimize(self, verbose=True, factr=FACTR, pgtol=PGTOL):
        return minimize(self, verbose=verbose, factr=factr, pgtol=pgtol)

    def maximize_scalar(self, desc="", verbose=True):
        return maximize_scalar(self, desc=desc, verbose=verbose)

    def minimize_scalar(self, desc="", verbose=True):
        return minimize_scalar(self, desc=desc, verbose=verbose)


class FunctionReduceDataFeed(object):
    def __init__(self, target, functions, name="unamed"):
        self._target = target
        self.functions = functions
        self.__name = name

    @property
    def name(self):
        return self.__name

    def value(self):
        value = dict()
        for (i, f) in enumerate(self.functions):
            value["%s[%d]" % (self.__name, i)] = f.value()
        vr = self._target.value_reduce
        return vr(value)

    def gradient(self):
        value = dict()
        for (i, f) in enumerate(self.functions):
            value["%s[%d]" % (self.__name, i)] = f.value()

        grad = collections.defaultdict(dict)
        for (i, f) in enumerate(self.functions):
            for gn, gv in iter(f.gradient().items()):
                grad["%s[%d]" % (self.__name, i)][gn] = gv
        gr = self._target.gradient_reduce
        return gr(value, grad)

    def variables(self):
        return self._target.variables()

    def maximize(self, verbose=True, factr=FACTR, pgtol=PGTOL):
        return maximize(self, verbose=verbose, factr=factr, pgtol=pgtol)

    def minimize(self, verbose=True, factr=FACTR, pgtol=PGTOL):
        return minimize(self, verbose=verbose, factr=factr, pgtol=pgtol)

    def maximize_scalar(self, desc="", verbose=True):
        return maximize_scalar(self, desc=desc, verbose=verbose)

    def minimize_scalar(self, desc="", verbose=True):
        return minimize_scalar(self, desc=desc, verbose=verbose)
