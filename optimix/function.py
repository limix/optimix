r"""
********
Function
********

Introduction
^^^^^^^^^^^^

- :class:`optimix.function.Function`
- :class:`optimix.function.Composite`

Public interface
^^^^^^^^^^^^^^^^
"""
from __future__ import unicode_literals

import collections

from ._unicode import unicode_airlock
from .variables import Variables, merge_variables


class Function(object):
    r"""Base-class for object representing functions.

    Args:
        kwargs (dict): map of variable name to variable value.
    """

    def __init__(self, **kwargs):
        self._variables = Variables(kwargs)
        self._data = dict()
        self._name = kwargs.get('name', 'unamed')

    @property
    def name(self):
        return self._name

    def feed(self, purpose='learn'):
        r"""Return a function with attached data."""
        purpose = unicode_airlock(purpose)
        return FunctionDataFeed(self, self._data[purpose], self._name)

    def fix(self, var_name):
        """Set a variable fixed.

        Args:
            var_name (str): variable name.
        """
        self._variables[var_name].fix()

    def unfix(self, var_name):
        """Set a variable unfixed.

        Args:
            var_name (str): variable name.
        """
        self._variables[var_name].unfix()

    def isfixed(self, var_name):
        """Return whether a variable it is fixed or not.

        Args:
            var_name (str): variable name.
        """
        return self._variables[var_name].isfixed()

    def variables(self):
        r"""Function variables."""
        return self._variables

    def set_nodata(self, purpose='learn'):
        r"""Disable data feeding.

        Args:
            purpose (str): name of the data source.
        """
        purpose = unicode_airlock(purpose)
        self._data[purpose] = tuple()

    def set_data(self, data, purpose='learn'):
        r"""Set a named data source.

        Args:
            purpose (str): name of the data source.
        """
        purpose = unicode_airlock(purpose)
        if not isinstance(data, collections.Sequence):
            data = (data, )
        self._data[purpose] = data

    def unset_data(self, purpose='learn'):
        r"""Unset a named data source.

        Args:
            purpose (str): name of the data source.
        """
        purpose = unicode_airlock(purpose)
        del self._data[purpose]


class FunctionReduce(object):
    def __init__(self, functions, name='unamed'):
        self.functions = functions
        self.__name = name

    def operand(self, i):
        return self.functions[i]

    def feed(self, purpose='learn'):
        purpose = unicode_airlock(purpose)
        fs = [f.feed(purpose) for f in self.functions]
        return FunctionReduceDataFeed(self, fs, self.__name)

    # def gradient(self, *args, **kwargs):
    #     grad = {}
    #     for i, l in enumerate(self.functions):
    #         grad['%s[%d]' % (self.__name, i)] = l.gradient(*args, **kwargs)
    #     return grad

    def variables(self):
        vars_list = [l.variables() for l in self.functions]
        vd = dict()
        for (i, vs) in enumerate(vars_list):
            vd['%s[%d]' % (self.__name, i)] = vs
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

    def maximize(self, progress=True):
        from .optimize import maximize as _maximize
        return _maximize(self, progress=progress)

    def minimize(self, progress=True):
        from .optimize import minimize as _minimize
        return _minimize(self, progress=progress)


class FunctionReduceDataFeed(object):
    def __init__(self, target, functions, name='unamed'):
        self._target = target
        self.functions = functions
        self.__name = name

    @property
    def name(self):
        return self.__name

    def value(self):
        value = dict()
        for (i, f) in enumerate(self.functions):
            value['%s[%d]' % (self.__name, i)] = f.value()
        vr = self._target.value_reduce
        return vr(value)

    def gradient(self):
        value = dict()
        for (i, f) in enumerate(self.functions):
            value['%s[%d]' % (self.__name, i)] = f.value()

        grad = collections.defaultdict(dict)
        for (i, f) in enumerate(self.functions):
            for gn, gv in iter(f.gradient().items()):
                grad['%s[%d]' % (self.__name, i)][gn] = gv
        gr = self._target.gradient_reduce
        return gr(value, grad)

    # def variables(self):
    #     vars_list = [l.variables() for l in self.functions]
    #     vd = dict()
    #     for (i, vs) in enumerate(vars_list):
    #         vd['%s[%d]' % (self.__name, i)] = vs
    #     return merge_variables(vd)

    # def gradient(self):
    #     grad = {}
    #     for i, l in enumerate(self.functions):
    #         g = l.gradient()
    #         for j, v in iter(g.items()):
    #             grad['%s[%d].%s' % (self.__name, i, j)] = v
    #     return grad

    def variables(self):
        return self._target.variables()

    def maximize(self, progress=True):
        from .optimize import maximize as _maximize
        return _maximize(self, progress=progress)

    def minimize(self, progress=True):
        from .optimize import minimize as _minimize
        return _minimize(self, progress=progress)
