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

    def get(self, name):
        return self._variables.get(name).value

    def set(self, name, value):
        self._variables.get(name).value = value

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

    def gradient(self, *args, **kwargs):
        grad = {}
        for i, l in enumerate(self.functions):
            grad['%s[%d]' % (self.__name, i)] = l.gradient(*args, **kwargs)
        return grad

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

    def get(self, name):
        return self._target.get(name)

    def set(self, name, value):
        self._target.set(name, value)


class FunctionReduceDataFeed(object):
    def __init__(self, target, functions, name='unamed'):
        self._target = target
        self.functions = functions
        self.__name = name

    @property
    def name(self):
        return self.__name

    def value(self):
        return self._target.value_reduce([f.value() for f in self.functions])

    def gradient(self):
        grad = {}
        for i, l in enumerate(self.functions):
            g = l.gradient()
            for j, v in iter(g.items()):
                grad['%s[%d].%s' % (self.__name, i, j)] = v
        return grad

    def variables(self):
        return self._target.variables()

    def maximize(self, progress=True):
        from .optimize import maximize as _maximize
        return _maximize(self, progress=progress)

    def minimize(self, progress=True):
        from .optimize import minimize as _minimize
        return _minimize(self, progress=progress)

    def get(self, name):
        return self._target.get(name)

    def set(self, name, value):
        self._target.set(name, value)


class Composite(object):
    def __init__(self, **kwargs):
        super(Composite, self).__init__()
        self.functions = kwargs
        self._data = dict()
        if 'prefix' in kwargs:
            self.__prefix = kwargs['prefix']
        else:
            self.__prefix = 'unamed'

    def feed(self, purpose='learn'):
        purpose = unicode_airlock(purpose)
        return CompositeDataFeed(self, purpose)

    def gradient(self, *args, **kwargs):
        fnames = sorted(self.functions.keys())
        grad = {}
        for fname in fnames:
            fg = getattr(self, 'gradient_' + fname)(*args, **kwargs)
            # grad += fg
            grad[fname] = fg
        return grad

    # def gradient(self, *args, **kwargs):
    #     names = sorted(self._variables.select(fixed=False).names())
    #     grad = {}
    #     for name in names:
    #         grad[name] = getattr(self, 'derivative_' + name)(*args, **kwargs)
    #     return grad

    def set_nodata(self, purpose='learn'):
        purpose = unicode_airlock(purpose)
        self._data[purpose] = tuple()

    def set_data(self, data, purpose='learn'):
        purpose = unicode_airlock(purpose)
        if not isinstance(data, collections.Sequence):
            data = (data, )
        self._data[purpose] = data

    def unset_data(self, purpose='learn'):
        purpose = unicode_airlock(purpose)
        del self._data[purpose]

    def variables(self):
        fnames = sorted(self.functions.keys())
        vars_list = [self.functions[fn].variables() for fn in fnames]
        vd = dict()
        for (i, vs) in enumerate(vars_list):
            vd['%s[%d]' % (self.__prefix, i)] = vs
        return merge_variables(vd)


class CompositeDataFeed(object):
    def __init__(self, target, purpose):
        purpose = unicode_airlock(purpose)
        self._target = target
        self._purpose = purpose

    def value(self):
        fnames = sorted(self._target.functions.keys())

        fvals = dict()
        for fname in fnames:
            f = self._target.functions[fname].feed(self._purpose)
            fvals[fname] = f.value()

        return self._target.value(**fvals)

    def gradient(self):
        fnames = sorted(self._target.functions.keys())

        g_fvals = dict()
        for fname in fnames:
            f = self._target.functions[fname].feed(self._purpose)
            g_fvals[fname] = f.value()
            g_fvals['g' + fname] = f.gradient()

        return self._target.gradient(**g_fvals)

    def variables(self):
        return self._target.variables()

    def maximize(self, progress=True):
        from .optimize import maximize as _maximize
        return _maximize(self, progress=progress)

    def minimize(self, progress=True):
        from .optimize import minimize as _minimize
        return _minimize(self, progress=progress)
