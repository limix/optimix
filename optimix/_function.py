r"""
********
Function
********
"""
from __future__ import unicode_literals

from collections import defaultdict
from collections.abc import Sequence

from ._optimize import maximize, maximize_scalar, minimize, minimize_scalar
from ._unicode import unicode_airlock
from ._variables import Variables, merge_variables, merge_variables2
from ._func_opt import FuncOpt

FACTR = 1e5
PGTOL = 1e-7


class Func(FuncOpt):
    def __init__(self, name, composite=[], **kwargs):
        super(Func, self).__init__()
        FuncOpt.__init__(self)
        self._name = name
        named_vars = {"": Variables(kwargs)}
        inherited_vars = [f._variables for f in composite]
        for (i, v) in enumerate(inherited_vars):
            named_vars[f"{self._name}[{i}]."] = v
        self._variables = merge_variables2(named_vars)

        # vars = [f._variables for f in kwargs.pop("composite", [])]
        # vars += [Variables(kwargs)]
        # self._variables = merge_variables(vars)
        # # merge_variables
        # # vars_list = [l.variables() for l in self.functions]
        # # vd = dict()
        # # for (i, vs) in enumerate(vars_list):
        # #     vd["%s[%d]" % (self.__name, i)] = vs
        # # return merge_variables(vd)

    def value(self):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def _fix(self, var_name):
        self._variables[var_name].fix()

    def _unfix(self, var_name):
        self._variables[var_name].unfix()

    def _isfixed(self, var_name):
        return self._variables[var_name].isfixed

    def _unfixed_names(self):
        return sorted(self._variables.select(fixed=False).names())

    def _check_grad(self, step=1.49e-08):
        from numpy import asarray
        from numpy.linalg import norm

        g = self.gradient()
        g = {n: asarray(gi) for n, gi in g.items()}
        fg = self._approx_fprime(step)

        names = set(g.keys()).union(fg.keys())
        return sum(norm(fg[name] - g[name]) for name in names)

    def _approx_fprime(self, step=1.49e-08):
        from collections import defaultdict
        from numpy import atleast_1d, asarray, squeeze, stack

        f0 = self.value()
        grad = defaultdict(list)
        for name in self._variables.names():
            value = self._variables.get(name).value
            ndim = value.ndim
            value = atleast_1d(value).ravel()
            for i in range(len(value)):
                value[i] += step
                grad[name].append(asarray((self.value() - f0) / step))
                value[i] -= step
            grad[name] = stack(grad[name], axis=-1)
            if ndim == 0:
                grad[name] = squeeze(grad[name], axis=-1)
        return grad


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

    def value(self, *args, **kwargs):
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

    def gradient(self, *args, **kwargs):
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
        if not isinstance(data, Sequence):
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

    def value(self, **kwargs):
        return self._target.value(*self.raw, **kwargs)

    def gradient(self, **kwargs):
        return self._target.gradient(*self.raw, **kwargs)

    def variables(self):
        return self._target.variables()

    def maximize(self, verbose=True, factr=FACTR, pgtol=PGTOL, **kwargs):
        return maximize(self, verbose=verbose, factr=factr, pgtol=pgtol, **kwargs)

    def minimize(self, verbose=True, factr=FACTR, pgtol=PGTOL, **kwargs):
        return minimize(self, verbose=verbose, factr=factr, pgtol=pgtol, **kwargs)

    def maximize_scalar(self, desc="", verbose=True, **kwargs):
        return maximize_scalar(self, desc=desc, verbose=verbose, **kwargs)

    def minimize_scalar(self, desc="", verbose=True, **kwargs):
        return minimize_scalar(self, desc=desc, verbose=verbose, **kwargs)


class FunctionReduceDataFeed(object):
    def __init__(self, target, functions, name="unamed"):
        self._target = target
        self.functions = functions
        self.__name = name

    @property
    def name(self):
        return self.__name

    def value(self, **kwargs):
        value = dict()
        for (i, f) in enumerate(self.functions):
            value["%s[%d]" % (self.__name, i)] = f.value()
        vr = self._target.value_reduce
        return vr(value, **kwargs)

    def gradient(self, **kwargs):
        value = dict()
        for (i, f) in enumerate(self.functions):
            value["%s[%d]" % (self.__name, i)] = f.value()

        grad = defaultdict(dict)
        for (i, f) in enumerate(self.functions):
            for gn, gv in iter(f.gradient(**kwargs).items()):
                grad["%s[%d]" % (self.__name, i)][gn] = gv
        gr = self._target.gradient_reduce
        return gr(value, grad, **kwargs)

    def variables(self):
        return self._target.variables()

    def maximize(self, verbose=True, factr=FACTR, pgtol=PGTOL, **kwargs):
        return maximize(self, verbose=verbose, factr=factr, pgtol=pgtol, **kwargs)

    def minimize(self, verbose=True, factr=FACTR, pgtol=PGTOL, **kwargs):
        return minimize(self, verbose=verbose, factr=factr, pgtol=pgtol, **kwargs)

    def maximize_scalar(self, desc="", verbose=True, **kwargs):
        return maximize_scalar(self, desc=desc, verbose=verbose, **kwargs)

    def minimize_scalar(self, desc="", verbose=True, **kwargs):
        return minimize_scalar(self, desc=desc, verbose=verbose, **kwargs)
