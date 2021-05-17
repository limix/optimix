import abc

from ._exception import OptimixError
from ._variables import Variables, merge_variables

__all__ = ["Function"]

FACTR = 1e5
PGTOL = 1e-7


class FuncOpt(metaclass=abc.ABCMeta):
    def __init__(self, variables):
        from numpy import zeros, nan

        self.__solutions = []
        self.__sign = +1.0
        self.__verbose = True
        self.__flat_gradient = zeros(1)
        self.__flat_solution = nan
        self._variables = variables

    def _minimize_scalar(
        self, desc="Progress", rtol=1.4902e-08, atol=1.4902e-08, verbose=True
    ):
        """
        Minimize a scalar function using Brent's method.

        Parameters
        ----------
        verbose : bool
            ``True`` for verbose output; ``False`` otherwise.
        """
        from tqdm import tqdm
        from numpy import asarray
        from brent_search import minimize as brent_minimize

        variables = self._variables.select(fixed=False)
        if len(variables) != 1:
            raise ValueError("The number of variables must be equal to one.")

        var = variables[variables.names()[0]]
        progress = tqdm(desc=desc, disable=not verbose)

        def func(x):
            progress.update(1)
            var.value = x
            return self.__sign * self.value()

        r = asarray(
            brent_minimize(func, a=var.bounds[0], b=var.bounds[1], rtol=rtol, atol=atol)
        )
        var.value = r[0]
        progress.close()

    @abc.abstractmethod
    def value(_):
        return 0.0

    @abc.abstractmethod
    def gradient(_):
        from numpy import zeros

        return {"zero": zeros(1)}

    def _maximize_scalar(
        self, desc="Progress", rtol=1.4902e-08, atol=1.4902e-08, verbose=True
    ):
        self.__sign = -1.0
        try:
            self._minimize_scalar(desc, rtol, atol, verbose)
        finally:
            self.__sign = +1.0

    def _minimize(self, verbose=True, factr=FACTR, pgtol=PGTOL):
        from numpy import abs as npabs, max as npmax
        from numpy import empty, atleast_1d

        self.__verbose = verbose
        varnames = self.__varnames()

        grad = self.gradient()
        sign_grad = {name: self.__sign * atleast_1d(grad[name]) for name in varnames}

        self.__flat_gradient = empty(sum(s.size for s in sign_grad.values()))
        self.__flat_solution = empty(sum(self._variables[n].size for n in varnames))

        _set_flat_arr(sign_grad, varnames, self.__flat_gradient)
        if npmax(npabs(self.__flat_gradient)) <= pgtol:
            if verbose:
                print(
                    "Gradient near zero before the first iteration. "
                    "Returning the current value."
                )
            return

        r = self.__try_minimize(5, factr=factr, pgtol=pgtol)

        if r[2]["warnflag"] == 1:
            msg = "L-BFGS-B: too many function evaluations or too many iterations"
            raise OptimixError(msg)
        if r[2]["warnflag"] == 2:
            raise OptimixError("L-BFGS-B: {}".format(r[2]["task"]))

        _set_var_arr(r[0], self.__varnames(), self._variables)

    def _maximize(self, verbose=True, factr=FACTR, pgtol=PGTOL):
        self.__sign = -1.0
        try:
            self._minimize(verbose=verbose, factr=factr, pgtol=pgtol)
        finally:
            self.__sign = +1.0

    def __varnames(self):
        return sorted(self._variables.select(fixed=False).names())

    def __call__(self, x):
        from numpy import atleast_1d

        x = atleast_1d(x).ravel()
        self.__solutions.append(x.copy())
        _set_var_arr(x, self.__varnames(), self._variables)
        _set_flat_arr(self.gradient(), self.__varnames(), self.__flat_gradient)

        return self.__sign * self.value(), self.__sign * self.__flat_gradient

    def __try_minimize(self, n, factr, pgtol):
        from scipy.optimize import fmin_l_bfgs_b

        disp = 1 if self.__verbose else 0

        if n == 0:
            raise OptimixError("Too many bad solutions")

        warn = False
        res = []
        try:
            _set_flat_arr(self._variables, self.__varnames(), self.__flat_solution)

            bounds = []

            for name in self.__varnames():
                if self._variables[name].ndim == 0:
                    bounds.append(self._variables[name].bounds)
                else:
                    bounds += self._variables[name].bounds

            x0 = self.__flat_solution
            res = fmin_l_bfgs_b(
                self, x0, bounds=bounds, factr=factr, pgtol=pgtol, disp=disp
            )

        except OptimixError:
            warn = True
        else:
            warn = res[2]["warnflag"] > 0

        if warn:
            xs = self.__solutions
            if len(xs) < 2:
                raise OptimixError("Bad solution at the first iteration.")

            _set_var_arr(xs[-2] / 2 + xs[-1] / 2, self.__varnames(), self._variables)
            res = self.__try_minimize(n - 1, factr, pgtol)

        return res


class Function(FuncOpt):
    def __init__(self, name, composite=[], **kwargs):
        """
        Base-class for object representing functions.

        Parameters
        ----------
        name : str
            Function name.
        composite : list
            List of functions whose variables will be inherited.
        kwargs : dict
            Map of variable name to variable value.
        """
        self._name = name

        named_vars = {"": Variables(kwargs)}
        for (i, f) in enumerate(composite):
            if isinstance(f, tuple):
                named_vars[f[0]] = f[1]._variables
            else:
                named_vars[f"{self._name}[{i}]"] = f._variables

        super(Function, self).__init__(merge_variables(named_vars))

    @property
    def name(self):
        """
        Name of this function.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def _fix(self, var_name):
        r"""Set a variable fixed.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        self._variables[var_name].fix()

    def _unfix(self, var_name):
        r"""Set a variable unfixed.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        self._variables[var_name].unfix()

    def _isfixed(self, var_name):
        r"""Return whether a variable it is fixed or not.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        return self._variables[var_name].isfixed

    def _unfixed_names(self):
        return sorted(self._variables.select(fixed=False).names())

    def _check_grad(self, step=1.49e-08):
        from numpy import asarray
        from numpy.linalg import norm

        g = self.gradient()
        g = {n: asarray(gi) for n, gi in g.items()}
        fg = self._approx_fprime(step)

        names = set(g.keys()).intersection(fg.keys())
        return sum(norm(fg[name] - g[name]) for name in names)

    def _approx_fprime(self, step=1.49e-08):
        from numpy import atleast_1d, asarray, squeeze, stack

        f0 = self.value()
        grad = {}
        for name in self._variables.select(fixed=False).names():
            value = self._variables.get(name).value
            ndim = value.ndim
            value = atleast_1d(value).ravel()
            grads = []
            for i in range(len(value)):
                value[i] += step
                grads.append(asarray((self.value() - f0) / step))
                value[i] -= step
            grad[name] = stack(grads, axis=-1)
            if ndim == 0:
                grad[name] = squeeze(grad[name], axis=-1)
        return grad


def _set_flat_arr(arrs, names, out):
    from numpy import asarray

    offset = 0
    for name in names:
        arr = asarray(arrs[name])
        size = arr.size
        out[offset : offset + size] = arr
        offset += size
    return out


def _set_var_arr(flat_arr, names, dict_arr):
    offset = 0
    for name in names:
        size = dict_arr[name].size
        dict_arr[name].value = flat_arr[offset : offset + size]
        offset += size
