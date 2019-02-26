from ._exception import OptimixError

FACTR = 1e5
PGTOL = 1e-7


class FuncOpt:
    def __init__(self):
        self.__solutions = []
        self.__sign = +1.0
        self.__verbose = True
        self.__flat_gradient = None
        self.__flat_solution = None

    def minimize(self, verbose=True, factr=FACTR, pgtol=PGTOL):
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
        elif r[2]["warnflag"] == 2:
            raise OptimixError("L-BFGS-B: {}".format(r[2]["task"]))

        _set_var_arr(r[0], self.__varnames(), self._variables)

    def maximize(self, verbose=True, factr=FACTR, pgtol=PGTOL):
        self.__sign = -1.0
        self.minimize(verbose=verbose, factr=factr, pgtol=pgtol)
        self.__sign = +1.0

    def __sign_value(self):
        return self.__sign * self.value()

    def __varnames(self):
        return sorted(self._variables.select(fixed=False).names())

    def __call__(self, x):
        from numpy import atleast_1d

        _set_var_arr(atleast_1d(x).ravel(), self.__varnames(), self._variables)
        _set_flat_arr(self.gradient(), self.__varnames(), self.__flat_gradient)

        return self.__sign * self.value(), self.__sign * self.__flat_gradient

    def __try_minimize(self, n, factr, pgtol):
        from scipy.optimize import fmin_l_bfgs_b

        disp = 1 if self.__verbose else 0

        if n == 0:
            raise OptimixError("Too many bad solutions")

        warn = False
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

            _set_var_arr(xs[-2] / 5 + xs[-1] / 5, self.__varnames(), self._variables)
            res = self.__try_minimize(n - 1, factr, pgtol)

        return res


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
