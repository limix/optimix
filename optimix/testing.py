from numpy import asarray as _asarray
from numpy import concatenate as _concat
from .check_grad import check_grad

def _ni(v):
    return next(iter(v))


def _compact(x):
    from numpy import vstack
    return vstack(x)

def _do_flatten(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return _concat([_asarray(xi).ravel() for xi in x])
    return _concat(x)


class ProxyFunction(object):

    def __init__(self, function):
        self._function = function

    def value(self):
        return self._function.value()

    def gradient(self):
        return [gi for gi in self._function.gradient()]

    def __call__(self, x):
        x = _asarray(x).ravel()
        self._function.variables().select(fixed=False).from_flat(x)
        v = self.value()
        g = _do_flatten(self.gradient())
        return v, g

    def set_solution(self, x):
        t = self._function.variables().select(fixed=False)
        t.from_flat(_asarray(x).ravel())

    def get_solution(self):
        return self._function.variables().select(fixed=False).flatten()


class Assertion(object):
    def __init__(self, func, item0, item1, value_example,
                 **derivative_examples):
        self._func = func
        self._item0 = item0
        self._item1 = item1
        self._value_example = value_example
        self._derivative_examples = {
            'derivative_' + k: v
            for (k, v) in iter(derivative_examples.items())
        }

    def _isitem(self, v, e):
        return isinstance(v, type(e))

    def _isvector(self, v, e):
        return not self._isitem(v, e) and self._isitem(_ni(v), e)

    def _ismatrix(self, v, e):
        return not self._isitem(v, e) and not self._isvector(v, e) and _ni(_ni(v))

    def _get_containers(self):
        item0 = self._item0
        item1 = self._item1

        containers = [(item0, "item"), ([item0], "vector"),
                      (_compact([item0]), "vector"),
                      ([item0, item1], "vector"),
                      (_compact([item0, item1]), "vector")]

        return containers

    def assert_layout(self):
        self._assert_value_shape()
        self._assert_derivative_shape()

    def assert_gradient(self):
        containers = self._get_containers()
        derivative_names = self._func().get_derivative_list()

        for dn in derivative_names:
            for cx in containers:
                for cy in containers:

                    f = self._func()
                    f.set_data((cx[0], cy[0]))

                    def func(x):
                        t = f.variables().select(fixed=False)
                        t.from_flat(_asarray(x).ravel())
                        return f.feed().value()

                    def grad(x):
                        t = f.variables().select(fixed=False)
                        t.from_flat(_asarray(x).ravel())
                        return f.feed().gradient()

                    pf = ProxyFunction(f)
                    x0 = pf.get_solution()
                    # print(check_grad(func, grad, x0))

            # TODO: finish this
            # npt.assert_almost_equal(check_grad(func, grad, [2.0]), 0, decimal=6)

    def _assert_value_shape(self):

        func = self._func
        containers = self._get_containers()

        for cx in containers:
            for cy in containers:
                f = func()
                self.__assert_value_shape(f.value(cx[0], cy[0]), cx[1], cy[1])

    def _assert_derivative_shape(self):

        func = self._func

        containers = self._get_containers()

        derivative_names = func().get_derivative_list()

        for dn in derivative_names:
            for cx in containers:
                for cy in containers:
                    f = func()
                    d = getattr(f, dn)(cx[0], cy[0])
                    self.__assert_derivative_shape(d, cx[1], cy[1], dn)

    def __assert_value_shape(self, value, cx, cy):
        def errmsg(premiss):
            return "Interface premiss %s violated." % premiss

        if cx == "item" and cy == "item":
            if not self._isitem(value, self._value_example):
                raise AssertionError(errmsg("value(item, item) -> item"))
        elif cx == "item" and cy == "vector":
            if not self._isvector(value, self._value_example):
                raise AssertionError(errmsg("value(item, vector) -> vector"))
        elif cx == "vector" and cy == "item":
            if not self._isvector(value, self._value_example):
                raise AssertionError(errmsg("value(vector, item) -> vector"))
        elif cx == "vector" and cy == "vector":
            if not self._ismatrix(value, self._value_example):
                raise AssertionError(errmsg("value(vector, vector) -> matrix"))

    def __assert_derivative_shape(self, derivative, cx, cy, derivative_name):
        def errmsg(premiss):
            return "Interface premiss %s violated." % premiss

        if cx == "item" and cy == "item":
            if not self._isitem(derivative,
                                self._derivative_examples[derivative_name]):
                raise AssertionError(
                    errmsg("%s(item, item) -> item" % derivative_name))
        elif cx == "item" and cy == "vector":
            if not self._isvector(derivative,
                                  self._derivative_examples[derivative_name]):
                raise AssertionError(
                    errmsg("%s(item, vector) -> vector" % derivative_name))
        elif cx == "vector" and cy == "item":
            if not self._isvector(derivative,
                                  self._derivative_examples[derivative_name]):
                raise AssertionError(
                    errmsg("%s(vector, item) -> vector" % derivative_name))
        elif cx == "vector" and cy == "vector":
            if not self._ismatrix(derivative,
                                  self._derivative_examples[derivative_name]):
                raise AssertionError(
                    errmsg("%s(vector, vector) -> matrix" % derivative_name))
