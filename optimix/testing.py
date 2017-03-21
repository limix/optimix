from numpy import asarray as _asarray
from numpy import concatenate as _concat
from numpy import stack
from numpy.testing import assert_allclose

from ndarray_listener import ndarray_listener

from .check_grad import check_grad


def _ni(v):
    return next(iter(v))


def _compact(x):
    return stack(x, axis=0)


def _do_flatten(x):
    if isinstance(x, (list, tuple)):
        return _concat([_asarray(xi).ravel() for xi in x])
    return _concat(x)


def _isitem(v, e):
    v = ndarray_listener(v)
    return isinstance(v, type(e)) and v.shape == e.shape


def _isvector(v, e):
    return not _isitem(v, e) and _isitem(_ni(v), e)


def _ismatrix(v, e):
    return not _isitem(v, e) and not _isvector(v, e) and _isitem(
        _ni(_ni(v)), e)


class Assertion(object):
    def __init__(self, func, item0, item1, value_example,
                 **derivative_examples):
        self._func = func
        self._item0 = item0
        self._item1 = item1
        self._value_example = ndarray_listener(value_example)
        de = derivative_examples
        de = {k: ndarray_listener(v) for k, v in de.items()}
        self._derivative_examples = {k: v for (k, v) in iter(de.items())}

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

        for cx in containers:
            for cy in containers:
                f = self._func()
                f.set_data((cx[0], cy[0]))
                assert_allclose(check_grad(f.feed()), 0, atol=1e-6)

    def _assert_value_shape(self):

        func = self._func
        containers = self._get_containers()

        for cx in containers:
            for cy in containers:
                f = func()
                self._assert_valshape_msg(f.value(cx[0], cy[0]), cx, cy)

    def _assert_derivative_shape(self):

        func = self._func

        containers = self._get_containers()

        for cx in containers:
            for cy in containers:
                f = func()
                d = f.gradient(cx[0], cy[0])
                for name in d.keys():
                    self._assert_dershape_message(d[name], cx, cy, name)

    def _assert_valshape_msg(self, value, cx, cy):

        if cx[1] == "item" and cy[1] == "item":
            if not _isitem(value, self._value_example):
                raise AssertionError(
                    _errmsg("value(item, item) -> item", value, cx[0], cy[0]))
        elif cx[1] == "item" and cy[1] == "vector":
            if not _isvector(value, self._value_example):
                raise AssertionError(
                    _errmsg("value(item, vector) -> vector", value, cx[0], cy[
                        0]))
        elif cx[1] == "vector" and cy[1] == "item":
            if not _isvector(value, self._value_example):
                raise AssertionError(
                    _errmsg("value(vector, item) -> vector", value, cx[0], cy[
                        0]))
        elif cx[1] == "vector" and cy[1] == "vector":
            if not _ismatrix(value, self._value_example):
                raise AssertionError(
                    _errmsg("value(vector, vector) -> matrix", value, cx[0],
                            cy[0]))

    def _assert_dershape_message(self, der, cx, cy, der_name):

        if cx[1] == "item" and cy[1] == "item":
            if not _isitem(der, self._derivative_examples[der_name]):
                raise AssertionError(
                    _errmsg("%s(item, item) -> item" % der_name, der, cx[0],
                            cy[0]))
        elif cx[1] == "item" and cy[1] == "vector":
            if not _isvector(der, self._derivative_examples[der_name]):
                raise AssertionError(
                    _errmsg("%s(item, vector) -> vector" % der_name, der, cx[
                        0], cy[0]))
        elif cx[1] == "vector" and cy[1] == "item":
            if not _isvector(der, self._derivative_examples[der_name]):
                raise AssertionError(
                    _errmsg("%s(vector, item) -> vector" % der_name, der, cx[
                        0], cy[0]))
        elif cx[1] == "vector" and cy[1] == "vector":
            if not _ismatrix(der, self._derivative_examples[der_name]):
                raise AssertionError(
                    _errmsg("%s(vector, vector) -> matrix" % der_name, der, cx[
                        0], cy[0]))


def _errmsg(premiss, value, lval, rval):
    msg = "Interface premiss %s violated." % premiss
    msg += "\n  Got (%s:%s, %s:%s) -> %s:%s instead." % (
        type(lval), lval, type(rval), rval, type(value), value)
    return msg
