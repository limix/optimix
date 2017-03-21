from numpy import asarray as _asarray
from numpy import concatenate as _concat
from numpy import stack
from numpy.testing import assert_allclose

from ndarray_listener import ndarray_listener

from .check_grad import check_grad


def _nparams(o):
    try:
        from inspect import signature

        return len(signature(o.value).parameters)
    except ImportError:
        from inspect import getargspec

        return len(getargspec(o.value).args) - 1 # pylint: disable=W1505

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
        self._assert_value_shape([], [])

    def assert_gradient(self):
        self._assert_gradient_value([])

    def _assert_gradient_value(self, input_values):
        f = self._func()
        if len(input_values) == _nparams(f):
            f.set_data(tuple(input_values))
            assert_allclose(check_grad(f.feed()), 0, atol=1e-6)
            return

        for cx in self._get_containers():
            self._assert_gradient_value(input_values + [cx[0]])

    def _assert_value_shape(self, input_values, container_names):

        if len(input_values) == _nparams(self._func()):
            v = self._func().value(*input_values)
            self._assert_valshape_msg(
                v, input_values=input_values, container_names=container_names)
            return

        for cx in self._get_containers():
            self._assert_value_shape(input_values + [cx[0]],
                                     container_names + [cx[1]])

    def _assert_valshape_msg(self, value, input_values, container_names):
        if len(input_values) == 1:
            return _assert_valshape_msg_1d(self._value_example, value,
                                           input_values[0], container_names[0])
        elif len(input_values) == 2:
            return _assert_valshape_msg_2d(self._value_example, value,
                                           input_values, container_names)
        assert False


def _assert_valshape_msg_1d(example, value, x, xname):
    def _errmsg(premiss, value, lval):
        msg = "Interface premiss %s violated." % premiss
        msg += "\n  Got (%s:%s, ) -> %s:%s instead." % (type(lval), lval,
                                                        type(value), value)
        return msg

    if xname == "item":
        if not _isitem(value, example):
            raise AssertionError(_errmsg("value(item, ) -> item", value, x))
    elif xname == "vector":
        if not _isvector(value, example):
            raise AssertionError(_errmsg("value(item, ) -> vector", value, x))


def _assert_valshape_msg_2d(example, value, xy, xyname):
    def _errmsg(premiss, value, lval, rval):
        msg = "Interface premiss %s violated." % premiss
        msg += "\n  Got (%s:%s, %s:%s) -> %s:%s instead." % (
            type(lval), lval, type(rval), rval, type(value), value)
        return msg

    x, y = xy
    xname, yname = xyname

    if xname == "item" and yname == "item":
        if not _isitem(value, example):
            raise AssertionError(
                _errmsg("value(item, item) -> item", value, x, y))
    elif xname == "item" and yname == "vector":
        if not _isvector(value, example):
            raise AssertionError(
                _errmsg("value(item, vector) -> vector", value, x, y))
    elif xname == "vector" and yname == "item":
        if not _isvector(value, example):
            raise AssertionError(
                _errmsg("value(vector, item) -> vector", value, x, y))
    elif xname == "vector" and yname == "vector":
        if not _ismatrix(value, example):
            raise AssertionError(
                _errmsg("value(vector, vector) -> matrix", value, x, y))
