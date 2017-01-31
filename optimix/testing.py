def _ni(v):
    return next(iter(v))


def _compact(x):
    from numpy import vstack
    return vstack(x)


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

    def assert_layout(self):
        self._assert_value_shape()
        self._assert_derivative_shape()

    def _assert_value_shape(self):

        func = self._func
        item0 = self._item0
        item1 = self._item1

        containers = [(item0, "item"), ([item0], "vector"),
                      (_compact([item0]), "vector"),
                      ([item0, item1], "vector"),
                      (_compact([item0, item1]), "vector")]

        for cx in containers:
            for cy in containers:
                f = func()
                self.__assert_value_shape(f.value(cx[0], cy[0]), cx[1], cy[1])

    def _assert_derivative_shape(self):

        func = self._func
        item0 = self._item0
        item1 = self._item1

        containers = [(item0, "item"), ([item0], "vector"),
                      (_compact([item0]), "vector"),
                      ([item0, item1], "vector"),
                      (_compact([item0, item1]), "vector")]

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
