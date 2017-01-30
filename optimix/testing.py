def _ni(v):
    return next(iter(v))


def _compact(x):
    from numpy import vstack
    return vstack(x)


class Assertion(object):
    def __init__(self, func, item0, item1, value_example):
        self._func = func
        self._item0 = item0
        self._item1 = item1
        self._value_example = value_example

    def isitem(self, v):
        e = self._value_example
        return isinstance(v, type(e))

    def isvector(self, v):
        return not self.isitem(v) and self.isitem(_ni(v))

    def ismatrix(self, v):
        return not self.isitem(v) and not self.isvector(v) and _ni(_ni(v))

    def assert_value_shape(self):

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
                self.assert_shape(f.value(cx[0], cy[0]), cx[1], cy[1])

    def assert_shape(self, value, cx, cy):
        def errmsg(premiss):
            return "Interface premiss %s violated." % premiss

        if cx == "item" and cy == "item":
            if not self.isitem(value):
                raise AssertionError(errmsg("value(item, item) -> item"))
        elif cx == "item" and cy == "vector":
            if not self.isvector(value):
                raise AssertionError(errmsg("value(item, vector) -> vector"))
        elif cx == "vector" and cy == "item":
            if not self.isvector(value):
                raise AssertionError(errmsg("value(vector, item) -> vector"))
        elif cx == "vector" and cy == "vector":
            if not self.ismatrix(value):
                raise AssertionError(errmsg("value(vector, vector) -> matrix"))
