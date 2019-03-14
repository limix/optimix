from numpy import asarray, atleast_1d
from numpy.testing import assert_, assert_allclose

from optimix import Scalar, Vector


def test_types_scalar_comparison():
    a = Scalar(1.0)
    b = Scalar(2.0)

    assert_(a < b)
    assert_(a <= b)
    assert_(a != b)

    b.value = 1.0

    assert_(a == b)


def test_types_scalar_fix():
    a = Scalar(1.0)

    assert_(not a.isfixed)

    a.fix()
    assert_(a.isfixed)


def test_types_scalar_copy():
    a = Scalar(1.0)
    b = a.copy()

    assert_(a is not b)
    assert_(a == b)


def test_types_scalar_listen():
    a = Scalar(1.0)

    class Listener(object):
        def __init__(self):
            self.value = None

        def __call__(self):
            self.value = 3.0

    l = Listener()
    a.listen(l)
    a.value = 3.0

    assert_(l.value == 3.0)


def test_types_scalar_listen_indirect():
    a = Scalar(1.0)

    class Listener(object):
        def __init__(self):
            self.value = None

        def __call__(self):
            self.value = 3.0

    l = Listener()

    a.listen(l)

    value = a.value
    value.itemset(3.0)

    assert_(l.value == 3.0)


def test_types_vector_comparison():
    a = Vector([1.0, 2.0])
    b = Vector([1.0, 1.5])

    assert_((a > b)[1])
    assert_((a == b)[0])
    assert_((a != b)[1])

    b.value = asarray([1.0, 2.0])

    assert_(all(a == b))


def test_types_vector_fix():
    a = Vector([1.0, 2.0])

    assert_(not a.isfixed)

    a.fix()
    assert_(a.isfixed)


def test_types_vector_copy():
    a = Vector([1.0])
    b = a.copy()

    assert_(a is not b)
    assert_(a == b)


def test_types_vector_listen():
    a = Vector([1.0, 2.0])

    class Listener(object):
        def __init__(self):
            self.value = None

        def __call__(self):
            self.value = asarray([3.0, -1.0])

    l = Listener()
    a.listen(l)
    a.value = asarray([3.0, -1.0])
    assert_allclose(l.value, [3.0, -1.0])


def test_types_modify_scalar():
    a = Scalar(1.0)
    value = atleast_1d(a.value)
    value[0] = 2.0
    assert_(a.value == value[0])


def test_types_modify_vector():
    a = Vector([1.0, 2.0])
    value = atleast_1d(a.value)
    value[0] = 2.0
    assert_allclose(a.value, value)
