from numpy import asarray
from numpy.testing import assert_allclose

from optimix import Scalar, Vector

def test_types_scalar_comparison():
    a = Scalar(1.0)
    b = Scalar(2.0)

    assert a < b
    assert a <= b
    assert a != b

    b.value = 1.0

    assert a == b

def test_types_scalar_fix():
    a = Scalar(1.0)

    assert not a.isfixed

    a.fix()
    assert a.isfixed

def test_types_scalar_listen():
    a = Scalar(1.0)

    class Listener(object): # pylint: disable=R0903
        def __init__(self):
            self.value = None

        def __call__(self, value):
            self.value = value

    l = Listener()
    a.listen(l)
    a.value = 3.0

    assert l.value == 3.0

def test_types_vector_fix():
    a = Vector([1.0, 2.0])

    assert not a.isfixed

    a.fix()
    assert a.isfixed

def test_types_vector_listen():
    a = Vector([1.0, 2.0])

    class Listener(object): # pylint: disable=R0903
        def __init__(self):
            self.value = None

        def __call__(self, value):
            self.value = value

    l = Listener()
    a.listen(l)
    a.value = asarray([3.0, -1.0])
    assert_allclose(l.value, [3.0, -1.0])
