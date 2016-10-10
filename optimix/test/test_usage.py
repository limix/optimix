from numpy import zeros
from numpy import array
from numpy.testing import assert_almost_equal

from optimix import Function
from optimix import Scalar
from optimix import Vector
from optimix import minimize


class Quadratic1(Function):

    def __init__(self):
        self._scale = Scalar(1.0)
        super(Quadratic1, self).__init__(scale=self._scale)

    @property
    def scale(self):
        return self._scale.value

    @scale.setter
    def scale(self, scale):
        self._scale.value = scale

    def value(self, x):
        return (self.scale - 5.0)**2 * x / 2.0

    def derivative_scale(self, x):
        return (self.scale - 5.0) * x


class Quadratic2(Function):

    def __init__(self):
        self._scale = Scalar(1.0)
        super(Quadratic2, self).__init__(scale=self._scale)

    @property
    def scale(self):
        return self._scale.value

    @scale.setter
    def scale(self, scale):
        self._scale.value = scale

    def value(self, x0, x1):
        return (self.scale - 5.0)**2 * x0 * x1 / 2.0

    def derivative_scale(self, x0, x1):
        return (self.scale - 5.0) * x0 * x1


class Quadratic3(Function):

    def __init__(self):
        self._scale = Scalar(1.0)
        super(Quadratic3, self).__init__(scale=self._scale)

    @property
    def scale(self):
        return self._scale.value

    @scale.setter
    def scale(self, scale):
        self._scale.value = scale

    def value(self, x0, x1):
        return (self.scale - 5.0)**2 * x0.dot(x1) / 2.0

    def derivative_scale(self, x0, x1):
        return (self.scale - 5.0) * x0.dot(x1)


def test_quadratic1_minimize():
    f = Quadratic1()
    f.set_data(2.3)
    minimize(f)
    assert_almost_equal(f.scale, 5)


def test_quadratic2_minimize():
    f = Quadratic2()
    f.set_data((2.3, 1.0))
    minimize(f)
    assert_almost_equal(f.scale, 5)


def test_quadratic3_minimize():
    f = Quadratic3()
    x0 = array([1.5, 1.0, 0.0])
    x1 = array([1.5, 1.0, 0.0])
    f.set_data((x0, x1))
    minimize(f)
    assert_almost_equal(f.scale, 5)
