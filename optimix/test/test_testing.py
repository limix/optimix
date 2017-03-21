from numpy import asarray, dot, newaxis, transpose

from optimix import Function, Scalar
from optimix.testing import Assertion


class Scalar1(Function):
    def __init__(self):
        super(Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        s = self.variables().get('scale').value
        return s * x

    def gradient(self, x):  # pylint: disable=R0201
        return dict(scale=x)


class Quadratic2Scalar1(Function):
    def __init__(self):
        super(Quadratic2Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x0, x1):
        s = self.variables().get('scale').value
        x0 = asarray(x0)[..., newaxis]
        x1 = asarray(x1)[..., newaxis]
        return (s - 5.0)**2 * dot(x0, transpose(x1)) / 2.0

    def gradient(self, x0, x1):
        s = self.variables().get('scale').value
        x0 = asarray(x0)[..., newaxis]
        x1 = asarray(x1)[..., newaxis]
        return dict(scale=(s - 5.0) * dot(x0, transpose(x1)))


def test_testing():
    x0 = 2.3
    x1 = 1.0
    value = 18.4

    a = Assertion(Quadratic2Scalar1, x0, x1, value, scale=1.0)
    a.assert_layout()
    a.assert_gradient()

    a = Assertion(Scalar1, x0, x1, value, scale=1.0)
    a.assert_layout()
    a.assert_gradient()


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
