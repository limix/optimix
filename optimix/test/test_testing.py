from numpy import asarray, newaxis, transpose, dot

from optimix.testing import Assertion
from optimix import Scalar, Function

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
        bla = (s - 5.0) * dot(x0, transpose(x1))
        return dict(scale=bla)


def test_testing():
    x0 = 2.3
    x1 = 1.0
    value = 18.4

    a = Assertion(Quadratic2Scalar1, x0, x1, value, scale=1.0)
    # a.assert_layout()
    a.assert_gradient()

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
