from __future__ import division

from numpy import asarray, newaxis, transpose, dot, array
from numpy.testing import assert_allclose

from optimix import Function, Scalar

class Quadratic1Scalar1(Function):
    def __init__(self):
        super(Quadratic1Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        s = self.get('scale')
        return (s - 5.0)**2 * x / 2.0

    def gradient(self, x):
        s = self.get('scale')
        return dict(scale=(s - 5.0) * x)

def test_function_quadratic1scalar1():
    f = Quadratic1Scalar1()
    f.set_data(1.2)
    f.feed().minimize(progress=False)
    assert_allclose(f.get('scale'), 5.0)
    f.set('scale', 1.0)

class Quadratic2Scalar1(Function):
    def __init__(self):
        super(Quadratic2Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x0, x1):
        s = self.get('scale')
        x0 = asarray(x0)[..., newaxis]
        x1 = asarray(x1)[..., newaxis]
        return (s - 5.0)**2 * dot(x0, transpose(x1)) / 2.0

    def gradient(self, x0, x1):
        s = self.get('scale')
        x0 = asarray(x0)[..., newaxis]
        x1 = asarray(x1)[..., newaxis]
        return dict(scale=(s - 5.0) * dot(x0, transpose(x1)))

def test_function_quadratic2scalar1():
    f = Quadratic2Scalar1()
    x1 = 2.3
    x2 = 1.0
    f.set_data((x1, x2))
    f.feed().minimize(progress=False)
    assert_allclose(f.get('scale'), 5.0)

class Quadratic1Scalar2(Function):
    def __init__(self):
        super(Quadratic1Scalar2, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x):
        a = self.get('a')
        b = self.get('b')
        return ((a - 5.0)**2 * (b + 5.0)**2 * x) / 2.0

    def gradient(self, x):
        return dict(a=self._derivative_a(x),
                    b=self._derivative_b(x))

    def _derivative_a(self, x):
        a = self.get('a')
        b = self.get('b')
        return 2 * (a - 5.0) * (b + 5.0)**2 * x

    def _derivative_b(self, x):
        a = self.get('a')
        b = self.get('b')
        return 2 * (a - 5.0)**2 * (b + 5.0) * x

def test_function_quadratic1scalar2():
    f = Quadratic1Scalar2()
    x = 1.2
    f.set_data(x)
    f.feed().minimize(progress=False)
    assert_allclose(f.get('a'), 4.99999999927461)
    assert_allclose(f.get('b'), -0.408820867345221)

class Quadratic2Scalar2(Function):
    def __init__(self):
        super(Quadratic2Scalar2, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x0, x1):
        a = self.get('a')
        b = self.get('b')
        return ((a - 5.0)**2 * x0 + (b + 5.0)**2 * x1) / 2.0

    def gradient(self, x0, x1):
        return dict(a=self._derivative_a(x0, x1),
                    b=self._derivative_b(x0, x1))

    def _derivative_a(self, x0, _):
        a = self.get('a')
        return 2 * (a - 5.0) * x0

    def _derivative_b(self, _, x1):
        b = self.get('b')
        return 2 * (b + 5.0) * x1

def test_function_quadratic2scalar2():
    f = Quadratic2Scalar2()
    x1 = 2.3
    x2 = 1.0
    f.set_data((x1, x2))
    f.feed().minimize(progress=False)
    assert_allclose(f.get('a'), 5.000000014635099)
    assert_allclose(f.get('b'), -4.999999925540513)

class VectorValued2Scalar2(Function):
    def __init__(self):
        super(VectorValued2Scalar2, self).__init__(
            a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x0, x1):
        a = self.get('a')
        b = self.get('b')
        return ((a - 5.0)**2 * x0 + (b + 5.0)**2 * x1) / 2.0

    def gradient(self, x0, x1):
        return dict(a=self._derivative_a(x0, x1),
                    b=self._derivative_b(x0, x1))

    def _derivative_a(self, x0, _):
        a = self.get('a')
        return 2 * (a - 5.0) * x0

    def _derivative_b(self, _, x1):
        b = self.get('b')
        return 2 * (b + 5.0) * x1

def test_function_vectorvalued2scalar2():
    f = VectorValued2Scalar2()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([0.0, -3.0, 1.0])
    assert_allclose(f.value(x1, x2), [12., -46., 18.])
    assert_allclose(f.gradient(x1, x2)['a'], array([-12., -8., -0.]))
    assert_allclose(f.gradient(x1, x2)['b'], array([0., -36., 12.]))


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
