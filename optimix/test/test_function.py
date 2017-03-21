from __future__ import division

from numpy import add, array, asarray, dot, empty, newaxis, transpose
from numpy.testing import assert_allclose

from optimix import Function, FunctionReduce, Scalar, Vector, check_grad


class Quadratic1Scalar1(Function):
    def __init__(self):
        super(Quadratic1Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        s = self.variables().get('scale').value
        return (s - 5.0)**2 * x / 2.0

    def gradient(self, x):
        s = self.variables().get('scale').value
        return dict(scale=(s - 5.0) * x)


def test_function_quadratic1scalar1():
    f = Quadratic1Scalar1()
    f.set_data(1.2)
    f.feed().minimize(progress=False)
    assert_allclose(f.variables().get('scale').value, 5.0)
    f.variables().get('scale').value = 1.0


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


def test_function_quadratic2scalar1():
    f = Quadratic2Scalar1()
    x1 = 2.3
    x2 = 1.0
    f.set_data((x1, x2))
    f.feed().minimize(progress=False)
    assert_allclose(f.variables().get('scale').value, 5.0)


class Quadratic1Scalar2(Function):
    def __init__(self):
        super(Quadratic1Scalar2, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x):
        a = self.variables().get('a').value
        b = self.variables().get('b').value
        return ((a - 5.0)**2 * (b + 5.0)**2 * x) / 2.0

    def gradient(self, x):
        return dict(a=self._derivative_a(x), b=self._derivative_b(x))

    def _derivative_a(self, x):
        a = self.variables().get('a').value
        b = self.variables().get('b').value
        return 2 * (a - 5.0) * (b + 5.0)**2 * x

    def _derivative_b(self, x):
        a = self.variables().get('a').value
        b = self.variables().get('b').value
        return 2 * (a - 5.0)**2 * (b + 5.0) * x


def test_function_quadratic1scalar2():
    f = Quadratic1Scalar2()
    x = 1.2
    f.set_data(x)
    f.feed().minimize(progress=False)
    assert_allclose(f.variables().get('a').value, 4.99999999927461)
    assert_allclose(f.variables().get('b').value, -0.408820867345221)


class Quadratic2Scalar2(Function):
    def __init__(self):
        super(Quadratic2Scalar2, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x0, x1):
        a = self.variables().get('a').value
        b = self.variables().get('b').value
        return ((a - 5.0)**2 * x0 + (b + 5.0)**2 * x1) / 2.0

    def gradient(self, x0, x1):
        return dict(a=self._derivative_a(x0, x1), b=self._derivative_b(x0, x1))

    def _derivative_a(self, x0, _):
        a = self.variables().get('a').value
        return 2 * (a - 5.0) * x0

    def _derivative_b(self, _, x1):
        b = self.variables().get('b').value
        return 2 * (b + 5.0) * x1


def test_function_quadratic2scalar2():
    f = Quadratic2Scalar2()
    x1 = 2.3
    x2 = 1.0
    f.set_data((x1, x2))
    f.feed().minimize(progress=False)
    assert_allclose(f.variables().get('a').value, 5.000000014635099)
    assert_allclose(f.variables().get('b').value, -4.999999925540513)


class VectorValued(Function):
    def __init__(self):
        super(VectorValued, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x0, x1):
        a = self.variables().get('a').value
        b = self.variables().get('b').value
        return ((a - 5.0)**2 * x0 + (b + 5.0)**2 * x1) / 2.0

    def gradient(self, x0, x1):
        return dict(a=self._derivative_a(x0, x1), b=self._derivative_b(x0, x1))

    def _derivative_a(self, x0, _):
        a = self.variables().get('a').value
        return 2 * (a - 5.0) * x0

    def _derivative_b(self, _, x1):
        b = self.variables().get('b').value
        return 2 * (b + 5.0) * x1


def test_function_vectorvalued():
    f = VectorValued()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([0.0, -3.0, 1.0])
    assert_allclose(f.value(x1, x2), [12., -46., 18.])
    assert_allclose(f.gradient(x1, x2)['a'], array([-12., -8., -0.]))
    assert_allclose(f.gradient(x1, x2)['b'], array([0., -36., 12.]))


class VectorValuedMix(Function):
    def __init__(self):
        super(VectorValuedMix, self).__init__(
            a=Scalar(1.0), b=Vector([1.0, 2.0]))

    def value(self, x0, x1):
        a = self.variables().get('a').value
        b = self.variables().get('b').value
        return ((a - 5.0)**2 * x0 + 5.0**2 * x1 + sum(b * b)) / 2.0

    def gradient(self, x0, x1):
        return dict(a=self._derivative_a(x0, x1), b=self._derivative_b(x0, x1))

    def _derivative_a(self, x0, _):
        a = self.variables().get('a').value
        return 2 * (a - 5.0) * x0

    def _derivative_b(self, x0, _):
        b = self.variables().get('b').value
        g = empty((len(x0), len(b)))
        g[:] = b
        return g


def test_function_vectorvaluedmix():
    f = VectorValuedMix()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([0.0, -3.0, 1.0])
    assert_allclose(f.value(x1, x2), [14.5, -27., 15.])
    assert_allclose(f.gradient(x1, x2)['a'], [-12., -8., -0.])
    assert_allclose(f.gradient(x1, x2)['b'], [[1., 2.], [1., 2.], [1., 2.]])


class QuadraticScalarReduce(FunctionReduce):
    def __init__(self, functions):
        super(QuadraticScalarReduce, self).__init__(functions, 'sum')

    def value_reduce(self, values):  # pylint: disable=R0201
        return add.reduce(list(values.values()))

    def gradient_reduce(self, _, gradients):  # pylint: disable=R0201
        grad = dict()
        for (gn, gv) in iter(gradients.items()):
            for n, v in iter(gv.items()):
                grad[gn + '.' + n] = v
        return grad


def test_function_quadratic1scalar1_reduce():
    f1 = Quadratic1Scalar1()
    f2 = Quadratic1Scalar1()
    f = QuadraticScalarReduce([f1, f2])
    x1 = 1.2
    x2 = -1.1
    f1.set_data(x1)
    f2.set_data(x2)

    f = f.feed()
    assert_allclose(f.value(), 0.8)
    assert_allclose(check_grad(f), 0, atol=1e-6)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
