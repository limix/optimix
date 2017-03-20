from __future__ import division

from numpy import add, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose

from optimix import (Function, FunctionReduce, Scalar, Vector, approx_fprime,
                     check_grad)


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


class Quadratic1Scalar1(Function):
    def __init__(self):
        super(Quadratic1Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        s = self.variables().get('scale').value
        return (s - 5.0)**2 * x / 2.0

    def gradient(self, x):
        s = self.variables().get('scale').value
        return dict(scale=(s - 5.0) * x)


def test_check_grad():
    f = Quadratic1Scalar1()
    f.set_data(1.2)
    assert_allclose(check_grad(f.feed()), 0, atol=1e-7)


def test_check_grad_fprime():
    f = Quadratic1Scalar1()
    f.set_data(1.2)
    f.feed().minimize(progress=False)
    assert_allclose(approx_fprime(f.feed())['scale'], 0, atol=1e-7)


def test_check_grad_reduce():
    f0 = Quadratic1Scalar1()
    f1 = Quadratic1Scalar1()
    f = QuadraticScalarReduce([f0, f1])
    f0.set_data(1.1)
    f1.set_data(0.3)
    assert_allclose(check_grad(f.feed()), 0, atol=1e-7)


class LinearMean(Function):
    def __init__(self, size):
        Function.__init__(self, effsizes=Vector(zeros(size)))

    def value(self, x):
        return x.dot(self.variables().get('effsizes').value)

    def gradient(self, x): # pylint: disable=R0201
        return dict(effsizes=x)


def test_check_grad_vectors():
    random = RandomState(1)
    mean = LinearMean(5)
    x = random.randn(2, 5)
    mean.set_data(x)
    assert_allclose(check_grad(mean.feed()), 0, atol=1e-7)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
