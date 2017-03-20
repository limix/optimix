from __future__ import division

from numpy import array, asarray, dot, empty, newaxis, transpose, add
from numpy.testing import assert_allclose

from optimix import Function, Scalar, Vector, FunctionReduce, check_grad, approx_fprime

class QuadraticScalarReduce(FunctionReduce):
    def __init__(self, functions):
        super(QuadraticScalarReduce, self).__init__(functions, 'sum')

    def value_reduce(self, values): # pylint: disable=R0201
        return add.reduce(values)

    def derivative_reduce(self, derivatives): # pylint: disable=R0201
        return add.reduce(derivatives)

class Quadratic1Scalar1(Function):
    def __init__(self):
        super(Quadratic1Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        s = self.get('scale')
        return (s - 5.0)**2 * x / 2.0

    def gradient(self, x):
        s = self.get('scale')
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

# def test_check_grad_reduce():
#     f0 = Quadratic1Scalar1()
#     f1 = Quadratic1Scalar1()
#     f = QuadraticScalarReduce([f0, f1])
#     f0.set_data(1.1)
#     f1.set_data(0.3)
#     print(check_grad(f.feed()))

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
