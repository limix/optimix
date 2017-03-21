from __future__ import division

from numpy import add
from numpy.testing import assert_allclose

from optimix import Function, FunctionReduce, Scalar


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


def test_function_variables():
    f = Quadratic2Scalar2()
    assert f.variables().names()[0] == 'a'
    assert f.variables().names()[1] == 'b'


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


def test_function_variables_reduce():
    f00 = Quadratic2Scalar2()
    f01 = Quadratic2Scalar2()
    f10 = Quadratic2Scalar2()
    f11 = Quadratic2Scalar2()
    f0 = QuadraticScalarReduce([f00, f01])
    f1 = QuadraticScalarReduce([f10, f11])
    f = QuadraticScalarReduce([f0, f1])

    letter = ['a', 'b']
    names = sorted(f.variables().names())
    for i in range(2):
        for j in range(2):
            for ii in range(2):
                idx = i * 4 + j * 2 + ii
                assert names[idx] == 'sum[%d].sum[%d].%s' % (i, j, letter[ii])


def test_function_variables_gradient_reduce():
    f00 = Quadratic2Scalar2()
    f01 = Quadratic2Scalar2()
    f10 = Quadratic2Scalar2()
    f11 = Quadratic2Scalar2()
    f0 = QuadraticScalarReduce([f00, f01])
    f1 = QuadraticScalarReduce([f10, f11])
    f = QuadraticScalarReduce([f0, f1])

    f00.set_data((2.3, 1.0))
    f01.set_data((2.3, 1.0))
    f10.set_data((2.3, 1.0))
    f11.set_data((2.3, 1.0))

    assert_allclose(f.feed().value(), 145.6)
    gradient = f.feed().gradient()
    names = sorted(list(gradient.keys()))
    letter = ['a', 'b']
    for i in range(2):
        for j in range(2):
            for ii in range(2):
                idx = i * 4 + j * 2 + ii
                assert names[idx] == 'sum[%d].sum[%d].%s' % (i, j, letter[ii])


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
