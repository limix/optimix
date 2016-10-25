from numpy import array, zeros
from numpy.testing import (assert_almost_equal, assert_allclose)

from optimix import (Composite, Function, Scalar, Vector, minimize,
                     minimize_scalar)
from quadratic_function_reduces import QuadraticScalarReduce
from quadratic_functions import (Quadratic1Scalar1, Quadratic1Scalar2,
                                 Quadratic2Scalar1, Quadratic2Scalar2,
                                 Quadratic3Scalar1, Quadratic3Scalar2,
                                 Quadratic4Scalar1, Quadratic4Scalar2)


def test_quadratic1scalar1_layout():
    f = Quadratic1Scalar1()
    f.set_data(1.2)
    minimize(f)
    assert_almost_equal(f.get('scale'), 5.0)
    f.set('scale', 1.0)
    minimize_scalar(f)


def test_quadratic2scalar1_layout():
    f = Quadratic2Scalar1()
    x1 = 2.3
    x2 = 1.0
    f.set_data((x1, x2))
    minimize(f)
    assert_almost_equal(f.get('scale'), 5.0)


def test_quadratic3scalar1_layout():
    f = Quadratic3Scalar1()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([1.5, 1.0, 0.0])
    f.set_data((x1, x2))
    minimize(f)
    assert_almost_equal(f.get('scale'), 5.0)


def test_quadratic4scalar1_layout():
    f = Quadratic4Scalar1()
    x1 = array([[1.5, 1.0, 0.0],
                [1.5, 5.0, 0.0]])
    x2 = array([[-1.5, 1.0, 0.0],
                [1.5, 5.0, 0.0]])
    f.set_data((x1, x2))
    minimize(f)
    assert_almost_equal(f.get('scale'), 5.0)


def test_quadratic1scalar2_layout():
    f = Quadratic1Scalar2()
    x = 1.2
    f.set_data(x)
    minimize(f)
    assert_almost_equal(f.get('a'), 9.23644786057)
    assert_almost_equal(f.get('b'), -4.99999846748)


def test_quadratic2scalar2_layout():
    f = Quadratic2Scalar2()
    x1 = 2.3
    x2 = 1.0
    f.set_data((x1, x2))
    minimize(f)
    assert_almost_equal(f.get('a'), 4.99999790547)
    assert_almost_equal(f.get('b'), -4.99999948821)


def test_quadratic3scalar2_layout():
    f = Quadratic3Scalar2()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([1.5, 1.0, 0.0])
    f.set_data((x1, x2))
    minimize(f)
    assert_almost_equal(f.get('a'), 9.23644786057)
    assert_almost_equal(f.get('b'), -4.99999846748)


def test_quadratic4scalar2_layout():
    f = Quadratic4Scalar2()
    x1 = array([[1.5, 1.0, 0.0],
                [1.5, 5.0, 0.0]])
    x2 = array([[-1.5, 1.0, 0.0],
                [1.5, 5.0, 0.0]])
    f.set_data((x1, x2))
    minimize(f)
    assert_almost_equal(f.get('a'), 5.0)
    assert_almost_equal(f.get('b'), -1.48771687512)


def test_quadratic1scalar1_reduce_layout():
    f1 = Quadratic1Scalar1()
    f2 = Quadratic1Scalar1()
    f = QuadraticScalarReduce([f1, f2])
    f1.set_data(1.2)
    f2.set_data(+4.2)
    assert_almost_equal(f.feed().value(), 43.2)
    minimize(f)
    assert_almost_equal(f.feed().value(), 0)
    assert_almost_equal(f1.get('scale'), 5, decimal=6)
    assert_almost_equal(f2.get('scale'), 5, decimal=6)


def test_composite():
    f1 = Quadratic1Scalar1()
    f2 = Quadratic2Scalar1()

    class SumFunction(Composite):

        def __init__(self, f1, f2):
            super(SumFunction, self).__init__(f1=f1, f2=f2)
            self._f1 = f1
            self._f2 = f2

        def value(self, f1, f2):
            return f1 + f2

        def gradient_f1(self, f1, f2, gf1, gf2):
            return gf1

        def gradient_f2(self, f1, f2, gf1, gf2):
            return gf2

    s = SumFunction(f1, f2)
    assert_allclose(s.value(f1.value(1.5), f2.value(-0.2, 3.2)),
                    6.879999999999999)
    f1.set('scale', 2.0)
    assert_allclose(s.value(f1.value(1.5), f2.value(-0.2, 3.2)),
                    1.629999999999999)
    assert_allclose(s.gradient_f1(f1.value(1.5), f2.value(-0.2, 3.2),
                    f1.gradient(1.5), f2.gradient(-0.2, 3.2)), [-4.5])
    assert_allclose(s.gradient_f2(f1.value(1.5), f2.value(-0.2, 3.2),
                    f1.gradient(1.5), f2.gradient(-0.2, 3.2)),
                    [2.5600000000000005])
    assert_allclose(s.gradient(f1.value(1.5), f2.value(-0.2, 3.2),
                    f1.gradient(1.5), f2.gradient(-0.2, 3.2)),
                    [-4.5, 2.5600000000000005])

    f1.set_data(1.5)
    f2.set_data([-0.2, 3.2])

    sa = s.feed()
    assert_allclose(sa.value(), 1.629999999999999)
    assert_allclose(sa.gradient(), [-4.5, 2.5600000000000005])

def test_composite_minimize():
    f1 = Quadratic1Scalar1()
    f2 = Quadratic2Scalar1()

    class SumFunction(Composite):

        def __init__(self, f1, f2):
            super(SumFunction, self).__init__(f1=f1, f2=f2)
            self._f1 = f1
            self._f2 = f2

        def value(self, f1, f2):
            return f1 + f2

        def gradient_f1(self, f1, f2, gf1, gf2):
            return gf1

        def gradient_f2(self, f1, f2, gf1, gf2):
            return gf2

    s = SumFunction(f1, f2)

    f1.set_data(1.5)
    f2.set_data([+0.2, 3.2])

    minimize(s)
    sa = s.feed()
    assert_allclose(sa.value(), 0, atol=1e-6)
    assert_allclose(sa.gradient(), [0, 0], atol=1e-6)



if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
