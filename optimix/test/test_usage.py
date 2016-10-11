from numpy import zeros
from numpy import array
from numpy.testing import assert_almost_equal

from optimix import Function
from optimix import Scalar
from optimix import Vector
from optimix import minimize
from optimix import minimize_scalar

from quadratic_functions import Quadratic1Scalar1
from quadratic_functions import Quadratic2Scalar1
from quadratic_functions import Quadratic3Scalar1
from quadratic_functions import Quadratic4Scalar1

from quadratic_functions import Quadratic1Scalar2
from quadratic_functions import Quadratic2Scalar2
from quadratic_functions import Quadratic3Scalar2
from quadratic_functions import Quadratic4Scalar2

from quadratic_function_reduces import QuadraticScalarReduce


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
    from optimix.util import as_data_function
    assert_almost_equal(as_data_function(f).value(), 43.2)
    minimize(f)
    assert_almost_equal(as_data_function(f).value(), 0)
    assert_almost_equal(f1.get('scale'), 5, decimal=6)
    assert_almost_equal(f2.get('scale'), 5, decimal=6)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
