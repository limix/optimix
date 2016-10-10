from numpy import zeros
from numpy import array
from numpy.testing import assert_almost_equal

from quadratic_functions import Quadratic1Scalar1
from quadratic_functions import Quadratic2Scalar1
from quadratic_functions import Quadratic3Scalar1
from quadratic_functions import Quadratic4Scalar1

from quadratic_functions import Quadratic1Scalar2
from quadratic_functions import Quadratic2Scalar2
from quadratic_functions import Quadratic3Scalar2
from quadratic_functions import Quadratic4Scalar2

from vector_valued_functions import VectorValued1Scalar1
from vector_valued_functions import VectorValued2Scalar1
from vector_valued_functions import VectorValued1Scalar2
from vector_valued_functions import VectorValued2Scalar2


def test_quadratic1scalar1_layout():
    f = Quadratic1Scalar1()
    x = 1.2
    assert_almost_equal(f.value(x), 9.6)
    assert_almost_equal(f.gradient(x), [-4.8])


def test_quadratic2scalar1_layout():
    f = Quadratic2Scalar1()
    x1 = 2.3
    x2 = 1.0
    assert_almost_equal(f.value(x1, x2), 18.4)
    assert_almost_equal(f.gradient(x1, x2), [-9.2])


def test_quadratic3scalar1_layout():
    f = Quadratic3Scalar1()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([1.5, 1.0, 0.0])
    assert_almost_equal(f.value(x1, x2), 26)
    assert_almost_equal(f.gradient(x1, x2), [-13.0])


def test_quadratic4scalar1_layout():
    f = Quadratic4Scalar1()
    x1 = array([[1.5, 1.0, 0.0],
                [1.5, 5.0, 0.0]])
    x2 = array([[-1.5, 1.0, 0.0],
                [1.5, 5.0, 0.0]])
    assert_almost_equal(f.value(x1, x2), 226.8744146)
    assert_almost_equal(f.gradient(x1, x2), [-113.43720729989786])


def test_quadratic1scalar2_layout():
    f = Quadratic1Scalar2()
    x = 1.2
    assert_almost_equal(f.value(x), 345.6)
    assert_almost_equal(f.gradient(x), [-345.59999999999997,
                                        230.39999999999998])


def test_quadratic2scalar2_layout():
    f = Quadratic2Scalar2()
    x1 = 2.3
    x2 = 1.0
    assert_almost_equal(f.value(x1, x2), 36.4)
    assert_almost_equal(f.gradient(x1, x2), [-18.4, 12.0])


def test_quadratic3scalar2_layout():
    f = Quadratic3Scalar2()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([1.5, 1.0, 0.0])
    assert_almost_equal(f.value(x1, x2), 936.0)
    assert_almost_equal(f.gradient(x1, x2), [-936.0, 624.0])


def test_quadratic4scalar2_layout():
    f = Quadratic4Scalar2()
    x1 = array([[1.5, 1.0, 0.0],
                [1.5, 5.0, 0.0]])
    x2 = array([[-1.5, 1.0, 0.0],
                [1.5, 5.0, 0.0]])
    assert_almost_equal(f.value(x1, x2), 3629.9906336)
    assert_almost_equal(f.gradient(x1, x2), [-8167.4789255926462,
                                             5444.9859503950975])


def test_vectorvalued1scalar1_layout():
    f = VectorValued1Scalar1()
    x = array([1.5, 1.0, 0.0])
    assert_almost_equal(f.value(x), [12., 8., 0.])
    assert_almost_equal(f.gradient(x), [array([-6., -4., -0.])])


def test_vectorvalued2scalar1_layout():
    f = VectorValued2Scalar1()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([0.0, -3.0, 1.0])
    assert_almost_equal(f.value(x1, x2), [0., - 24., 0.])
    assert_almost_equal(f.gradient(x1, x2), [array([-0., 12., -0.])])


def test_vectorvalued1scalar2_layout():
    f = VectorValued1Scalar2()
    x = array([1.5, 1.0, 0.0])
    assert_almost_equal(f.value(x), [432., 288., 0.])
    assert_almost_equal(f.gradient(x), [array(
        [-432., -288., -0.]), array([288., 192., 0.])])


def test_vectorvalued2scalar2_layout():
    f = VectorValued2Scalar2()
    x1 = array([1.5, 1.0, 0.0])
    x2 = array([0.0, -3.0, 1.0])
    assert_almost_equal(f.value(x1, x2), [12., -46., 18.])
    assert_almost_equal(f.gradient(x1, x2), [array(
        [-12.,  -8.,  -0.]), array([0., -36.,  12.])])

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
