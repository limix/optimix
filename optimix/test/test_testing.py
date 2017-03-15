from numpy import array
from numpy.testing import assert_almost_equal

from quadratic_functions import Quadratic2Scalar1
from vector_valued_functions import VectorValued1Scalar2
from optimix.testing import Assertion

def test_testing():
    x0 = array([1.5, 1.0, 0.0])
    x1 = array([-1.5, 0.1, 2.0])
    value0 = array([[432., 288., 0.]])

    # f = Quadratic2Scalar1()
    x0 = 2.3
    x1 = 1.0
    value = 18.4
    gradient = [-9.2]
    # assert_almost_equal(f.value(x1, x2), 18.4)
    # assert_almost_equal(f.gradient(x1, x2), [-9.2])

    # assert_almost_equal(f.value(x), [432., 288., 0.])
    # assert_almost_equal(f.gradient(x), [array(
    #     [-432., -288., -0.]), array([288., 192., 0.])])

    a = Assertion(lambda: Quadratic2Scalar1(), x0, x1, value, scale=1.0)
    a.assert_layout()
