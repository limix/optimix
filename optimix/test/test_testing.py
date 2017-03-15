from optimix.testing import Assertion

from quadratic_functions import Quadratic2Scalar1


def test_testing():
    x0 = 2.3
    x1 = 1.0
    value = 18.4

    a = Assertion(Quadratic2Scalar1, x0, x1, value, scale=1.0)
    a.assert_layout()
    a.assert_gradient()
