from numpy import zeros
from numpy import array
from numpy.testing import assert_almost_equal

from optimix import Function
from optimix import Scalar
from optimix import Vector
from optimix import minimize

from funcs_testing import Quadratic1Scalar1
from funcs_testing import Quadratic2Scalar1
from funcs_testing import Quadratic3Scalar1
from funcs_testing import Quadratic4Scalar1


# def test_quadratic1_minimize():
#     f = Quadratic1()
#     f.set_data(2.3)
#     minimize(f)
#     assert_almost_equal(f.scale, 5)
#
#
# def test_quadratic2_minimize():
#     f = Quadratic2()
#     f.set_data((2.3, 1.0))
#     minimize(f)
#     assert_almost_equal(f.scale, 5)
#
#
# def test_quadratic3_minimize():
#     f = Quadratic3()
#     x0 = array([1.5, 1.0, 0.0])
#     x1 = array([1.5, 1.0, 0.0])
#     f.set_data((x0, x1))
#     minimize(f)
#     assert_almost_equal(f.scale, 5)
