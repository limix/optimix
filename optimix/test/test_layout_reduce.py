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


from quadratic_function_reduces import QuadraticScalarReduce


def test_quadratic1scalar1_layout():
    f1 = Quadratic1Scalar1()
    f2 = Quadratic1Scalar1()
    f = QuadraticScalarReduce([f1, f2])
    x1 = 1.2
    x2 = -1.1
    f1.set_data(x1)
    f2.set_data(x2)

    f = f.feed()
    assert_almost_equal(f.value(), 0.8)
    assert_almost_equal(f.gradient(), [-4.8, 4.4])


def test_vectorvalued1scalar1_layout():
    f1 = VectorValued1Scalar1()
    x1 = array([1.5, 1.0, 0.0])
    f2 = VectorValued1Scalar1()
    x2 = array([1.5, -1.0, 0.0])

    f1.set_data(x1)
    f2.set_data(x2)
    f = QuadraticScalarReduce([f1, f2])

    f = f.feed()
    assert_almost_equal(f.value(), [24., 0., 0.])
    assert_almost_equal(f.gradient(), [array([-6., -4., -0.]),
                                       array([-6.,  4., -0.])])

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
