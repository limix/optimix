# from numpy import array
# from numpy.testing import assert_almost_equal
#
# from quadratic_functions import (Quadratic1Scalar1, Quadratic1Scalar2,
#                                  Quadratic2Scalar1, Quadratic2Scalar2,
#                                  Quadratic3Scalar1, Quadratic3Scalar2,
#                                  Quadratic4Scalar1, Quadratic4Scalar2)
# from vector_valued_functions import (VectorValued1Scalar1,
#                                      VectorValued1Scalar2,
#                                      VectorValued2Scalar1,
#                                      VectorValued2Scalar2)
#
#
# def test_layout_quadratic1scalar1_layout():
#     f = Quadratic1Scalar1()
#     x = 1.2
#     assert_almost_equal(f.value(x), 9.6)
#     assert_almost_equal(f.gradient(x)['scale'], [-4.8])
#
#
# def test_layout_quadratic2scalar1_layout():
#     f = Quadratic2Scalar1()
#     x1 = 2.3
#     x2 = 1.0
#     assert_almost_equal(f.value(x1, x2), 18.4)
#     assert_almost_equal(f.gradient(x1, x2)['scale'], [-9.2])
#
#
# def test_layout_quadratic3scalar1_layout():
#     f = Quadratic3Scalar1()
#     x1 = array([1.5, 1.0, 0.0])
#     x2 = array([1.5, 1.0, 0.0])
#     assert_almost_equal(f.value(x1, x2), 26)
#     assert_almost_equal(f.gradient(x1, x2)['scale'], [-13.0])
#
#
# def test_layout_quadratic4scalar1_layout():
#     f = Quadratic4Scalar1()
#     x1 = array([[1.5, 1.0, 0.0], [1.5, 5.0, 0.0]])
#     x2 = array([[-1.5, 1.0, 0.0], [1.5, 5.0, 0.0]])
#     assert_almost_equal(f.value(x1, x2), 226.8744146)
#     assert_almost_equal(f.gradient(x1, x2)['scale'], [-113.43720729989786])
#
#
# def test_layout_quadratic1scalar2_layout():
#     f = Quadratic1Scalar2()
#     x = 1.2
#     assert_almost_equal(f.value(x), 345.6)
#     assert_almost_equal(f.gradient(x)['a'], [-345.59999999999997])
#     assert_almost_equal(f.gradient(x)['b'], [230.39999999999998])
#
#
# def test_layout_quadratic2scalar2_layout():
#     f = Quadratic2Scalar2()
#     x1 = 2.3
#     x2 = 1.0
#     assert_almost_equal(f.value(x1, x2), 36.4)
#     assert_almost_equal(f.gradient(x1, x2)['a'], [-18.4])
#     assert_almost_equal(f.gradient(x1, x2)['b'], [12.0])
#
#
# def test_layout_quadratic3scalar2_layout():
#     f = Quadratic3Scalar2()
#     x1 = array([1.5, 1.0, 0.0])
#     x2 = array([1.5, 1.0, 0.0])
#     assert_almost_equal(f.value(x1, x2), 936.0)
#     assert_almost_equal(f.gradient(x1, x2)['a'], [-936.0])
#     assert_almost_equal(f.gradient(x1, x2)['b'], [624.0])
#
#
# def test_layout_quadratic4scalar2_layout():
#     f = Quadratic4Scalar2()
#     x1 = array([[1.5, 1.0, 0.0], [1.5, 5.0, 0.0]])
#     x2 = array([[-1.5, 1.0, 0.0], [1.5, 5.0, 0.0]])
#     assert_almost_equal(f.value(x1, x2), 3629.9906336)
#     assert_almost_equal(f.gradient(x1, x2)['a'], [-8167.4789255926462])
#     assert_almost_equal(f.gradient(x1, x2)['b'], [5444.9859503950975])
#
#
# def test_layout_vectorvalued1scalar1_layout():
#     f = VectorValued1Scalar1()
#     x = array([1.5, 1.0, 0.0])
#     assert_almost_equal(f.value(x), [12., 8., 0.])
#     assert_almost_equal(f.gradient(x)['scale'], array([-6., -4., -0.]))
#
#
# def test_layout_vectorvalued2scalar1_layout():
#     f = VectorValued2Scalar1()
#     x1 = array([1.5, 1.0, 0.0])
#     x2 = array([0.0, -3.0, 1.0])
#     assert_almost_equal(f.value(x1, x2), [0., -24., 0.])
#     assert_almost_equal(f.gradient(x1, x2)['scale'], array([-0., 12., -0.]))
#
#
# def test_layout_vectorvalued1scalar2_layout():
#     f = VectorValued1Scalar2()
#     x = array([1.5, 1.0, 0.0])
#     assert_almost_equal(f.value(x), [432., 288., 0.])
#     assert_almost_equal(f.gradient(x)['a'], array([-432., -288., -0.]))
#     assert_almost_equal(f.gradient(x)['b'], array([288., 192., 0.]))
#
#
# def test_layout_vectorvalued2scalar2_layout():
#     f = VectorValued2Scalar2()
#     x1 = array([1.5, 1.0, 0.0])
#     x2 = array([0.0, -3.0, 1.0])
#     assert_almost_equal(f.value(x1, x2), [12., -46., 18.])
#     assert_almost_equal(f.gradient(x1, x2)['a'], array([-12., -8., -0.]))
#     assert_almost_equal(f.gradient(x1, x2)['b'], array([0., -36., 12.]))
#
#
# if __name__ == '__main__':
#     __import__('pytest').main([__file__, '-s'])
