# from numpy.testing import assert_allclose
#
# from optimix.variables import Variables
# from optimix import Scalar
#
# def test_variables_set():
#     a = Scalar(1.0)
#     b = a
#     a.value = 2.0
#     assert a is b
#     assert a.raw is b.raw
#
#     v = Variables(dict(a=Scalar(1.0), b=Scalar(1.5)))
#     v.from_named({'a': 0.5})
#     assert_allclose(v.get('a'), 0.5)
#
# def test_variables_str():
#     v = Variables(dict(a=Scalar(1.0), b=Scalar(1.5)))
#     assert v.__str__() == """Variables(a=Scalar(1.0),
#           b=Scalar(1.5))"""
#
# if __name__ == '__main__':
#     __import__('pytest').main([__file__, '-s'])
