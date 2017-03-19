from __future__ import division

from numpy.testing import assert_allclose

from optimix import Function, Scalar

class Quadratic1Scalar1(Function):
    def __init__(self):
        super(Quadratic1Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        s = self.get('scale')
        return (s - 5.0)**2 * x / 2.0

    def gradient(self, x):
        s = self.get('scale')
        return dict(scale=(s - 5.0) * x)

    # def derivative_scale(self, x):
    #     s = self.get('scale')
    #     return (s - 5.0) * x

def test_function_quadratic1scalar1():
    f = Quadratic1Scalar1()
    f.set_data(1.2)
    f.feed().minimize(progress=False)
    assert_allclose(f.get('scale'), 5.0)
    f.set('scale', 1.0)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
