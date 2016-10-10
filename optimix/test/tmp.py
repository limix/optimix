from optimix import Function
from optimix import Scalar
from optimix import Vector


class Quadratic1Scalar1(Function):

    def __init__(self):
        super(Quadratic1Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        return (self.scale - 5.0)**2 * x / 2.0

    def derivative_scale(self, x):
        return (self.scale - 5.0) * x

if __name__ == '__main__':
    f = Quadratic1Scalar1()
    print(f.get('scale'))
    f.set('scale', 2)
    print(f.get('scale'))
