from numpy.linalg import norm

from optimix import Function, Scalar, Vector, minimize


class Quadratic(Function):

    def __init__(self):
        super(Quadratic, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x):
        a = self.get('a')
        b = self.get('b')
        return ((a - 5.0)**2 + (b + 5.0)**2 * x) / 2.0

    def derivative_a(self, x):
        a = self.get('a')
        return (a - 5.0)

    def derivative_b(self, x):
        b = self.get('b')
        return (b + 5.0) * x

f = Quadratic()
x = 1.2
f.set_data(x)
minimize(f)

print("Optimum found: (%g, %g)" % (f.get('a'), f.get('b')))
