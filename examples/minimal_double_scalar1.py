from numpy.linalg import norm

from optimix import Function, Scalar, Vector, minimize


class Quadratic(Function):

    def __init__(self):
        super(Quadratic, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x0, x1):
        a = self.get('a')
        b = self.get('b')
        return ((a - 5.0)**2 * x0 + (b + 5.0)**2 * x1) / 2.0

    def derivative_a(self, x0, x1):
        a = self.get('a')
        return 2 * (a - 5.0) * x0

    def derivative_b(self, x0, x1):
        b = self.get('b')
        return 2 * (b + 5.0) * x1

f = Quadratic()
x0 = 2.3
x1 = 1.0
f.set_data((x0, x1))
minimize(f)

print("Optimum found: (%g, %g)" % (f.get('a'), f.get('b')))
