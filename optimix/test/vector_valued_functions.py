from numpy.linalg import norm

from optimix import Function
from optimix import Scalar
from optimix import Vector


class VectorValued1Scalar1(Function):

    def __init__(self):
        super(VectorValued1Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        s = self.get('scale')
        return (s - 5.0)**2 * x / 2.0

    def derivative_scale(self, x):
        s = self.get('scale')
        return (s - 5.0) * x


class VectorValued2Scalar1(Function):

    def __init__(self):
        super(VectorValued2Scalar1, self).__init__(scale=Scalar(1.0))

    def value(self, x0, x1):
        s = self.get('scale')
        return (s - 5.0)**2 * (x0 * x1) / 2.0

    def derivative_scale(self, x0, x1):
        s = self.get('scale')
        return (s - 5.0) * (x0 * x1)


class VectorValued1Scalar2(Function):

    def __init__(self):
        super(VectorValued1Scalar2, self).__init__(
            a=Scalar(1.0), b=Scalar(1.0))

    def value(self, x):
        a = self.get('a')
        b = self.get('b')
        return ((a - 5.0)**2 * (b + 5.0)**2 * x) / 2.0

    def derivative_a(self, x):
        a = self.get('a')
        b = self.get('b')
        return 2 * (a - 5.0) * (b + 5.0)**2 * x

    def derivative_b(self, x):
        a = self.get('a')
        b = self.get('b')
        return 2 * (a - 5.0)**2 * (b + 5.0) * x


class VectorValued2Scalar2(Function):

    def __init__(self):
        super(VectorValued2Scalar2, self).__init__(
            a=Scalar(1.0), b=Scalar(1.0))

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
