from numpy.linalg import norm

from optimix import Function, Scalar, Vector, minimize


class Quadratic(Function):

    def __init__(self):
        super(Quadratic, self).__init__(scale=Scalar(1.0))

    def value(self, x):
        s = self.get('scale')
        return (s - 5.0)**2 * x / 2.0

    def derivative_scale(self, x):
        s = self.get('scale')
        return (s - 5.0) * x

f = Quadratic()
x = 1.2

print("Function evaluation at x: %g" % f.value(x))
print("Function gradient at x: %s" % f.gradient(x))

# For optimizating the function, we need a dataset
# associated with it. This is accomplished by calling
# the set_data method as follows:
f.set_data(x)

# We are now ready to minimize the function.
minimize(f)

# This will print the optimum found.
print("Optimum found: %g" % f.get('scale'))
