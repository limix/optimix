from optimix import Function
from optimix import Scalar
from optimix import minimize


class Quadratic(Function):

    def __init__(self):
        self._scale = Scalar(1.0)
        super(Quadratic, self).__init__(scale=self._scale)

    @property
    def scale(self):
        return self._scale.value

    @scale.setter
    def scale(self, scale):
        self._scale.value = scale

    def value(self, x):
        return (self.scale - 5.0)**2 * x / 2.0

    def derivative_scale(self, x):
        return (self.scale - 5.0) * x


def test_function_optimization():
    f = Quadratic()
    f.set_data(2.3)
    minimize(f)
    print(f.scale)


if __name__ == '__main__':
    test_function_optimization()
