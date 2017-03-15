from numpy import add
from optimix import FunctionReduce


class QuadraticScalarReduce(FunctionReduce):
    def __init__(self, functions):
        super(QuadraticScalarReduce, self).__init__(functions, 'sum')

    def value_reduce(self, values): # pylint: disable=R0201
        return add.reduce(values)

    def derivative_reduce(self, derivatives): # pylint: disable=R0201
        return add.reduce(derivatives)
