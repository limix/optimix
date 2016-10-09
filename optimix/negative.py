class NegativeFunction(object):

    def __init__(self, function):
        self._function = function

    def value(self):
        return - self._function.value()

    def gradient(self):
        g = self._function.gradient()
        return [-gi for gi in g]

    def variables(self):
        return self._function.variables()


def negative_function(function):
    return NegativeFunction(function)
