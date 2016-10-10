import collections
from six import create_bound_method
from six import string_types

from .variables import Variables
from .variables import merge_variables


def variable_getter(self):
    print("Hello, i am variable_getter")
    return 1


class Function(object):

    def __init__(self, **kwargs):
        self.__variables = Variables(kwargs)
        self._data = dict()

    def get(self, name):
        return self.__variables.get(name).value

    def set(self, name, value):
        self.__variables.get(name).value = value

    def gradient(self, *args, **kwargs):
        names = sorted(self.__variables.names())
        grad = []
        for name in names:
            g = getattr(self, 'derivative_' + name)(*args, **kwargs)
            grad.append(g)
        return grad

    def fix(self, var_name):
        self.__variables[var_name].fix()

    def unfix(self, var_name):
        self.__variables[var_name].unfix()

    def isfixed(self, var_name):
        return self.__variables[var_name].isfixed()

    def variables(self):
        return self.__variables

    def set_data(self, data, purpose='learn'):
        assert isinstance(purpose, string_types)
        if not isinstance(data, collections.Sequence):
            data = (data,)
        self._data[purpose] = data

    def unset_data(self, purpose='learn'):
        del self._data[purpose]


class FunctionReduce(object):

    def __init__(self, functions, prefix='noname'):
        self.__functions = functions
        self.__prefix = prefix

    def gradient(self, *args, **kwargs):
        grad = []
        for l in self.__functions:
            grad += l.gradient(*args, **kwargs)
        return grad

    def variables(self):
        vars_list = [l.variables() for l in self.__functions]
        vd = dict()
        for (i, vs) in enumerate(vars_list):
            vd['%s[%d]' % (self.__prefix, i)] = vs
        return merge_variables(vd)


class FunctionDataFeed(object):

    def __init__(self, target, data):
        self._target = target
        self.raw = data

    def value(self):
        return self._target.value(*self.raw)

    def gradient(self):
        return self._target.gradient(*self.raw)

    def variables(self):
        return self._target.variables()


class FunctionReduceDataFeed(object):

    def __init__(self, target, functions):
        self._target = target
        self._functions = functions

    def value(self):
        return self._target.value_reduce([f.value() for f in self._functions])

    def gradient(self):
        grad = []
        for f in self._functions:
            grad += f.gradient()
        return grad
