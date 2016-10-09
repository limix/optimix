from .variables import Variables
from .variables import merge_variables


class Learnable(object):

    def __init__(self, variables):
        assert isinstance(variables, Variables)
        self.__variables = variables

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


class LearnableReduce(object):

    def __init__(self, learnables, prefix='noname'):
        self.__learnables = learnables
        self.__prefix = prefix

    def gradient(self, *args, **kwargs):
        grad = []
        for l in self.__learnables:
            grad += l.gradient(*args, **kwargs)
        return grad

    def variables(self):
        vars_list = [l.variables() for l in self.__learnables]
        vd = dict()
        for (i, vs) in enumerate(vars_list):
            vd['%s[%d]' % (self.__prefix, i)] = vs
        return merge_variables(vd)
