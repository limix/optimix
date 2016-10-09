import collections


class FuncDataWrapper(object):

    def __init__(self, target, data):
        self._target = target
        self.raw = data

    def value(self):
        return self._target.value(*self.raw)

    def gradient(self):
        return self._target.gradient(*self.raw)


class FuncData(object):

    def __init__(self):
        self.__data = dict()

    def data(self, purpose='learn'):
        return FuncDataWrapper(self, self.__data[purpose])

    def set_data(self, data, purpose='learn'):
        if not isinstance(data, collections.Sequence):
            data = (data,)
        self.__data[purpose] = data

    def unset_data(self, purpose='learn'):
        del self.__data[purpose]


class FuncDataReduceWrapper(object):

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


class FuncDataReduce(object):

    def __init__(self, functions):
        self.__functions = functions

    def data(self, *args, **kwargs):
        fs = [f.data(*args, **kwargs) for f in self.__functions]
        return FuncDataReduceWrapper(self, fs)
