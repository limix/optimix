from __future__ import unicode_literals

from numpy import array
from numpy import asarray
from numpy import atleast_1d

from ndarray_listener import ndarray_listener


class Scalar(object):
    __slots__ = ['raw', '_listeners', '_fixed', 'value']

    def __init__(self, value):
        self._listeners = []
        self._fixed = False
        self.raw = value

    def copy(self):
        return Scalar(self.raw)

    @property
    def size(self):
        return 1

    def to_ndarray(self):
        return array([self.raw])

    @property
    def isfixed(self):
        return self._fixed

    def fix(self):
        self._fixed = True

    def unfix(self):
        self._fixed = False

    def __setattr__(self, name, value):
        if name == 'value':
            try:
                v = float(value)
            except TypeError:
                v = value[0]
            Scalar.__dict__['raw'].__set__(self, v)
            self._notify()
        else:
            Scalar.__dict__[name].__set__(self, value)

    def __getattr__(self, name):
        if name == 'value':
            name = 'raw'
        return Scalar.__dict__[name].__get__(self)

    def listen(self, you):
        self._listeners.append(you)

    def _notify(self):
        for l in self._listeners:
            l(self.value)

    def __str__(self):
        return 'Scalar(' + str(self.raw) + ')'

    def __repr__(self):
        return repr(self.raw)


class Vector(object):

    __slots__ = ['raw', '_listeners', '_fixed', '__array_interface__',
                 '__array_struct__']

    def __init__(self, value):
        self._listeners = []
        self._fixed = False
        value = asarray(value)
        value = atleast_1d(value).ravel()
        self.raw = value
        self.__array_interface__ = value.__array_interface__
        self.__array_struct__ = value.__array_struct__

    @property
    def size(self):
        return self.raw.size

    def to_ndarray(self):
        return array([self.raw])

    @property
    def isfixed(self):
        return self._fixed

    def fix(self):
        self._fixed = True

    def unfix(self):
        self._fixed = False

    @property
    def shape(self):
        return self.raw.shape

    def __setattr__(self, name, value):
        if name == 'value':
            if not hasattr(value, "__array_interface__"):
                raise TypeError(("'%s'" % type(value)) +
                                " object has no attribute" +
                                " '__array_interface__'")
            Vector.__dict__['raw'].__set__(self, value)
            t = Vector.__dict__['__array_interface__']
            t.__set__(self, value.__array_interface__)
            t = Vector.__dict__['__array_struct__']
            t.__set__(self, value.__array_struct__)
            self._notify()
        else:
            Vector.__dict__[name].__set__(self, value)

    def __getattr__(self, name):
        if name == 'value':
            v = ndarray_listener(Vector.__dict__['raw'].__get__(self))
            for l in self._listeners:
                v.talk_to(l)
            return v
        return Vector.__dict__[name].__get__(self)

    def listen(self, you):
        self._listeners.append(you)

    def _notify(self):
        for l in self._listeners:
            l(self.value)

    def __str__(self):
        return 'Vector(' + str(self.raw) + ')'

    def __repr__(self):
        return repr(self.raw)


class Matrix(object):
    __slots__ = ['raw', '_listeners', '_fixed']

    def __init__(self, value):
        self._listeners = []
        self._fixed = False
        self.raw = value

    @property
    def size(self):
        return self.raw.size

    def to_ndarray(self):
        return asarray([self.raw])

    @property
    def isfixed(self):
        return self._fixed

    def fix(self):
        self._fixed = True

    def unfix(self):
        self._fixed = False

    def __setattr__(self, name, value):
        if name == 'value':
            Matrix.__dict__['raw'].__set__(self, value)
            self._notify()
        else:
            Matrix.__dict__[name].__set__(self, value)

    def __getattr__(self, name):
        if name == 'value':
            name = 'raw'
        return Matrix.__dict__[name].__get__(self)

    def listen(self, you):
        self._listeners.append(you)

    def _notify(self):
        for l in self._listeners:
            l(self.value)

    def __str__(self):
        return 'Matrix(' + str(self.raw) + ')'

    def __repr__(self):
        return repr(self.raw)
