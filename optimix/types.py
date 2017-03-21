r"""
*****
Types
*****

Introduction
^^^^^^^^^^^^

We have two variables types: :class:`optimix.types.Scalar` and
:class:`optimix.types.Vector`.

Public interface
^^^^^^^^^^^^^^^^
"""
from __future__ import unicode_literals

# pylint: disable=E0611
from numpy import array, asarray, atleast_1d, float64

from ndarray_listener import ndarray_listener


class Scalar(object):
    r"""Scalar variable type.

    It holds a ``float64`` value, listen to changes, and fix or
    unfix its value.

    Args:
        value (float): initial value.
    """
    __slots__ = [
        'raw', '_listeners', '_fixed', 'value', '__array_interface__',
        '__array_struct__'
    ]

    def __init__(self, value):
        self._listeners = []
        self._fixed = False
        value = ndarray_listener(float64(value))
        self.raw = value
        self.__array_interface__ = value.__array_interface__
        self.__array_struct__ = value.__array_struct__

    def copy(self):
        """Return a copy."""
        return Scalar(self.raw)

    @property
    def shape(self):
        """Shape according to :mod:`numpy`."""
        return self.raw.shape

    @property
    def size(self):
        """Size according to :mod:`numpy`."""
        return self.raw.size

    def asarray(self):
        """Return a :class:`numpy.ndarray` representation."""
        return array(self.raw)

    @property
    def isfixed(self):
        """Return whether it is fixed or not."""
        return self._fixed

    def fix(self):
        """Set it fixed."""
        self._fixed = True

    def unfix(self):
        """Set it unfixed."""
        self._fixed = False

    def listen(self, you):
        """Request a callback for value modification.

        Args:
            you (object): an instance having ``__call__`` attribute.
        """
        self._listeners.append(you)

    def __setattr__(self, name, value):
        if name == 'value':
            try:
                value = ndarray_listener(float64(value))
            except TypeError:
                value = value[0]
            Scalar.__dict__['raw'].__set__(self, value)
            t = Scalar.__dict__['__array_interface__']
            t.__set__(self, value.__array_interface__)
            t = Scalar.__dict__['__array_struct__']
            t.__set__(self, value.__array_struct__)
            self._notify()
        else:
            Scalar.__dict__[name].__set__(self, value)

    def __getattr__(self, name):
        if name == 'value':
            name = 'raw'
        return Scalar.__dict__[name].__get__(self)

    def _notify(self):
        for l in self._listeners:
            l(self.value)

    def __str__(self):
        return 'Scalar(' + str(self.raw) + ')'

    def __repr__(self):
        return repr(self.raw)

    def __ge__(self, that):
        return self.raw >= that.raw

    def __gt__(self, that):
        return self.raw > that.raw

    def __le__(self, that):
        return self.raw <= that.raw

    def __lt__(self, that):
        return self.raw < that.raw

    def __eq__(self, that):
        return self.raw == that.raw

    def __ne__(self, that):
        return self.raw != that.raw


class Vector(object):
    r"""Vector variable type.

    It holds an array of ``float64`` values, listen to changes, and fix or
    unfix its values.

    Args:
        value (float): initial value.
    """
    __slots__ = [
        'raw', '_listeners', '_fixed', '__array_interface__',
        '__array_struct__', 'value'
    ]

    def __init__(self, value):
        self._listeners = []
        self._fixed = False
        value = asarray(value)
        value = atleast_1d(value).ravel()
        self.raw = value
        self.__array_interface__ = value.__array_interface__
        self.__array_struct__ = value.__array_struct__

    def copy(self):
        """Return a copy."""
        return Vector(self.raw)

    @property
    def shape(self):
        """Shape according to :mod:`numpy`."""
        return self.raw.shape

    @property
    def size(self):
        """Size according to :mod:`numpy`."""
        return self.raw.size

    def asarray(self):
        """Return a :class:`numpy.ndarray` representation."""
        return array(self.raw)

    @property
    def isfixed(self):
        """Return whether it is fixed or not."""
        return self._fixed

    def fix(self):
        """Set it fixed."""
        self._fixed = True

    def unfix(self):
        """Set it unfixed."""
        self._fixed = False

    def listen(self, you):
        """Request a callback for value modification.

        Args:
            you (object): an instance having ``__call__`` attribute.
        """
        self._listeners.append(you)

    def __setattr__(self, name, value):
        if name == 'value':
            value = asarray(value)
            value = atleast_1d(value).ravel()

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

    def _notify(self):
        for l in self._listeners:
            l(self.asarray())

    def __str__(self):
        return 'Vector(' + str(self.raw) + ')'

    def __repr__(self):
        return repr(self.raw)

    def __ge__(self, that):
        return self.raw >= that.raw

    def __gt__(self, that):
        return self.raw > that.raw

    def __le__(self, that):
        return self.raw <= that.raw

    def __lt__(self, that):
        return self.raw < that.raw

    def __eq__(self, that):
        return self.raw == that.raw

    def __ne__(self, that):
        return self.raw != that.raw


class Matrix(object):
    __slots__ = ['raw', '_listeners', '_fixed']

    def __init__(self, value):
        self._listeners = []
        self._fixed = False
        self.raw = value

    @property
    def size(self):
        return self.raw.size

    def asarray(self):
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
