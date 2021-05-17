__all__ = ["Scalar", "Vector", "Matrix"]


class Scalar(object):
    """
    Scalar variable type.

    It holds a 64-bits floating point value, stored via a zero-dimensional
    ``ndl``, listen to changes, and fix or unfix its value.

    Parameters
    ----------
    value : float
        Initial value.
    """

    __slots__ = [
        "raw",
        "_fixed",
        "value",
        "__array_interface__",
        "__array_struct__",
        "_bounds",
    ]

    def __init__(self, value):
        from ndarray_listener import ndl
        from numpy import float64, inf

        self._bounds = (-inf, +inf)
        self._fixed = False
        value = ndl(float64(value))
        self.raw = value
        self.__array_interface__ = value.__array_interface__
        self.__array_struct__ = value.__array_struct__

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, v):
        self._bounds = v

    def copy(self):
        """Return a copy."""
        return Scalar(self.raw)

    @property
    def shape(self):
        """
        Shape according to :mod:`numpy`.
        """
        return self.raw.shape

    @property
    def ndim(_):
        """
        Number of dimensions.
        """
        return 0

    @property
    def size(self):
        """
        Size according to :mod:`numpy`.
        """
        return self.raw.size

    def asarray(self):
        """
        Return a :class:`numpy.ndarray` representation.
        """
        from numpy import array

        return array(self.raw)

    @property
    def isfixed(self):
        """
        Return whether it is fixed or not.
        """
        return self._fixed

    def fix(self):
        """
        Set it fixed.
        """
        self._fixed = True

    def unfix(self):
        """
        Set it unfixed.
        """
        self._fixed = False

    def listen(self, you):
        """
        Request a callback for value modification.

        Parameters
        ----------
        you : object
            An instance having ``__call__`` attribute.
        """
        self.raw.talk_to(you)

    def __setattr__(self, name, value):
        from numpy import float64

        if name == "value":
            try:
                value = float64(value)
            except TypeError:
                value = value[0]
            self.raw.itemset(value)
        else:
            Scalar.__dict__[name].__set__(self, value)

    def __getattr__(self, name):
        if name == "value":
            name = "raw"
        return Scalar.__dict__[name].__get__(self)

    def __str__(self):
        return "Scalar(" + str(self.raw) + ")"

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
    """
    Vector variable type.

    It holds an array of 64-bits floating point values, via an one-dimensional
    ``ndl``, listen to changes, and fix or unfix its values.

    Parameters
    ----------
    value : float
        Initial value.
    """

    __slots__ = [
        "raw",
        "_fixed",
        "__array_interface__",
        "__array_struct__",
        "value",
        "_bounds",
    ]

    def __init__(self, value):
        from numpy import asarray, atleast_1d, inf
        from ndarray_listener import ndl

        self._bounds = [(-inf, +inf)] * len(value)
        self._fixed = False
        value = asarray(value, float)
        value = ndl(atleast_1d(value).ravel())
        self.raw = value
        self.__array_interface__ = value.__array_interface__
        self.__array_struct__ = value.__array_struct__

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, v):
        self._bounds = v

    def copy(self):
        """
        Return a copy.
        """
        return Vector(self.raw)

    @property
    def shape(self):
        """
        Shape according to :mod:`numpy`.
        """
        return self.raw.shape

    @property
    def ndim(self):
        """
        Number of dimensions.
        """
        return len(self.shape)

    @property
    def size(self):
        """
        Size according to :mod:`numpy`.
        """
        return self.raw.size

    def asarray(self):
        """
        Return a :class:`numpy.ndarray` representation.
        """
        from numpy import array

        return array(self.raw)

    @property
    def isfixed(self):
        """
        Return whether it is fixed or not.
        """
        return self._fixed

    def fix(self):
        """
        Set it fixed.
        """
        self._fixed = True

    def unfix(self):
        """
        Set it unfixed.
        """
        self._fixed = False

    def listen(self, you):
        """
        Request a callback for value modification.

        Parameters
        ----------
        you : object
            An instance having ``__call__`` attribute.
        """
        self.raw.talk_to(you)

    def __setattr__(self, name, value):
        from numpy import asarray, atleast_1d

        if name == "value":
            value = asarray(value)
            value = atleast_1d(value).ravel()
            self.raw[:] = value
        else:
            Vector.__dict__[name].__set__(self, value)

    def __getattr__(self, name):
        if name == "value":
            name = "raw"
        return Vector.__dict__[name].__get__(self)

    def __str__(self):
        return "Vector(" + str(self.raw) + ")"

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
    __slots__ = ["raw", "_fixed"]

    def __init__(self, value):
        self._fixed = False
        self.raw = value

    @property
    def size(self):
        return self.raw.size

    def asarray(self):
        from numpy import asarray

        return asarray([self.raw])

    @property
    def isfixed(self):
        return self._fixed

    def fix(self):
        self._fixed = True

    def unfix(self):
        self._fixed = False

    def __setattr__(self, name, value):
        if name == "value":
            Matrix.__dict__["raw"].__set__(self, value)
        else:
            Matrix.__dict__[name].__set__(self, value)

    def __getattr__(self, name):
        if name == "value":
            name = "raw"
        return Matrix.__dict__[name].__get__(self)

    def listen(self, you):
        self.raw.talk_to(you)

    def __str__(self):
        return "Matrix(" + str(self.raw) + ")"

    def __repr__(self):
        return repr(self.raw)
