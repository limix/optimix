***********
Quick start
***********

A function in :py:mod:`optimix` sense is a map of input and variable values to
output values.
Those values can be scalars or vectors but what is most important
is that input values and variable values are not treated in the same way.
Derivatives are always over variables (not inputs) and thus the optimisation
is always performed over variables.
The reason for this is that input values are meant to be datasets and variable
values are meant to be model parameters.
I hope the next examples help clarify this matter.

Single input
^^^^^^^^^^^^

We create a class that inherits from :py:class:`optimix.function.Function`,
define a scalar variable (which we named here as `scale`), and implement the
:py:meth:`optimix.function.Function.value` and
:py:meth:`optimix.function.Function.gradient` methods.

.. doctest::

  >>> from optimix import Function, Scalar, minimize
  >>>
  >>> class Quadratic(Function):
  ...
  ...   def __init__(self):
  ...     super(Quadratic, self).__init__(scale=Scalar(1.0))
  ...
  ...   def value(self, x):
  ...     s = self.variables().get('scale').value
  ...     return (s - 5.0)**2 * x / 2.0
  ...
  ...   def gradient(self, x):
  ...     s = self.variables().get('scale').value
  ...     return dict(scale=(s - 5.0) * x)
  >>>
  >>> f = Quadratic()
  >>> x = 1.2
  >>>
  >>> print("Function evaluation at x: %g" % f.value(x))
  Function evaluation at x: 9.6
  >>> print("Function gradient at x: %s" % f.gradient(x))
  Function gradient at x: {'scale': ndarray_listener(-4.8)}
  >>>
  >>> # For optimizating the function, we need a dataset
  >>> # associated with it. This is accomplished by calling
  >>> # the set_data method as follows:
  >>> f.set_data(x)
  >>>
  >>> func = f.feed()
  >>>
  >>> # We are now ready to minimize the function.
  >>> minimize(func, verbose=False)
  >>>
  >>> # This will print the optimum found.
  >>> print("Optimum found: %g" % f.variables().get('scale').value)
  Optimum found: 5

And an example for two variables:

.. doctest::

  >>> from optimix import Function, Scalar, minimize
  >>>
  >>> class Quadratic(Function):
  ...
  ...   def __init__(self):
  ...     super(Quadratic, self).__init__(a=Scalar(1.0), b=Scalar(1.0))
  ...
  ...   def value(self, x):
  ...     a = self.variables().get('a').value
  ...     b = self.variables().get('b').value
  ...     return ((a - 5.0)**2 + (b + 5.0)**2 * x) / 2.0
  ...
  ...   def gradient(self, x):
  ...     a = self.variables().get('a').value
  ...     b = self.variables().get('b').value
  ...     return dict(a=(a - 5.0), b=(b + 5.0) * x)
  >>>
  >>> f = Quadratic()
  >>> x = 1.2
  >>> f.set_data(x)
  >>> minimize(f.feed(), verbose=False)
  >>> a = f.variables().get('a').value
  >>> b = f.variables().get('b').value
  >>> print("Optimum found: (%g, %g)" % (a, b))
  Optimum found: (5, -5)

Double inputs
^^^^^^^^^^^^^

You can also define a function of two inputs (or more) in a very natural way:

.. doctest::

  >>> from optimix import Function, Scalar, minimize
  >>>
  >>> class Quadratic(Function):
  ...
  ...   def __init__(self):
  ...     super(Quadratic, self).__init__(a=Scalar(1.0), b=Scalar(1.0))
  ...
  ...   def value(self, x0, x1):
  ...     a = self.variables().get('a').value
  ...     b = self.variables().get('b').value
  ...     return ((a - 5.0)**2 * x0 + (b + 5.0)**2 * x1) / 2.0
  ...
  ...   def gradient(self, x0, x1):
  ...     a = self.variables().get('a').value
  ...     b = self.variables().get('b').value
  ...     return dict(a=2 * (a - 5.0) * x0, b=2 * (b + 5.0) * x1)
  >>>
  >>> f = Quadratic()
  >>> x0 = 2.3
  >>> x1 = 1.0
  >>> f.set_data((x0, x1))
  >>> minimize(f.feed(), verbose=False)
  >>>
  >>> a = f.variables().get('a').value
  >>> b = f.variables().get('b').value
  >>> print("Optimum found: (%g, %g)" % (a, b))
  Optimum found: (5, -5)
