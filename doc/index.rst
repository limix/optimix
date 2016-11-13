Optimix's documentation
=======================

You can get the source and open issues `on Github.`_

Install
-------

The recommended way of installing it is via `conda`_::

  conda install -c conda-forge optimix

An alternative way would be via pip::

  pip install optimix

Quick start
-----------

A function in :py:mod:`optimix` sense is a map of input values and variable
values to output values.
Those values can be scalars, vectors and/or matrices but what is most important
is that input values and variable values are not treated in the same way.
Derivatives are always over variables (not inputs) and thus the optimization
is always performed over variables.
The reason for this is that input values are meant to be datasets and variable
values are meant to be model parameters. I hope the next examples help clarify
this.

Single input
^^^^^^^^^^^^

We create a class that inherits from :py:class:`optimix.Function`, define a
scalar variable (which we named here as `scale`), and implement the
:py:meth:`value` and :py:meth:`derivative_scale` methods. Note that the
suffix of :meth:`derivative_scale` is due to the name we chosen for the
variable.

.. testcode::

  from optimix import Function, Scalar, minimize

  class Quadratic(Function):

      def __init__(self):
          super(Quadratic, self).__init__(scale=Scalar(1.0))

      def value(self, x):
          s = self.get('scale')
          return (s - 5.0)**2 * x / 2.0

      def derivative_scale(self, x):
          s = self.get('scale')
          return (s - 5.0) * x

  f = Quadratic()
  x = 1.2

  print("Function evaluation at x: %g" % f.value(x))
  print("Function gradient at x: %s" % f.gradient(x))

  # For optimizating the function, we need a dataset
  # associated with it. This is accomplished by calling
  # the set_data method as follows:
  f.set_data(x)

  func = f.feed()

  # We are now ready to minimize the function.
  minimize(func)

  # This will print the optimum found.
  print("Optimum found: %g" % f.get('scale'))

The output should be

.. testoutput::

  Function evaluation at x: 9.6
  Function gradient at x: [-4.8]
  Optimum found: 5

And an example for two variables:

.. testcode::

  from optimix import Function, Scalar, minimize

  class Quadratic(Function):

      def __init__(self):
          super(Quadratic, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

      def value(self, x):
          a = self.get('a')
          b = self.get('b')
          return ((a - 5.0)**2 + (b + 5.0)**2 * x) / 2.0

      def derivative_a(self, _):
          a = self.get('a')
          return (a - 5.0)

      def derivative_b(self, x):
          b = self.get('b')
          return (b + 5.0) * x

  f = Quadratic()
  x = 1.2
  f.set_data(x)
  minimize(f.feed())

  print("Optimum found: (%g, %g)" % (f.get('a'), f.get('b')))

The output should be

.. testoutput::

  Optimum found: (5, -5)

Double inputs
^^^^^^^^^^^^^

You can also define a function of two inputs (or more) in a very natural way:

.. testcode::

  from optimix import Function, Scalar, minimize

  class Quadratic(Function):

      def __init__(self):
          super(Quadratic, self).__init__(a=Scalar(1.0), b=Scalar(1.0))

      def value(self, x0, x1):
          a = self.get('a')
          b = self.get('b')
          return ((a - 5.0)**2 * x0 + (b + 5.0)**2 * x1) / 2.0

      def derivative_a(self, x0, _):
          a = self.get('a')
          return 2 * (a - 5.0) * x0

      def derivative_b(self, _, x1):
          b = self.get('b')
          return 2 * (b + 5.0) * x1

  f = Quadratic()
  x0 = 2.3
  x1 = 1.0
  f.set_data((x0, x1))
  minimize(f.feed())

  print("Optimum found: (%g, %g)" % (f.get('a'), f.get('b')))

The output should be

.. testoutput::

  Optimum found: (5, -5)

.. _on Github.: https://github.com/Horta/optimix
.. _conda: http://conda.pydata.org/docs/index.html
