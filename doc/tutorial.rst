Examples
--------

A function in :py:mod:`optimix` sense is a map of input values and variable values
to output values. Those values can be scalars, vectors and/or matrices but what
is most important is that input values and variable values are not treated in
the same way. Derivatives are always over variables (not inputs) and thus the
optimization is always performed over variables. The reason for this is that
input values are meant to be datasets and variable values are meant to be
model parameters. I hope the next examples help clarify this.

Single input
^^^^^^^^^^^^

We create a class that inherits from :py:class:`optimix.Function`, define a
scalar variable (which we named here as `scale`), and implement the
:py:meth:`value` and :py:meth:`derivative_scale` methods. Note that the
suffix of :meth:`derivative_scale` is due to the name we chosen for the
variable.

.. literalinclude:: ../examples/minimal_single_scalar1.py

The output should be similar to:

.. program-output:: python ../examples/minimal_single_scalar1.py

And an example for two variables:

.. literalinclude:: ../examples/minimal_single_scalar2.py

The output should be similar to:

.. program-output:: python ../examples/minimal_single_scalar2.py

Double inputs
^^^^^^^^^^^^^

You can also define a function of two inputs (or more) in a very natural way:

.. literalinclude:: ../examples/minimal_single_scalar2.py

The output should be similar to:

.. program-output:: python ../examples/minimal_single_scalar2.py
