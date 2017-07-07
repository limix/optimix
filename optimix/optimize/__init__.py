r"""
*****************************
Function optimization methods
*****************************

BFGS
^^^^

.. autofunction:: .minimize
.. autofunction:: .maximize

Brent
^^^^^

.. autofunction:: .minimize_scalar
.. autofunction:: .maximize_scalar

"""

from .bfgs import maximize, minimize
from .brent import maximize as maximize_scalar
from .brent import minimize as minimize_scalar
