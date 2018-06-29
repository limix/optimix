r"""
********************
Optimization methods
********************

L-BFGS-B
^^^^^^^^

.. autofunction:: optimix.optimize.minimize
.. autofunction:: optimix.optimize.maximize

Brent's method
^^^^^^^^^^^^^^

.. autofunction:: optimix.optimize.minimize_scalar
.. autofunction:: optimix.optimize.maximize_scalar

"""

from ._bfgs import maximize, minimize
from ._brent import maximize as maximize_scalar
from ._brent import minimize as minimize_scalar

__all__ = ["minimize", "maximize", "minimize_scalar", "maximize_scalar"]
