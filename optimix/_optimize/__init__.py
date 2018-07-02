r"""
********************
Optimization methods
********************

L-BFGS-B
^^^^^^^^

.. autofunction:: optimix._optimize.minimize
.. autofunction:: optimix._optimize.maximize

Brent's method
^^^^^^^^^^^^^^

.. autofunction:: optimix._optimize.minimize_scalar
.. autofunction:: optimix._optimize.maximize_scalar

"""

from ._bfgs import maximize, minimize
from ._brent import maximize as maximize_scalar
from ._brent import minimize as minimize_scalar

__all__ = ["minimize", "maximize", "minimize_scalar", "maximize_scalar"]
