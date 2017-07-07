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

from .bfgs import maximize, minimize
from .brent import maximize as maximize_scalar
from .brent import minimize as minimize_scalar

__all__ = ['minimize', 'maximize', 'minimize_scalar', 'maximize_scalar']
