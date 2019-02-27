"""
Optimization methods.
"""

from ._bfgs import maximize, minimize
from ._brent import maximize as maximize_scalar, minimize as minimize_scalar

__all__ = ["minimize", "maximize", "minimize_scalar", "maximize_scalar"]
