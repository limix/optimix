"""
Abstract function optimisation.
"""
from ._exception import OptimixError
from ._function import Function
from ._testit import test
from ._types import Matrix, Scalar, Vector

__version__ = "3.0.4"

__all__ = [
    "__version__",
    "Function",
    "Matrix",
    "OptimixError",
    "Scalar",
    "test",
    "Vector",
]
