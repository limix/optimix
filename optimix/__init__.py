r"""
***************
optimix package
***************

Abstract function optimisation.

"""
from __future__ import absolute_import

from . import testing
from ._check_grad import approx_fprime, check_grad
from ._exception import OptimixError
from ._function import Function, FunctionReduce
from ._optimize import maximize, maximize_scalar, minimize, minimize_scalar
from ._testit import test
from ._types import Matrix, Scalar, Vector

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "test",
    "testing",
    "approx_fprime",
    "check_grad",
    "OptimixError",
    "Function",
    "FunctionReduce",
    "maximize",
    "maximize_scalar",
    "minimize",
    "minimize_scalar",
    "Matrix",
    "Scalar",
    "Vector",
]
