r"""
***************
optimix package
***************

Abstract function optimisation.

"""
from __future__ import absolute_import as _absolute_import

from ._check_grad import approx_fprime, check_grad
from ._exception import OptimixError
from ._function import Function, FunctionReduce
from ._testit import test
from ._types import Matrix, Scalar, Vector
from ._testing import Assertion

__version__ = "2.0.2"

__all__ = [
    "__version__",
    "test",
    "approx_fprime",
    "check_grad",
    "OptimixError",
    "Function",
    "FunctionReduce",
    "Matrix",
    "Scalar",
    "Vector",
    "Assertion",
]
