r"""
***************
optimix package
***************

Abstract function optimisation.

"""
from __future__ import absolute_import as _

from . import testing
from .check_grad import approx_fprime, check_grad
from .exception import OptimixError
from .function import Function, FunctionReduce
from .optimize import maximize, maximize_scalar, minimize, minimize_scalar
from .testit import test
from .types import Matrix, Scalar, Vector

__version__ = "1.2.23"

__all__ = [
    "__version__", "test", 'testing', 'approx_fprime', 'check_grad',
    'OptimixError', 'Function', 'FunctionReduce', 'maximize',
    'maximize_scalar', 'minimize', 'minimize_scalar', 'Matrix', 'Scalar',
    'Vector'
]
