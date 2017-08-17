r"""
***************
optimix package
***************

Abstract function optimisation.

"""
from . import testing
from ._test import test
from .check_grad import approx_fprime, check_grad
from .exception import OptimixError
from .function import Function, FunctionReduce
from .optimize import maximize, maximize_scalar, minimize, minimize_scalar
from .types import Matrix, Scalar, Vector

__version__ = "1.2.15"

__all__ = [
    '__version__', 'test', 'testing', 'approx_fprime', 'check_grad',
    'OptimixError', 'Function', 'FunctionReduce', 'maximize',
    'maximize_scalar', 'minimize', 'minimize_scalar', 'Matrix', 'Scalar',
    'Vector'
]
