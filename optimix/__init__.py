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

__name__ = "optimix"
__version__ = "1.2.19"
__author__ = "Danilo Horta"
__author_email__ = "horta@ebi.ac.uk"

__all__ = [
    "__name__", "__version__", "__author__", "__author_email__", "test",
    'testing', 'approx_fprime', 'check_grad', 'OptimixError', 'Function',
    'FunctionReduce', 'maximize', 'maximize_scalar', 'minimize',
    'minimize_scalar', 'Matrix', 'Scalar', 'Vector'
]
