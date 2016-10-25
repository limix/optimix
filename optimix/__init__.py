from pkg_resources import DistributionNotFound as _DistributionNotFound
from pkg_resources import get_distribution as _get_distribution

from .check_grad import approx_fprime, check_grad
from .function import Composite, Function, FunctionReduce
from .optimize import maximize, maximize_scalar, minimize, minimize_scalar
from .types import Matrix, Scalar, Vector

try:
    __version__ = _get_distribution('optimix').version
except _DistributionNotFound:
    __version__ = 'unknown'


def test():
    import os
    p = __import__('optimix').__path__[0]
    src_path = os.path.abspath(p)
    old_path = os.getcwd()
    os.chdir(src_path)

    try:
        return_code = __import__('pytest').main([])
    finally:
        os.chdir(old_path)

    return return_code
