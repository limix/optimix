from .function import Function
from .function import FunctionReduce
from .types import Scalar
from .types import Vector
from .types import Matrix
from .check_grad import check_grad
from .optimize import minimize_scalar
from .optimize import maximize_scalar
from .optimize import minimize
from .optimize import maximize
from .variables import merge_variables
from .util import as_data_function

from pkg_resources import get_distribution as _get_distribution
from pkg_resources import DistributionNotFound as _DistributionNotFound

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
