from .function import Function
from .function import FunctionReduce
from .function import FunctionDataFeed
from .function import FunctionReduceDataFeed


def as_data_function(function, purpose='learn'):
    if isinstance(function, Function):
        return FunctionDataFeed(function, function._data[purpose])
    elif isinstance(function, FunctionReduce):
        fs = [as_data_function(f, purpose) for f in function._functions]
        return FunctionReduceDataFeed(function, fs)
    raise Exception
