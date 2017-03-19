r"""
*********
Variables
*********

Introduction
^^^^^^^^^^^^

:class:`.Variables` holds a set of variables.

Public interface
^^^^^^^^^^^^^^^^
"""
from __future__ import unicode_literals

class Variables(dict):
    r"""Set of variables."""
    # def new(self):
    #     return Variables({name: None for name in self.names()})

    # def flatten(self):
    #     names = sorted(self.names())
    #     x = [self[k].asarray().ravel() for k in names]
    #     return concatenate(x)

    # def from_flat(self, x):
    #     names = sorted(self.names())
    #     offset = 0
    #     for n in names:
    #         size = self[n].size
    #         self[n].value = x[offset:offset + size]
    #         offset += size

    def set(self, x):
        """Set variable values via a dictionary mapping name -> value."""
        for name, value in iter(x.items()):
            self[name].value = value

    def select(self, fixed):
        """Return a subset of variables according to ``fixed``."""
        names = [n for n in self.names() if self[n].isfixed == fixed]
        return Variables({n: self[n] for n in names})

    def names(self):
        """Return the variable names."""
        return sorted(super(Variables, self).keys())

    def keys(self):
        msg = "'Variables' object has no attribute 'keys'. "
        msg += "You might want to use attribute 'names' instead."
        raise AttributeError(msg)

    def __str__(self):
        msg = 'Variables('
        names = sorted(self.names())

        for i, n in enumerate(names):
            v = self[n]
            if i > 0:
                msg += '          '
            msg += '%s=%s' % (n, v)
            if i + 1 < len(names):
                msg += ',\n'

        msg += ')'
        return msg

    def __repr__(self):
        return str(self)


def merge_variables(variables_dict):
    variables = Variables()

    for (prefix, vs) in iter(variables_dict.items()):
        for (name, value) in iter(vs.items()):
            variables[prefix + '.' + name] = value

    return variables
