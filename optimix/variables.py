from __future__ import unicode_literals

from numpy import concatenate


class Variables(dict):

    def __init__(self, *args, **kw):
        super(Variables, self).__init__(*args, **kw)

    def new(self):
        return Variables({name: None for name in self.names()})

    def flatten(self):
        names = sorted(self.names())
        x = [self[k].to_ndarray().ravel() for k in names]
        return concatenate(x)

    def from_flat(self, x):
        names = sorted(self.names())
        offset = 0
        for n in names:
            size = self[n].size
            self[n].value = x[offset:offset + size]
            offset += size

    def select(self, fixed):
        names = [n for n in self.names() if self[n].isfixed == fixed]
        return Variables({n: self[n] for n in names})

    def names(self):
        return sorted(super(Variables, self).keys())

    def keys(self):
        msg = "'Variables' object has no attribute 'keys'. "
        msg += "You might want to use attribute 'names' instead."
        raise AttributeError(msg)

    def __str__(self):
        msg = 'Variables('
        if len(self) == 0:
            return msg + ')'

        names = sorted(self.names())

        for i in range(len(names)):
            n = names[i]
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
