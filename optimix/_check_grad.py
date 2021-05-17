__all__ = ["approx_fprime", "check_grad"]


def approx_fprime(f, step=1.49e-08):
    from numpy import asarray, atleast_1d, squeeze, stack

    f0 = f.value()
    grad = {}
    for name in f.variables().names():
        value = f.variables().get(name).value
        ndim = value.ndim
        value = atleast_1d(value).ravel()
        grads = []
        for i in range(len(value)):
            value[i] += step
            grads.append(asarray((f.value() - f0) / step))
            value[i] -= step
        grad[name] = stack(grads, axis=-1)
        if ndim == 0:
            grad[name] = squeeze(grad[name], axis=-1)
    return grad


def check_grad(func, step=1.49e-08):
    from numpy.linalg import norm
    from numpy import asarray

    g = func.gradient()
    g = {n: asarray(gi) for n, gi in iter(g.items())}
    fg = approx_fprime(func, step)

    names = set(g.keys()).union(fg.keys())
    return sum(norm(fg[name] - g[name]) for name in names)
