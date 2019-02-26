from numpy import array, add
from numpy.testing import assert_allclose

from optimix import Func, Scalar, Vector


class Foo1(Func):
    def __init__(self):
        self._a = Vector([0, 0])
        self._b = Vector([0, 0])
        self._c = Scalar(1)
        super(Foo1, self).__init__("Foo1", a=self._a, b=self._b, c=self._c)

    @property
    def a(self):
        return self._a.value

    @property
    def b(self):
        return self._b.value

    @property
    def c(self):
        return self._c.value

    def fix_c(self):
        self._c.fix()

    def unfix_c(self):
        self._c.unfix()

    @c.setter
    def c(self, v):
        self._c.value = v

    def value(self):
        a = self.a
        b = self.b
        c = self.c
        return (a @ b - 3 + a @ [1, 1] - b @ [1, 2] + 1 / c) ** 2

    def gradient(self):
        a = self.a
        b = self.b
        c = self.c
        v = a @ b - 3 + a @ [1, 1] - b @ [1, 2] + 1 / c
        da = 2 * v * array([b[0] + 1, b[1] + 1])
        db = 2 * v * array([a[0] - 1, a[1] - 2])
        dc = 2 * v * -1 / (c ** 2)
        return {"a": da, "b": db, "c": dc}

    def check_grad(self):
        return self._check_grad()


def test_foo1func(capsys):
    f = Foo1()

    assert_allclose(f.value(), 4)
    assert_allclose(f.check_grad(), 0, atol=1e-6)

    f.minimize(verbose=False)
    assert_allclose(f.value(), 0, atol=1e-6)
    assert_allclose(f.a, [0.296_404_030_827_721_43, 0.320_073_722_957_560_3])
    assert_allclose(f.b, [-0.296_404_030_827_721_43, -0.569_138_369_525_604])
    assert_allclose(f.c, 0.820_436_694_621_109_5)

    f.minimize(verbose=True)
    captured = capsys.readouterr()
    assert (
        captured.out == "Gradient near zero before the first iteration. "
        "Returning the current value.\n"
    )

    f.c = 1.0
    f.fix_c()
    f.minimize(verbose=False)
    assert_allclose(f.value(), 0, atol=1e-6)
    assert_allclose(f.a, [0.335_604_171_088_476_3, 0.344_148_411_173_441_14])
    assert_allclose(f.b, [-0.335_604_171_088_476_3, -0.662_664_102_091_987_9])
    assert_allclose(f.c, 1)


class Foo2(Func):
    def __init__(self):
        self._c = Scalar(1)
        self._c.bounds = [1e-9, 1e9]
        super(Foo2, self).__init__("Foo2", c=self._c)

    @property
    def c(self):
        return self._c.value

    @c.setter
    def c(self, v):
        self._c.value = v

    def value(self):
        c = self.c
        return 1 / (c ** 2)

    def gradient(self):
        c = self.c
        return {"c": -2 / (c ** 3)}

    def check_grad(self):
        return self._check_grad()


def test_foo2func(capsys):
    f = Foo2()

    assert_allclose(f.value(), 1)
    assert_allclose(f.check_grad(), 0, atol=1e-6)

    f.maximize(verbose=False)
    assert_allclose(f.c, 0, atol=1e-6)


class Foo3(Func):
    def __init__(self, funcs):
        self._funcs = funcs
        super(Foo3, self).__init__("Foo3", funcs)

    def value(self):
        return add.reduce([f.value() for f in self._funcs])

    def gradient(self):
        grad = {}
        for i, f in enumerate(self._funcs):
            for varname, g in f.gradient().items():
                grad[f"{self._name}[{i}].{varname}"] = g
        return grad

    def check_grad(self):
        return self._check_grad()

    def operand(self, i):
        return self._funcs[i]


def test_foo3func(capsys):
    f1 = Foo1()
    f2 = Foo2()
    f = Foo3([f1, f2])

    assert_allclose(f.value(), 5)
    assert_allclose(f.check_grad(), 0, atol=1e-6)

    f.minimize(verbose=False)
    assert_allclose(f.value(), 0, atol=1e-6)
    assert_allclose(f.operand(0).c, 0.352_550_152_809_346)
    assert_allclose(f1.c, f.operand(0).c)

    f1.c = 1.5
    f1.fix_c()
    f.minimize(verbose=False)
    assert_allclose(f.value(), 0, atol=1e-6)
    assert_allclose(f.operand(0).c, 1.5)
