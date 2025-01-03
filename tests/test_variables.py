from numpy.testing import assert_, assert_allclose, assert_equal

from optimix import Scalar
from optimix._variables import Variables, merge_variables


def test_variables_set():
    a = Scalar(1.0)
    b = a
    a.value = 2.0
    assert_(a is b)
    assert_(a.raw is b.raw)

    v = Variables(dict(a=Scalar(1.0), b=Scalar(1.5)))
    v.set({"a": 0.5})
    assert_allclose(v.get("a"), 0.5)


def test_variables_str():
    v = Variables(dict(a=Scalar(1.0), b=Scalar(1.5)))
    msg = "Variables(a=Scalar(1.0),\n"
    msg += " " * 10 + "b=Scalar(1.5))"
    assert_equal(v.__str__(), msg)


def test_variables_merge():
    a = Variables(a0=Scalar(1.0))
    b = Variables(b0=Scalar(1.0))
    c = merge_variables(dict(a=a, b=b))

    a.get("a0").value += 1.0

    assert_equal(a.get("a0").value, 2.0)
    assert_equal(a.get("a0").value, c.get("a.a0").value)


def test_variables_setattr():
    a = Variables(a0=Scalar(1.0))

    a["a1"] = Scalar(2.0)
    a["a1"].value += 1.0

    assert_equal(a.get("a0").value, 1.0)
    assert_equal(a.get("a1").value, 3.0)
