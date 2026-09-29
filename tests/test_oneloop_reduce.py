"""One-loop reduction in `symbolica.community.hep.oneloop`. Run unparallelised."""

import importlib

import pytest

from symbolica import E, S
from symbolica.community import hep
from symbolica.community.hep import oneloop

dot, k, d = S("oneloopreduce::dot", "oneloopreduce::k", "oneloopreduce::d")
q1, q2, q4 = S("oneloopreduce::q1", "oneloopreduce::q2", "oneloopreduce::q4")
A0, B0, C0 = S("oneloopreduce::A0", "oneloopreduce::B0", "oneloopreduce::C0")


def family(masses, invariants, **kwargs):
    return oneloop.IntegralFamily(
        [oneloop.Propagator(E(str(m))) for m in masses],
        [E(str(s)) for s in invariants],
        **kwargs,
    )


def same(a, b):
    return (a - b).expand() == E("0")


def test_namespace_and_dot_attributes():
    assert importlib.import_module("symbolica.community.hep.oneloop") is oneloop
    assert hep.oneloop is oneloop
    for name in ("Propagator", "IntegralFamily", "Reduction", "MasterIntegral"):
        assert getattr(oneloop, name).__module__ == "symbolica.community.hep.oneloop"
    # Importing hep registers dot as symmetric and linear before user code can.
    assert dot(q1, k) == dot(k, q1)
    assert dot(k, q1 + q2) == dot(k, q1) + dot(k, q2)
    with pytest.raises(TypeError):
        S("oneloopreduce::dot", is_symmetric=False)


@pytest.mark.parametrize(
    "head,masses,invariants,arguments",
    [
        ("A0", ["m0"], [], ["m0"]),
        ("B0", ["m0", "m1"], ["p01"], ["p01", "m0", "m1"]),
        ("C0", ["m0", "m1", "m2"], ["p01", "p02", "p12"], ["p01", "p12", "p02", "m0", "m1", "m2"]),
        (
            "D0",
            ["m0", "m1", "m2", "m3"],
            ["p01", "p02", "p03", "p12", "p13", "p23"],
            ["p01", "p12", "p23", "p03", "p02", "p13", "m0", "m1", "m2", "m3"],
        ),
    ],
)
def test_scalar_masters_in_avh_order(head, masses, invariants, arguments):
    # Invariants go in lexicographically; master arguments come out in AVH order.
    reduction = family(masses, invariants).reduce().simplify()
    ((coefficient, master),) = reduction.terms
    args = [E(a) for a in arguments]
    assert coefficient == E("1") and master.head == head and master.arguments == args
    assert reduction.to_expression() == S(f"oneloopreduce::{head}")(*args)
    assert reduction.to_oneloopmaster(E("mu2")) == S(f"oneloopmaster::{head}")(*args, E("mu2"))


def test_rank_one_triangle():
    m, p1, p2, s = S("t::m", "t::p1", "t::p2", "t::s")
    triangle = oneloop.IntegralFamily(
        [oneloop.Propagator(m)] * 3, [p1, s, p2], numerator=dot(k, q1)
    )
    want = (B0(s, m, m) - B0(p2, m, m) - p1 * C0(p1, p2, s, m, m, m)) / 2
    assert same(triangle.reduce().simplify().to_expression(), want)


def test_raised_powers():
    m = S("t::m")
    tadpole = oneloop.IntegralFamily([oneloop.Propagator(m)], [], exponents=[2]).reduce().simplify()
    assert same(tadpole.to_expression(), (d - 2) / (2 * m) * A0(m))
    # At zero momentum with equal masses the two lines are one denominator.
    power4 = (d - 6) * (d - 4) * (d - 2) / 48 * A0(1)
    for exponents in ([2, 2], [3, 1]):
        assert same(family([1, 1], [0], exponents=exponents).reduce().simplify().to_expression(), power4)


def test_on_shell_limits_that_need_the_exact_fallback():
    # No termwise limit of the off-shell regulator exists for these two.
    cases = [
        family([0] * 4, [0, -7, -3, 0, -5, -2], exponents=[2, 1, 1, 1]),
        family([1] * 3, [0, -3, 0], exponents=[2, 2, 1], numerator=dot(k, q2)),
    ]
    for f in cases:
        assert len(f.reduce()) > 0


@pytest.mark.parametrize(
    "numerator",
    [dot(k, q4), dot(k, q2), dot(k, S("t::polarization")), k, 1 / dot(k, k),
     dot(k, q1) ** E("1/2")],
)
def test_unsupported_numerators_raise(numerator):
    with pytest.raises(ValueError, match="unsupported numerator"):
        family([1, 1], [-2], numerator=numerator).reduce()


def test_numerator_limits():
    with pytest.raises(ValueError, match="exceeds the bound"):
        family([1], [], numerator=dot(k, k) ** 21).reduce()
    # Any k-free prefactor is fine, including poles in d.
    assert len(family([1, 1], [-2], numerator=S("t::g") / (d - 4) * dot(k, q1)).reduce()) > 0


@pytest.mark.parametrize("name", ["reg_delta", "xll", "xq1", "den1"])
def test_internal_names_are_rejected(name):
    reserved = S(f"oneloopreduce::{name}")
    with pytest.raises(ValueError, match="reserved"):
        oneloop.IntegralFamily([oneloop.Propagator(reserved)] * 2, [E("0")]).reduce()
    with pytest.raises(ValueError, match="reserved"):
        family([1], [], numerator=reserved).reduce()


@pytest.mark.parametrize(
    "count,invariants,exponents",
    [(0, [], None), (2, [], None), (2, ["s", "t"], None), (2, ["s"], [1])],
)
def test_malformed_families_raise(count, invariants, exponents):
    with pytest.raises(ValueError):
        family([0] * count, invariants, exponents=exponents)


@pytest.mark.parametrize("exponents", [[40, 1], [-1, 1]])
def test_unsupported_indices_raise(exponents):
    with pytest.raises(ValueError):
        family([0, 0], ["s"], exponents=exponents).reduce()
