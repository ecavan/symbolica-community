"""Symbolic one-loop reduction under `symbolica.community.hep.oneloop`.

Run unparallelised: restricted Symbolica aborts when touched from a second thread.
"""

import importlib

import pytest

from symbolica import E, Expression, S
from symbolica.community import hep
from symbolica.community.hep import oneloop


def test_namespace_and_dot_attributes():
    assert importlib.import_module("symbolica.community.hep.oneloop") is oneloop
    assert hep.oneloop is oneloop
    for name in ("Propagator", "IntegralFamily", "Reduction", "MasterIntegral"):
        assert getattr(oneloop, name).__module__ == "symbolica.community.hep.oneloop"

    # Importing hep must register these attributes before user code can create
    # the symbol with incompatible defaults in the shared symbol table.
    dot = S("oneloopreduce::dot", is_symmetric=True, is_linear=True)
    k, q1, q2 = S("oneloopreduce::k", "oneloopreduce::q1", "oneloopreduce::q2")
    assert dot(q1, k) == dot(k, q1)
    assert dot(k, q1 + q2) == dot(k, q1) + dot(k, q2)
    with pytest.raises(TypeError):
        S("oneloopreduce::dot", is_symmetric=False)


@pytest.mark.parametrize(
    "kind,head,masses,invariants,arguments",
    [
        ("tadpole", "A0", ["m0"], [], ["m0"]),
        ("bubble", "B0", ["m0", "m1"], ["p01"], ["p01", "m0", "m1"]),
        (
            "triangle", "C0", ["m0", "m1", "m2"], ["p01", "p02", "p12"],
            ["p01", "p12", "p02", "m0", "m1", "m2"],
        ),
        (
            "box", "D0", ["m0", "m1", "m2", "m3"],
            ["p01", "p02", "p03", "p12", "p13", "p23"],
            ["p01", "p12", "p23", "p03", "p02", "p13", "m0", "m1", "m2", "m3"],
        ),
    ],
)
def test_scalar_master_mapping(kind, head, masses, invariants, arguments):
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(E(mass)) for mass in masses], list(map(E, invariants)),
    )
    reduction = family.reduce().simplify()
    ((coefficient, master),) = reduction.terms
    assert coefficient == E("1")
    assert (master.kind, master.head) == (kind, head)
    assert master.arguments == list(map(E, arguments))
    assert reduction.to_expression() == S(f"oneloopreduce::{head}")(*map(E, arguments))
    expected = S(f"oneloopmaster::{head}")(*map(E, arguments), E("mu2"))
    assert master.to_oneloopmaster(E("mu2")) == expected
    assert reduction.to_oneloopmaster(E("mu2")) == expected


def test_rank_one_triangle_matches_the_scalar_master_combination():
    dot, k, q1 = S("oneloopreduce::dot", "oneloopreduce::k", "oneloopreduce::q1")
    m, p1, p2, s = S("olr_test::m", "olr_test::p1", "olr_test::p2", "olr_test::s")
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(m)] * 3, [p1, s, p2], numerator=dot(k, q1),
    )
    bubble, triangle = S("oneloopreduce::B0", "oneloopreduce::C0")
    expected = (bubble(s, m, m) - bubble(p2, m, m) - p1 * triangle(p1, p2, s, m, m, m)) / 2
    assert (family.reduce().simplify().to_expression() - expected).expand() == E("0")


def test_raised_tadpole_keeps_its_dimension_dependence():
    m, d = S("olr_test::tadpole_m"), S("oneloopreduce::d")
    reduction = oneloop.IntegralFamily(
        [oneloop.Propagator(m)], [], exponents=[2],
    ).reduce().simplify()
    ((coefficient, master),) = reduction.terms
    assert (coefficient - (d - 2) / (2 * m)).expand() == E("0")
    assert master.head == "A0"


@pytest.mark.parametrize("exponents", [[2, 2], [3, 1]])
def test_coincident_dotted_bubble_is_the_power_four_tadpole(exponents):
    # At zero momentum and equal masses the two lines are one denominator.
    # These used to come back as an indeterminate coefficient.
    bubble = oneloop.IntegralFamily(
        [oneloop.Propagator(E("1"))] * 2, [E("0")], exponents=exponents,
    ).reduce().simplify()
    tadpole = oneloop.IntegralFamily(
        [oneloop.Propagator(E("1"))], [], exponents=[4],
    ).reduce().simplify()
    d = S("oneloopreduce::d")
    assert (tadpole.to_expression() - (d - 6) * (d - 4) * (d - 2) / 48
            * S("oneloopreduce::A0")(1)).expand() == E("0")
    assert (bubble.to_expression() - tadpole.to_expression()).expand() == E("0")


def unsupported_numerators():
    dot, k = S("oneloopreduce::dot"), S("oneloopreduce::k")
    q1, q2, q4 = S("oneloopreduce::q1", "oneloopreduce::q2", "oneloopreduce::q4")
    pol = S("olr_test::polarization")
    return {
        "direction outside the chain": dot(k, q4),
        "direction a bubble lacks": dot(k, q2),
        "polarization vector": dot(k, pol),
        "bare loop momentum": k,
        "negative power": 1 / dot(k, k),
        "fractional power": dot(k, q1) ** E("1/2"),
    }


@pytest.mark.parametrize("what", list(unsupported_numerators()))
def test_unsupported_numerators_raise(what):
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(E("1"))] * 2, [E("-2")],
        numerator=unsupported_numerators()[what],
    )
    with pytest.raises(ValueError, match="unsupported numerator"):
        family.reduce()


def test_numerator_degree_bound_is_an_error():
    dot, k = S("oneloopreduce::dot"), S("oneloopreduce::k")
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(E("1"))], [], numerator=dot(k, k) ** 21,
    )
    with pytest.raises(ValueError, match="exceeds the supported bound"):
        family.reduce()


def test_external_prefactors_stay_valid():
    dot, k, q1, d = S("oneloopreduce::dot", "oneloopreduce::k", "oneloopreduce::q1",
                      "oneloopreduce::d")
    prefactor = S("olr_test::g") / (d - 4)
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(E("1"))] * 2, [E("-2")], numerator=prefactor * dot(k, q1),
    )
    assert isinstance(family.reduce().to_expression(), Expression)


@pytest.mark.parametrize("name", ["reg_delta", "xll", "xq1", "den1"])
def test_internal_names_are_rejected(name):
    reserved = S(f"oneloopreduce::{name}")
    family = oneloop.IntegralFamily([oneloop.Propagator(reserved)] * 2, [E("0")])
    with pytest.raises(ValueError, match="uses internally"):
        family.reduce()
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(E("1"))], [], numerator=reserved,
    )
    with pytest.raises(ValueError, match="uses internally"):
        family.reduce()


@pytest.mark.parametrize(
    "count,invariants,exponents",
    [(0, [], None), (2, [], None), (2, ["s", "t"], None), (2, ["s"], [1])],
)
def test_malformed_family_raises_value_error(count, invariants, exponents):
    with pytest.raises(ValueError):
        oneloop.IntegralFamily(
            [oneloop.Propagator(E("0"))] * count, list(map(E, invariants)),
            exponents=exponents,
        )


@pytest.mark.parametrize("exponents", [[40, 1], [-1, 1]])
def test_unsupported_indices_raise_value_error(exponents):
    family = oneloop.IntegralFamily(
        [oneloop.Propagator(E("0"))] * 2, [E("s")], exponents=exponents,
    )
    with pytest.raises(ValueError):
        family.reduce()
