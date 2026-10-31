"""Тесты метода: ответы на все 20 вариантов задания и характер точек."""

import pytest
import sympy as sp

from lagrange import solve

r = sp.sqrt

VARIANTS = [  # целевая функция, ограничения, наименьшее и наибольшее f среди точек
    ("2*x + y", ["x^2 + y^2 = 5"], -5, 5),
    ("x + 2*y", ["4*x^2 + y^2 = 4"], -r(17), r(17)),
    ("x^2 + 2*y^2", ["x^2 + y^2 = 9"], 9, 18),
    ("x*y", ["x^2 + 4*y^2 = 8"], -2, 2),
    ("3*x - 4*y", ["x^2 + y^2 = 25"], -25, 25),
    ("x^2 + y^2", ["x^2 + 2*y^2 = 8"], 4, 8),
    ("x + 2*y + 3*z", ["x^2 + y^2 + z^2 = 14"], -14, 14),
    ("2*x - y + 2*z", ["x^2 + y^2 + z^2 = 9"], -9, 9),
    ("x^2 + 3*y^2 + 3*z^2", ["x^2 + y^2 + z^2 = 6", "z = x"], 12, 18),
    ("x + 2*y + z", ["x^2 + y^2 + z^2 = 22", "x = y"], -11, 11),
    ("x^2 + y^2 + 4*z^2", ["x^2 + y^2 + z^2 = 6", "x + y + z = 0"], 6, 18),
    ("f(x, y, z, w) = 2*w + x + y + z", ["w^2 + x^2 + y^2 + z^2 = 7"], -7, 7),
    ("f(x, y, z, w) = 2*w + x + 2*y - z", ["w^2 + x^2 + y^2 + z^2 = 10"], -10, 10),
    ("f(x, y, z, w) = 3*w^2 + x^2 + y^2 + 2*z^2", ["w^2 + x^2 + y^2 + z^2 = 10", "x = y"], 10, 30),
    ("f(x, y, z, w) = 2*w^2 + x*y + z^2", ["w^2 + x^2 + y^2 + z^2 = 8", "x = y"], 4, 16),
    ("x", ["y^2 = x^2*(1 - x^2)"], -1, 1),
    ("y", ["y^2 = x^2*(1 - x^2)"], -sp.Rational(1, 2), sp.Rational(1, 2)),
    ("x", ["y^2 = x^3*(1 - x)"], 0, 1),
    ("x^2 + 2*y^2 + 3*z^2", ["x^2 + y^2 + z^2 = 4", "x*y = 0"], 4, 12),
    ("x + y + 2*z", ["x^2 + y^2 + z^2 = 5", "x*z = 0"], -5, 5),
]


@pytest.mark.parametrize("number, f, g, low, high", [(i, *v) for i, v in enumerate(VARIANTS, 1)])
def test_variant(number, f, g, low, high):
    points = solve(f, g).points
    assert sp.simplify(points[0].value - low) == 0
    assert sp.simplify(points[-1].value - high) == 0


def test_variant_20_matches_manual_solution():
    """Четыре точки Лагранжа и две нерегулярные точки, как в отчёте."""
    t = r(10) / 2
    expected = {
        (0, -1, -2): (-5, "min"), (-t, -t, 0): (-2 * t, "min"),
        (0, -r(5), 0): (-r(5), "irregular"), (0, r(5), 0): (r(5), "irregular"),
        (t, t, 0): (2 * t, "max"), (0, 1, 2): (5, "max"),
    }
    solution = solve("x + y + 2*z", ["x^2 + y^2 + z^2 = 5", "x*z = 0"])
    assert solution.bounded
    assert {p.coords: (p.value, p.kind) for p in solution.points} == expected


def test_variable_order_from_header():
    """В варианте 12 координаты идут в порядке (x, y, z, w), а не по алфавиту."""
    assert solve("f(x, y, z, w) = 2*w + x + y + z", ["w^2 + x^2 + y^2 + z^2 = 7"]).points[-1].coords == (1, 1, 1, 2)


def test_saddle_is_not_an_extremum():
    """Вариант 14: в точках с f = 20 форма d²L знакопеременна."""
    points = solve("f(x, y, z, w) = 3*w^2 + x^2 + y^2 + 2*z^2", ["w^2 + x^2 + y^2 + z^2 = 10", "x = y"]).points
    assert [p.kind for p in points if p.value == 20] == ["saddle", "saddle"]


def test_degenerate_second_order():
    """f = x³ при y = 0: точка (0, 0) стационарна, но d²L = 0."""
    assert [p.kind for p in solve("x^3 + y", ["y = 0"]).points] == ["degenerate"]


def test_as_many_constraints_as_variables():
    """m = n: допустимое множество конечно, все его точки изолированные."""
    solution = solve("x + y", ["x^2 = 4", "y = 1"])
    assert solution.bounded
    assert [(p.value, p.kind) for p in solution.points] == [(-1, "isolated"), (3, "isolated")]


def test_irregular_point_only():
    """x² + y² = 0: единственная допустимая точка нерегулярна (пример из методических указаний)."""
    assert [(p.coords, p.kind) for p in solve("x", ["x^2 + y^2 = 0"]).points] == [((0, 0), "irregular")]


def test_complex_roots_are_dropped():
    """Кубическое уравнение с одним вещественным корнем: комплексные решения отбрасываются."""
    points = solve("x", ["y^2 = x^3 - x + 1"]).points
    assert len(points) == 1 and points[0].kind == "min"
    assert abs(float(sp.re(sp.N(points[0].value))) + 1.3247179572) < 1e-9


def test_no_points():
    assert solve("x + y", ["x^2 + y^2 = -1"]).points == []
    assert solve("x - y", ["x*y = 1"]).points == []


def test_unbounded_set_is_not_declared_bounded():
    assert not solve("x^2 + y^2", ["x + y = 1"]).bounded


@pytest.mark.parametrize("f, g", [
    ("y + 0*x", ["y^2 = 1"]),                  # f постоянна на прямых y = ±1
    ("x", ["x + y = 1", "2*x + 2*y = 2"]),     # зависимые ограничения
    ("x + y", ["(x^2 + y^2 - 1)^2 = 0"]),      # вся окружность нерегулярна
])
def test_infinitely_many_points(f, g):
    with pytest.raises(ValueError, match="бесконечно много"):
        solve(f, g)
