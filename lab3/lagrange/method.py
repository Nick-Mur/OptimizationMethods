"""Метод множителей Лагранжа для задачи f(x) -> extr при g_i(x) = 0, i = 1..m."""

from itertools import combinations
from typing import NamedTuple

import mpmath
import sympy as sp

DIGITS = 50       # точность численной проверки, значащих цифр
TOLERANCE = 1e-30  # числа меньше по модулю считаются нулём


class Point(NamedTuple):
    """Точка, подозрительная на экстремум.

    Attributes:
        coords: Значения переменных.
        value: Значение f.
        multipliers: Множители Лагранжа; None для нерегулярной точки.
        kind: Вывод о точке:
            "min", "max" — строгий локальный минимум, максимум;
            "saddle" — не экстремум (форма второго порядка знакопеременна);
            "degenerate" — условия второго порядка не дают ответа;
            "isolated" — изолированная точка допустимого множества;
            "irregular" — нерегулярная точка, метод к ней неприменим.
    """

    coords: tuple
    value: sp.Expr
    multipliers: tuple | None
    kind: str


class Solution(NamedTuple):
    """Результат решения задачи.

    Attributes:
        variables: Переменные.
        f: Целевая функция.
        g: Левые части ограничений g_i(x) = 0.
        equations: Система ∂L/∂x_j = 0, g_i = 0.
        points: Все точки, подозрительные на экстремум, по возрастанию f.
        bounded: Допустимое множество заведомо ограничено.
    """

    variables: list
    f: sp.Expr
    g: list
    equations: list
    points: list[Point]
    bounded: bool


def is_zero(expr: sp.Expr) -> bool:
    """Проверяет численно с точностью DIGITS знаков, что значение выражения равно нулю."""
    return abs(complex(sp.N(expr, DIGITS))) < TOLERANCE


def _is_real(value: sp.Expr) -> bool:
    """Значение вещественно; если SymPy не может это установить, проверяется численно."""
    if value.is_real is not None:
        return value.is_real
    return is_zero(sp.im(value))


def _real_solutions(equations: list, unknowns: list, variables: list) -> list[dict]:
    """Решает систему и оставляет вещественные решения.

    Args:
        equations: Уравнения вида выражение = 0.
        unknowns: Неизвестные системы.
        variables: Переменные задачи (координаты точки) среди неизвестных.

    Returns:
        Решения в виде словарей {неизвестная: значение}.

    Raises:
        ValueError: Система не решается аналитически или имеет бесконечно много решений.
    """
    try:
        solutions = sp.solve(equations, unknowns, dict=True)
    except NotImplementedError:
        raise ValueError("SymPy не может решить систему уравнений аналитически") from None

    # Оставить вещественные решения с определёнными координатами точек.
    real = []
    for solution in solutions:
        values = [solution.get(u, u) for u in unknowns]
        if any(v.is_real is False for v in values):
            continue
        coords = [solution.get(x, x) for x in variables]
        if any(c.free_symbols for c in coords):
            family = f"({', '.join(map(str, variables))}) = ({', '.join(map(str, coords))})"
            raise ValueError(f"подозрительных точек бесконечно много: {family.replace('**', '^')}. "
                             "Конечного списка кандидатов нет, задача исследуется без метода Лагранжа")
        if all(_is_real(v) for v in values if not v.free_symbols):
            real.append(solution)
    return real


def _classify(hessian: sp.Matrix, jacobian: sp.Matrix) -> str:
    """Определяет характер регулярной точки по условиям второго порядка.

    Знак формы dᵀ·H·d проверяется на касательном подпространстве J·d = 0. Если Z — базис
    этого подпространства, знак определяют собственные числа матрицы Zᵀ·H·Z.

    Args:
        hessian: Матрица Гессе функции Лагранжа по x в точке.
        jacobian: Матрица Якоби ограничений в точке.

    Returns:
        Значение Point.kind.
    """
    # Найти направления, касательные ко всем ограничениям: J·d = 0.
    tangent = jacobian.nullspace(simplify=True)
    if not tangent:
        return "isolated"
    # Перенести второй дифференциал в базис касательного пространства.
    basis = sp.Matrix.hstack(*tangent)
    reduced = (basis.T * hessian * basis).evalf(DIGITS).applyfunc(sp.re)
    with mpmath.workdps(DIGITS):
        eigenvalues = mpmath.eigsy(mpmath.matrix(reduced.tolist()), eigvals_only=True)
    # По знакам собственных чисел определить характер квадратичной формы.
    positive = any(e > TOLERANCE for e in eigenvalues)
    negative = any(e < -TOLERANCE for e in eigenvalues)
    if positive and negative:
        return "saddle"
    if any(abs(e) <= TOLERANCE for e in eigenvalues):
        return "degenerate"
    return "min" if positive else "max"


def _is_bounded(g: list, variables: list) -> bool:
    """Достаточный признак ограниченности допустимого множества.

    Множество ограничено, если:
        - ограничений столько же, сколько переменных: тогда каждая регулярная допустимая
          точка является решением системы Лагранжа, и множество состоит из найденных точек;
        - какое-либо g_i — многочлен второй степени со знакоопределённой матрицей
          квадратичной части: множество g_i = 0 является эллипсоидом.
    """
    if len(g) == len(variables):
        return True
    for gi in g:
        if sp.Poly(gi, *variables).total_degree() == 2:
            quadratic = sp.hessian(gi, variables)
            if quadratic.is_positive_definite or quadratic.is_negative_definite:
                return True
    return False


def find_extrema(f: sp.Expr, g: list, variables: list) -> Solution:
    """Находит и классифицирует все точки, подозрительные на условный экстремум.

    Шаги метода:
        1. Нерегулярные точки: g = 0 и rank J < m (все миноры порядка m матрицы Якоби равны нулю).
        2. Функция Лагранжа L = f + Σ λ_i·g_i и система ∂L/∂x_j = 0, ∂L/∂λ_i = g_i = 0.
        3. Вещественные решения системы в регулярных точках.
        4. Проверка подстановкой: ограничения и уравнения системы выполняются.
        5. Характер регулярных точек по знаку d²L на касательном подпространстве.

    Args:
        f: Целевая функция.
        g: Левые части ограничений g_i(x) = 0, от 1 до len(variables) штук.
        variables: Переменные.

    Returns:
        Решение задачи.

    Raises:
        ValueError: Точек бесконечно много, система не решается или результат не прошёл проверку.
    """
    # Найти допустимые точки, где градиенты ограничений линейно зависимы.
    m = len(g)
    jacobian = sp.Matrix(g).jacobian(variables)
    minors = [jacobian[:, list(cols)].det() for cols in combinations(range(len(variables)), m)]
    irregular = _real_solutions([*g, *minors], variables, variables)

    # Составить функцию Лагранжа, её производные и матрицу вторых производных.
    multipliers = sp.symbols(f"λ1:{m + 1}")
    lagrangian = f + sum(lam * gi for lam, gi in zip(multipliers, g))
    equations = [sp.diff(lagrangian, x) for x in variables] + g
    hessian = sp.hessian(lagrangian, variables)

    # Добавить нерегулярные точки и классифицировать регулярные решения системы.
    points = [Point(tuple(s.get(x, x) for x in variables), sp.simplify(f.subs(s)), None, "irregular")
              for s in irregular]
    for s in _real_solutions(equations, [*variables, *multipliers], variables):
        if jacobian.subs(s).rank(simplify=True) < m:
            continue  # нерегулярная точка, уже учтена
        points.append(Point(tuple(s[x] for x in variables), sp.simplify(f.subs(s)),
                            tuple(s[lam] for lam in multipliers),
                            _classify(hessian.subs(s), jacobian.subs(s))))

    # Проверить кандидатов подстановкой в ограничения и уравнения системы.
    for p in points:
        at = dict(zip(variables, p.coords)) | dict(zip(multipliers, p.multipliers or ()))
        residuals = g if p.multipliers is None else equations
        if not all(is_zero(r.subs(at)) for r in residuals):
            raise ValueError(f"точка {p.coords} не прошла проверку подстановкой")

    # Упорядочить кандидатов по значению целевой функции.
    points.sort(key=lambda p: sp.re(sp.N(p.value, DIGITS)))
    return Solution(variables, f, g, equations, points, _is_bounded(g, variables))
