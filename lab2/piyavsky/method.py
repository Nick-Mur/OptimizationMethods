"""Метод ломаных (метод Пиявского).

g_i(x) = W(u_i) - L|x - u_i|,  p_n(x) = max g_i(x) <= W(x).
u_0 = a, u_1 = b, u_n = argmin p_{n-1};  стоп: W(u_n) - p_{n-1}(u_n) < eps.
"""

import time
import warnings
from typing import NamedTuple

import numpy as np

from .validation import check_finite, check_parameters, parse_function


class Result(NamedTuple):
    """Результат minimize.

    Attributes:
        x, value: Найденная точка x* и W(x*).
        gap: W(x*) - min p, оценка погрешности сверху.
        L: Константа Липшица.
        iterations: Номер последней итерации n (u_0 = a, u_1 = b).
        seconds: Время работы.
        status: "eps" — точность достигнута, "max_iter" — лимит итераций,
            "float_limit" — предел точности double.
        points: Пары (u_k, W(u_k)) в порядке вычисления.
        function, mode: Исходная функция и "min" / "max".
    """

    x: float
    value: float
    gap: float
    L: float
    iterations: int
    seconds: float
    status: str
    points: np.ndarray
    function: str
    mode: str


def estimate_L(W, a: float, b: float, grid: int = 10_000,
               reserve: float = 1.2) -> float:
    """L = reserve * max |dW/dx| по сетке из grid шагов (как в Excel)."""
    xs = np.linspace(a, b, grid + 1)
    if np.any(np.diff(xs) <= 0):
        raise ValueError("отрезок слишком короткий: узлы сетки совпадают в double")
    ys = W(xs)
    check_finite(xs, ys)
    slopes = np.abs(np.diff(ys) / np.diff(xs))

    # Если на сетке в 10 раз реже наклон сильно меньше, оценка ненадёжна.
    coarse = np.abs(np.diff(ys[::10]) / np.diff(xs[::10])).max()
    if slopes.max() > 1.5 * coarse:
        warnings.warn(
            f"оценка L зависит от шага сетки ({coarse:g} -> {slopes.max():g}): "
            "функция может быть не липшицевой или сетка слишком грубая"
        )
    return reserve * float(slopes.max())


def _violates(x1: float, z1: float, x2: float, z2: float, L: float) -> bool:
    """|z2 - z1| > L|x2 - x1| с допуском на округление."""
    return abs(z2 - z1) > L * abs(x2 - x1) + 1e-12 * (1 + abs(z1) + abs(z2))


def minimize(function: str, a: float, b: float, eps: float,
             L: float | None = None, mode: str = "min",
             max_iter: int = 10_000, grid: int = 10_000,
             reserve: float = 1.2) -> Result:
    """Глобальный экстремум W на [a, b] методом ломаных.

    Args:
        function: Функция от x, например "x + sin(3.14159*x)".
        a, b: Отрезок, a < b.
        eps: Точность по значению функции.
        L: Константа Липшица; если None — оценивается по сетке.
        mode: "min" или "max".
        max_iter: Наибольший номер итерации.
        grid: Число шагов сетки для оценки L.
        reserve: Запас при оценке L.

    Raises:
        ValueError: Некорректный ввод, W не конечна или заданная L мала.
    """
    start = time.perf_counter()
    a, b, eps = float(a), float(b), float(eps)
    check_parameters(a, b, eps, L, mode, max_iter, grid, reserve)

    sign = 1 if mode == "min" else -1  # максимум W = минимум -W
    user_W = parse_function(function)

    def W(x):
        return sign * user_W(x)

    adaptive = L is None
    if adaptive:
        # eps / (b - a) — чтобы L не была 0 у постоянной функции.
        L = max(estimate_L(W, a, b, grid, reserve), eps / (b - a))

    xs, zs = np.array([a, b]), W(np.array([a, b]))  # итерации 0 и 1
    check_finite(xs, zs)
    order = [(a, zs[0]), (b, zs[1])]

    def check_lipschitz(i: int) -> bool:
        """Проверяет соседние точки i, i+1. True, если L увеличена."""
        nonlocal L
        if not _violates(xs[i], zs[i], xs[i + 1], zs[i + 1], L):
            return False
        slope = abs(zs[i + 1] - zs[i]) / (xs[i + 1] - xs[i])
        message = (f"между x = {xs[i]:g} и x = {xs[i + 1]:g} наклон {slope:g} "
                   f"больше L = {L:g}: условие Липшица нарушено")
        if not adaptive:
            raise ValueError(message + ". Увеличьте L или не задавайте её.")
        L = reserve * slope
        warnings.warn(message + f", L увеличена до {L:g}")
        return True

    check_lipschitz(0)
    status = "max_iter"
    for _ in range(2, max_iter + 1):
        # Нижние вершины ломаной между соседними точками.
        lows = (zs[:-1] + zs[1:]) / 2 - L * np.diff(xs) / 2
        i = int(np.argmin(lows))
        lower = lows[i]  # min p_{n-1}
        u = (xs[i] + xs[i + 1]) / 2 + (zs[i] - zs[i + 1]) / (2 * L)

        inside = xs[i] < u < xs[i + 1]
        if inside:
            z = float(W(u))
            check_finite(u, z)
            xs, zs = np.insert(xs, i + 1, u), np.insert(zs, i + 1, z)
            order.append((u, z))
            if any([check_lipschitz(i), check_lipschitz(i + 1)]):
                continue  # L изменилась — ломаная пересчитывается
        else:
            # Вершина совпала с уже вычисленной точкой.
            u, z = (xs[i], zs[i]) if u <= xs[i] else (xs[i + 1], zs[i + 1])

        if z - lower < eps:
            status = "eps"
            break
        if not inside:
            status = "float_limit"
            break

    if status != "eps":
        warnings.warn(f"точность eps не достигнута ({status}), "
                      f"результат может быть хуже на W(x*) - min p")

    best = int(np.argmin(zs))  # лучшая из вычисленных точек
    return Result(
        x=float(xs[best]),
        value=float(sign * zs[best]),
        gap=float(zs[best] - lower),
        L=float(L),
        iterations=len(order) - 1,
        seconds=time.perf_counter() - start,
        status=status,
        points=np.array(order) * [1, sign],
        function=function,
        mode=mode,
    )


def summary(result: Result) -> str:
    """Результат текстом."""
    word = "минимум" if result.mode == "min" else "максимум"
    return "\n".join([
        f"Функция: {result.function}",
        f"Глобальный {word}: x* = {result.x:.6f}, W(x*) = {result.value:.6f}",
        f"Гарантированная погрешность по значению: {result.gap:.2e}",
        f"Константа Липшица: L = {result.L:.6g}",
        f"Итераций: {result.iterations} (вычислений W: {result.iterations + 1})",
        f"Время: {result.seconds * 1000:.1f} мс, статус: {result.status}",
    ])
