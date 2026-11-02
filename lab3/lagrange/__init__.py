"""Условный экстремум методом множителей Лагранжа."""

from .method import Point, Solution, find_extrema, is_zero
from .parsing import parse_problem

__all__ = ["Point", "Solution", "find_extrema", "is_zero", "parse_problem", "solve"]


def solve(f_text: str, g_texts: list[str]) -> Solution:
    """Решает задачу, заданную строками.

    Args:
        f_text: Целевая функция, например "x + y + 2*z".
        g_texts: Ограничения-равенства, например ["x^2 + y^2 + z^2 = 5", "x*z = 0"].

    Returns:
        Решение задачи.

    Raises:
        ValueError: Некорректное условие или задача не решается методом (причина в сообщении).
    """
    # Преобразовать текст условия в символьные выражения и передать их методу.
    return find_extrema(*parse_problem(f_text, g_texts))
