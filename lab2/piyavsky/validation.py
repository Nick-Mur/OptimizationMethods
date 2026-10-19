"""Разбор и проверка входных данных."""

import ast
import math
import re
from typing import Callable

import numpy as np

FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "min": np.minimum, "max": np.maximum,
}
CONSTANTS = {"pi": math.pi, "e": math.e}

# Допустимые узлы выражения; всё остальное запрещено.
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub,
)

# Левая часть "f(x) =", "y =" отбрасывается.
PREFIX = re.compile(r"^\s*[A-Za-z]\w*\s*(\(\s*x\s*\))?\s*=(?!=)")


def _compile(text: str, variables: tuple[str, ...]):
    """Проверяет выражение по белому списку и компилирует его."""
    source = text.replace("^", "**").strip()
    if not source:
        raise ValueError("пустое выражение")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        raise ValueError(
            f"не удалось разобрать «{text}» (умножение пишется явно: 2*x)"
        ) from None

    names = {*variables, *CONSTANTS, *FUNCTIONS}
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"недопустимая конструкция «{ast.unparse(node)}»")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS
        ):
            raise ValueError(
                f"неизвестная функция «{ast.unparse(node.func)}», "
                f"доступны: {', '.join(FUNCTIONS)}"
            )
        if isinstance(node, ast.Name) and node.id not in names:
            allowed = ", ".join([*variables, *CONSTANTS])
            raise ValueError(f"неизвестное имя «{node.id}», допустимы: {allowed}")
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            raise ValueError(f"недопустимая константа {node.value!r}")
    return compile(tree, "<expr>", "eval")


def _evaluate(code, text: str, **variables) -> np.ndarray:
    # Вне области определения numpy даёт nan/inf, их ловит check_finite.
    with np.errstate(all="ignore"):
        try:
            scope = {**FUNCTIONS, **CONSTANTS, **variables}
            return np.asarray(eval(code, {"__builtins__": {}}, scope), dtype=float)
        except Exception as error:
            raise ValueError(f"не удалось вычислить «{text}»: {error}") from None


def parse_function(text: str) -> Callable[[np.ndarray], np.ndarray]:
    """Строка "f(x) = ..." -> функция от массива x."""
    code = _compile(PREFIX.sub("", text, count=1), ("x",))

    def W(x):
        x = np.asarray(x, dtype=float)
        # broadcast_to — на случай константы.
        return np.broadcast_to(_evaluate(code, text, x=x), x.shape)

    return W


def parse_number(text: str) -> float:
    """Число из строки: "0,01", "-2*pi", "1e-3"."""
    text = str(text)
    value = float(_evaluate(_compile(text.replace(",", "."), ()), text))
    if not math.isfinite(value):
        raise ValueError(f"«{text}» не является конечным числом")
    return value


def check_finite(x, z) -> None:
    """Ошибка, если среди значений W есть nan или inf."""
    bad = ~np.isfinite(z)
    if np.any(bad):
        x0, z0 = np.atleast_1d(x)[bad][0], np.atleast_1d(z)[bad][0]
        raise ValueError(
            f"W({x0:g}) = {z0}: функция не определена или не конечна "
            f"на отрезке, значит она не липшицева"
        )


def check_parameters(a: float, b: float, eps: float, L: float | None,
                     mode: str, max_iter: int, grid: int,
                     reserve: float) -> None:
    """Ошибка, если параметры метода некорректны."""
    if mode not in ("min", "max"):
        raise ValueError(f"mode должен быть 'min' или 'max', получено {mode!r}")
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("концы отрезка должны быть конечными числами")
    if not a < b:
        raise ValueError(f"нужно a < b, получено a = {a:g}, b = {b:g}")
    if not math.isfinite(b - a):
        raise ValueError("длина отрезка не помещается в double")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"нужно eps > 0, получено {eps:g}")
    if L is not None and not (math.isfinite(L) and L > 0):
        raise ValueError(f"константа Липшица должна быть > 0, получено {L:g}")
    if int(max_iter) != max_iter or max_iter < 2:
        raise ValueError("max_iter должно быть целым числом >= 2")
    if int(grid) != grid or grid < 10:
        raise ValueError("grid должно быть целым числом >= 10")
    if not reserve >= 1:
        raise ValueError("reserve должен быть >= 1")
