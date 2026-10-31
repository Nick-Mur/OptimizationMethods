"""Разбор условия задачи: строки -> выражения SymPy."""

import ast
import re

import sympy as sp

CONSTANTS = {"pi": sp.pi, "e": sp.E}
MAX_LENGTH = 500  # символов в одном выражении
MAX_POWER = 100   # наибольший показатель степени

HINT = "допустимы числа, переменные, знаки + - * / ^ и скобки; умножение пишется явно: 2*x"

# Заголовок целевой функции: «f(x, y) =» или «f =».
HEADER = re.compile(r"^\s*[^\W\d]\w*\s*(?:\(([^()]*)\))?\s*=(?!=)")

# Символы, которые появляются при копировании формул из PDF.
TYPOGRAPHY = str.maketrans({"−": "-", "–": "-", "·": "*", "×": "*", "²": "^2", "³": "^3"})

ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load,
           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)


def _source(expression: str, text: str) -> tuple[str, set[str]]:
    """Проверяет запись выражения.

    Args:
        expression: Проверяемое выражение.
        text: Исходная запись для сообщений об ошибках.

    Returns:
        Текст в синтаксисе Python и множество имён переменных.

    Raises:
        ValueError: Запись недопустима.
    """
    source = expression.translate(TYPOGRAPHY).replace("^", "**").strip()
    if not source:
        raise ValueError("пустое выражение")
    if len(source) > MAX_LENGTH:
        raise ValueError(f"выражение длиннее {MAX_LENGTH} символов")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError):
        raise ValueError(f"не удалось разобрать «{text}»: {HINT}") from None

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED) or (
                isinstance(node, ast.Constant) and type(node.value) not in (int, float)):
            raise ValueError(f"недопустимый элемент «{ast.unparse(node)}» в «{text}»: {HINT}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            power = node.right
            if not (isinstance(power, ast.Constant) and type(power.value) is int
                    and power.value <= MAX_POWER):
                raise ValueError(f"в «{text}» показатель степени должен быть целым числом "
                                 f"от 0 до {MAX_POWER}")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return source, names - CONSTANTS.keys()


def _zero_form(text: str) -> str:
    """Ограничение «левая = правая» -> выражение «(левая) - (правая)»."""
    sides = text.split("=")
    if len(sides) != 2 or not all(side.strip() for side in sides) or re.search(r"[<>!≤≥≠]", text):
        raise ValueError(f"ограничение «{text}» должно быть равенством «левая часть = правая часть»")
    return f"({sides[0]}) - ({sides[1]})"


def _declared_variables(header: re.Match | None) -> list[str] | None:
    """Переменные из заголовка «f(x, y) =»; None, если они не указаны."""
    if header is None or not header.group(1) or not header.group(1).strip():
        return None
    names = [name.strip() for name in header.group(1).split(",")]
    if not all(name.isidentifier() and name not in CONSTANTS for name in names):
        raise ValueError(f"в заголовке «{header.group(0)}» должны быть перечислены имена переменных")
    if len(set(names)) != len(names):
        raise ValueError(f"в заголовке «{header.group(0)}» переменные повторяются")
    return names


def parse_problem(f_text: str, g_texts: list[str]) -> tuple[sp.Expr, list[sp.Expr], list[sp.Symbol]]:
    """Преобразует условие задачи в выражения SymPy.

    Порядок переменных берётся из заголовка «f(x, y, z) =», при его отсутствии — алфавитный.
    Целевая функция и ограничения должны быть многочленами.

    Args:
        f_text: Целевая функция, например "x + y + 2*z" или "f(x, y, z) = x + y + 2*z".
        g_texts: Ограничения-равенства, например ["x^2 + y^2 + z^2 = 5", "x*z = 0"].

    Returns:
        Целевая функция f, левые части ограничений g_i(x) = 0 и список переменных.

    Raises:
        ValueError: Условие записано некорректно; причина указана в сообщении.
    """
    if not g_texts:
        raise ValueError("нужно хотя бы одно ограничение-равенство")
    header = HEADER.match(f_text)
    f_body = f_text[header.end():] if header else f_text
    if "=" in f_body:
        raise ValueError("целевая функция записывается без знака «=» (допустим заголовок «f(x, y) = ...»)")

    checked = [_source(f_body, f_text)] + [_source(_zero_form(text), text) for text in g_texts]
    used = set().union(*(names for _, names in checked))
    names = _declared_variables(header) or sorted(used)
    if undeclared := used - set(names):
        raise ValueError(f"в заголовке «{header.group(0)}» не указаны переменные: {', '.join(sorted(undeclared))}")
    if not names:
        raise ValueError("в условии нет ни одной переменной")
    if len(g_texts) > len(names):
        raise ValueError(f"ограничений ({len(g_texts)}) больше, чем переменных ({len(names)})")

    variables = [sp.Symbol(name, real=True) for name in names]
    scope = {**CONSTANTS, **dict(zip(names, variables))}
    f, *g = [sp.sympify(source, locals=scope, rational=True) for source, _ in checked]

    for text, expr in zip([f_text, *g_texts], [f, *g]):
        if expr.has(sp.zoo, sp.nan):
            raise ValueError(f"в «{text}» деление на ноль")
        if not expr.is_polynomial(*variables):
            raise ValueError(f"«{text}» не является многочленом (деление на переменную)")
    if not f.free_symbols:
        raise ValueError("целевая функция постоянна: любая допустимая точка является нестрогим экстремумом")
    for text, gi in zip(g_texts, g):
        if not gi.free_symbols:
            raise ValueError(f"ограничение «{text}» не зависит от переменных")
    return f, g, variables
