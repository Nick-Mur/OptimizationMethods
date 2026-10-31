"""Условный экстремум методом множителей Лагранжа.

Запуск: python main.py "x + y + 2*z" "x^2 + y^2 + z^2 = 5" "x*z = 0"
"""

import multiprocessing
import sys

import sympy as sp

from lagrange import Solution, is_zero, solve

TIME_LIMIT = 60  # секунд на решение задачи

USAGE = """\
Использование: python main.py "ФУНКЦИЯ" "ОГРАНИЧЕНИЕ" ["ОГРАНИЧЕНИЕ" ...]
Пример:        python main.py "x + y + 2*z" "x^2 + y^2 + z^2 = 5" "x*z = 0"
Порядок переменных можно задать заголовком: "f(x, y, z, w) = 2*w + x + y + z"."""

VERDICTS = {
    "min": "строгий локальный минимум",
    "max": "строгий локальный максимум",
    "saddle": "не экстремум: d²L на касательном подпространстве знакопеременен",
    "degenerate": "условия второго порядка не дают ответа, нужно исследование вручную",
    "isolated": "изолированная точка допустимого множества",
    "irregular": "нерегулярная точка: метод Лагранжа её не классифицирует",
}


def text(expr) -> str:
    """Запись выражения в том же виде, что и ввод: степень через ^."""
    return str(expr).replace("**", "^")


def number(value: sp.Expr) -> str:
    """Точное значение и, если оно не рациональное, приближённое."""
    if value.is_Rational:
        return text(value)
    return f"{text(value)} ≈ {float(sp.re(sp.N(value))):.6g}"


def report(solution: Solution) -> str:
    """Текст решения: постановка, система, найденные точки и вывод о глобальных экстремумах."""
    s = solution
    names = ", ".join(map(str, s.variables))
    lagrangian = " + ".join([text(s.f), *(f"λ{i}*({text(gi)})" for i, gi in enumerate(s.g, 1))])
    lines = [f"f({names}) = {text(s.f)}",
             *(f"g{i} = {text(gi)} = 0" for i, gi in enumerate(s.g, 1)),
             f"L = {lagrangian}",
             "Система ∂L/∂x = 0, g = 0:",
             *(f"  {text(e)} = 0" for e in s.equations),
             ""]

    if not s.points:
        lines.append("Допустимое множество пусто." if s.bounded else
                     "Подозрительных точек нет: допустимое множество пусто или f на нём экстремумов не имеет.")
        return "\n".join(lines)

    lines.append(f"Точки ({names}), подозрительные на экстремум, по возрастанию f:")
    for p in s.points:
        lam = "" if p.multipliers is None else f", λ = ({', '.join(map(text, p.multipliers))})"
        lines.append(f"  ({', '.join(map(text, p.coords))}): f = {number(p.value)}{lam}")
        lines.append(f"      {VERDICTS[p.kind]}")

    if s.bounded:
        lines += ["", "Допустимое множество замкнуто и ограничено. По теореме Вейерштрасса f достигает на нём",
                  "наибольшего и наименьшего значений, и только в найденных точках:"]
        for name, best in ("минимум", s.points[0]), ("максимум", s.points[-1]):
            where = [p.coords for p in s.points if is_zero(p.value - best.value)]
            lines.append(f"  глобальный {name} f = {number(best.value)} в "
                         + ", ".join(f"({', '.join(map(text, c))})" for c in where))
    else:
        lines += ["", "Ограниченность допустимого множества не установлена, поэтому вывод",
                  "о глобальных экстремумах делается отдельно."]
    return "\n".join(lines)


def solve_in_time(f_text: str, g_texts: list[str]) -> Solution:
    """Решает задачу в отдельном процессе и прерывает его по истечении TIME_LIMIT."""
    with multiprocessing.Pool(1) as pool:
        try:
            return pool.apply_async(solve, (f_text, g_texts)).get(TIME_LIMIT)
        except multiprocessing.TimeoutError:
            raise ValueError(f"задача не решена за {TIME_LIMIT} с: система, по-видимому, "
                             "не решается аналитически") from None


def main(argv: list[str] | None = None) -> int:
    """Точка входа: аргументы командной строки -> текст решения. Код выхода 2 — ошибка."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    try:
        solution = solve_in_time(argv[0], argv[1:])
    except ValueError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # сбой внутри SymPy на допустимом вводе
        print(f"Ошибка: SymPy не справился с задачей ({type(error).__name__}: {error})", file=sys.stderr)
        return 2
    print(report(solution))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
