"""Тесты разбора условия и командной строки."""

import pytest

import main
from lagrange import parse_problem


@pytest.mark.parametrize("f, g, message", [
    ("x", [], "хотя бы одно ограничение"),
    ("", ["x = 1"], "пустое выражение"),
    ("x", ["x^2 + y^2"], "должно быть равенством"),
    ("x", ["x^2 + y^2 <= 1"], "должно быть равенством"),
    ("x", ["x ≥ 1"], "должно быть равенством"),
    ("x", ["x == 1"], "должно быть равенством"),
    ("x", ["x = 1 = 2"], "должно быть равенством"),
    ("x", ["x = "], "должно быть равенством"),
    ("x + y = 3", ["x = 1"], "без знака «=»"),
    ("2x", ["x = 1"], "умножение пишется явно"),
    ("0,5*x", ["x = 1"], "недопустимый элемент"),
    ("sin(x)", ["x = 1"], "недопустимый элемент"),
    ("__import__('os').system('dir')", ["x = 1"], "недопустимый элемент"),
    ("x", ["'a' = 1"], "недопустимый элемент"),
    ("x", ["True + x = 1"], "недопустимый элемент"),
    ("1j*x", ["x = 1"], "недопустимый элемент"),
    ("x^0.5", ["x = 1"], "показатель степени"),
    ("x^y", ["x = 1"], "показатель степени"),
    ("x + 2^10^10", ["x = 1"], "показатель степени"),
    ("x^1000", ["x = 1"], "показатель степени"),
    ("+" * 600 + "x", ["x = 1"], "длиннее"),
    ("x/0", ["x = 1"], "деление на ноль"),
    ("1/x", ["x = 1"], "не является многочленом"),
    ("5", ["x = 1"], "постоянна"),
    ("x", ["1 = 1"], "не зависит от переменных"),
    ("x", ["x = x"], "не зависит от переменных"),
    ("x + y", ["x = 1", "y = 2", "x + y = 3"], "больше, чем переменных"),
    ("f(x, y) = x + z", ["x = 1"], "не указаны переменные: z"),
    ("f(x, x) = x", ["x = 1"], "повторяются"),
])
def test_bad_input_is_rejected(f, g, message):
    with pytest.raises(ValueError, match=message):
        parse_problem(f, g)


def test_good_input():
    """Заголовок задаёт порядок переменных; допустимы ^, pi, e, деление на число и символы из PDF."""
    f, g, variables = parse_problem("f(y, x) = x/2 + pi*y", ["x² + y² = e", "x − y = 0.5"])
    assert [str(v) for v in variables] == ["y", "x"] and len(g) == 2


def run(capsys, *argv):
    """Запускает программу; возвращает код выхода, stdout и stderr."""
    code = main.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_cli_variant_20(capsys):
    code, out, _ = run(capsys, "x + y + 2*z", "x^2 + y^2 + z^2 = 5", "x*z = 0")
    assert code == 0
    assert "глобальный минимум f = -5 в (0, -1, -2)" in out
    assert "глобальный максимум f = 5 в (0, 1, 2)" in out
    assert out.count("нерегулярная точка") == 2


def test_cli_leading_minus(capsys):
    """Функция и ограничение, начинающиеся с минуса, не принимаются за ключи."""
    code, out, _ = run(capsys, "-x", "-x^2 - y^2 = -1")
    assert code == 0 and "глобальный максимум f = 1 в (-1, 0)" in out


def test_cli_empty_set(capsys):
    assert "Допустимое множество пусто" in run(capsys, "x + y", "x^2 + y^2 = -1")[1]


def test_cli_help_and_errors(capsys):
    assert run(capsys, "--help")[0] == 0
    assert run(capsys)[0] == 2
    code, _, err = run(capsys, "x", "x^2 <= 1")
    assert code == 2 and "равенством" in err


def test_cli_time_limit(capsys, monkeypatch):
    monkeypatch.setattr(main, "TIME_LIMIT", 0.01)
    code, _, err = run(capsys, "x", "x^2 + y^2 = 1")
    assert code == 2 and "не решена за" in err
