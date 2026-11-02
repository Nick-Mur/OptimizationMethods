import matplotlib

matplotlib.use("Agg")  # до импорта pyplot

import numpy as np
import pytest

from main import main
from piyavsky import estimate_L, minimize, parse_function, parse_number
from piyavsky.plotting import plot

EXCEL = "0.5*x+0.2*cos(3.14*x)+0.2*sin(2*3.14*x)"
EXCEL_L = 2.1542997412292588  # ячейка F30 файла 05_Piyavsky.xlsm
RASTRIGIN = "10 + x^2 - 10*cos(2*pi*x)"
ACKLEY = "-20*exp(-0.2*sqrt(x^2)) - exp(cos(2*pi*x)) + 20 + e"


def true_extremum(function, a, b, mode="min"):
    xs = np.linspace(a, b, 2_000_001)
    ys = parse_function(function)(xs)
    return ys.min() if mode == "min" else ys.max()


# --- Метод -------------------------------------------------------------

def test_s1_excel_points():
    """S1: точки совпадают с Excel (итерации 2-8)."""
    r = minimize(EXCEL, -2, 1, 0.01, L=EXCEL_L)
    excel = [-0.75486009969835521, -1.4883294615403926, -0.021390737856317787,
             -1.2252623006071657, -1.7513966224736197, -1.31159378504,
             -1.13893081617]
    # С итерации 5 есть равные нижние вершины, выбор между ними в Excel другой.
    assert r.points[2:5, 0] == pytest.approx(excel[:3], abs=1e-9)
    assert sorted(r.points[2:9, 0]) == pytest.approx(sorted(excel), abs=1e-9)


def test_s2_excel_lipschitz_estimate():
    """S2: оценка L совпадает с Excel."""
    assert estimate_L(parse_function(EXCEL), -2, 1) == pytest.approx(EXCEL_L, rel=1e-9)


@pytest.mark.parametrize("function, a, b", [
    (EXCEL, -2, 1),
    (RASTRIGIN, -5.12, 5.12),       # 11 локальных минимумов
    (ACKLEY, -5, 5),                # излом в глобальном минимуме
    ("x + sin(3.14159*x)", -2, 2),  # пример из задания
    ("abs(x - 0.3)", -1, 1),        # негладкая липшицева функция
    ("x", 0, 1),                    # минимум на левом конце
    ("-x", 0, 1),                   # минимум на правом конце
    ("cos(x)", 0, 4 * np.pi),       # два равных глобальных минимума
    ("5", 0, 1),                    # константа
])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_s3_accuracy_guarantee(function, a, b, mode):
    """S3: 0 <= W(x*) - W* <= gap < eps."""
    eps = 0.01
    r = minimize(function, a, b, eps, mode=mode)
    best = true_extremum(function, a, b, mode)
    sign = 1 if mode == "min" else -1
    assert r.status == "eps"
    assert 0 <= sign * (r.value - best) < eps
    assert sign * (r.value - best) <= r.gap + 1e-12 < eps
    assert parse_function(function)(r.x) == pytest.approx(r.value)


def test_s4_stopping_rule_from_lecture():
    """S4: при большом eps остановка на итерации 2."""
    r = minimize(RASTRIGIN, -5.12, 5.12, eps=1000)
    assert r.status == "eps" and r.iterations == 2


def test_s5_larger_L_is_still_correct_but_slower():
    """S5: завышенная L — ответ тот же, итераций больше."""
    small = minimize(EXCEL, -2, 1, 0.01, L=EXCEL_L)
    large = minimize(EXCEL, -2, 1, 0.01, L=10 * EXCEL_L)
    assert large.value == pytest.approx(small.value, abs=0.01)
    assert large.iterations > small.iterations


# --- Константа Липшица --------------------------------------------------

def test_s6_given_L_too_small_is_an_error():
    """S6: заданная L мала -> ошибка."""
    with pytest.raises(ValueError, match="условие Липшица нарушено"):
        minimize(RASTRIGIN, -5.12, 5.12, 0.01, L=1)


def test_s7_auto_L_grows_when_grid_missed_a_well():
    """S7: сетка пропустила узкую яму -> L увеличивается."""
    f = "-exp(-((x-0.505)/0.003)^2)"
    with pytest.warns(UserWarning, match="L увеличена"):
        r = minimize(f, 0, 1, 0.01, grid=100)
    assert r.value == pytest.approx(-1, abs=0.01)


def test_s8_non_lipschitz_function_warns():
    """S8: sqrt(x) не липшицева -> предупреждение."""
    with pytest.warns(UserWarning, match="не липшицевой"):
        minimize("sqrt(x)", 0, 1, 0.01)


# --- Остановка без достижения eps ---------------------------------------

def test_s9_max_iter():
    """S9: лимит итераций -> max_iter, gap остаётся верным."""
    with pytest.warns(UserWarning, match="max_iter"):
        r = minimize(RASTRIGIN, -5.12, 5.12, 1e-9, max_iter=5)
    assert r.status == "max_iter" and r.iterations == 5
    assert r.value - r.gap <= true_extremum(RASTRIGIN, -5.12, 5.12) + 1e-12


@pytest.mark.parametrize("max_iter", [33, 65, 128])
def test_s9_gap_after_L_increase(max_iter):
    """S9: остановка по лимиту сразу после увеличения L -> gap всё равно верный."""
    f = "-exp(-((x-0.505)/0.003)^2)"
    with pytest.warns(UserWarning):
        r = minimize(f, 0, 1, 0.01, grid=100, max_iter=max_iter)
    assert r.status == "max_iter"
    assert r.value - r.gap <= true_extremum(f, 0, 1) + 1e-12


def test_s10_float_limit():
    """S10: eps меньше точности double -> float_limit."""
    with pytest.warns(UserWarning, match="float_limit"):
        r = minimize("abs(x - 1/3)", 0, 1, 1e-30, L=2)
    assert r.status == "float_limit"
    assert r.x == pytest.approx(1 / 3, abs=1e-15)


# --- Входные данные ------------------------------------------------------

@pytest.mark.parametrize("text, x, expected", [
    ("f(x) = x^2", 3, 9),
    ("W(x)=2*x", 3, 6),
    ("y = x", 3, 3),
    ("e^x", 0, 1),
    ("max(x, 1) + min(x, 0)", 3, 3),
    ("ln(x) + log10(x)", 1, 0),
    ("7", 3, 7),
])
def test_s11_function_formats(text, x, expected):
    """S11: формы записи функции."""
    assert parse_function(text)(x) == pytest.approx(expected)
    assert parse_function(text)(np.zeros(4)).shape == (4,)


@pytest.mark.parametrize("text, message", [
    ("", "пустое"),
    ("f(x) =", "пустое"),
    ("x +", "разобрать"),
    ("2x", "умножение пишется явно"),
    ("y + x", "неизвестное имя «y»"),
    ("foo(x)", "неизвестная функция «foo»"),
    ("x.real", "недопустимая конструкция"),
    ("__import__('os')", "неизвестная функция"),
    ("x > 1", "недопустимая конструкция"),
    ("x if x else 1", "недопустимая конструкция"),
    ("'abc'", "недопустимая константа"),
    ("2j*x", "недопустимая константа"),
    ("sin + x", "не удалось вычислить"),
    ("max(x)", "не удалось вычислить"),
    ("(-8)**(1/3)", "не удалось вычислить"),
    ("1/0", "не удалось вычислить"),
])
def test_s12_bad_function_text(text, message):
    """S12: некорректная строка функции -> ошибка."""
    with pytest.raises(ValueError, match=message):
        minimize(text, 0, 1, 0.01)


@pytest.mark.parametrize("text, a, b", [
    ("log(x)", -1, 1),     # не определена при x <= 0
    ("sqrt(x)", -1, 1),    # не определена при x < 0
    ("1/x", -1, 1),        # бесконечна в 0
    ("exp(1000*x)", 0, 1), # переполнение double
])
def test_s13_function_not_finite_on_segment(text, a, b):
    """S13: W не конечна на отрезке -> ошибка."""
    with pytest.raises(ValueError, match="не конечна"):
        minimize(text, a, b, 0.01)


@pytest.mark.parametrize("kwargs, message", [
    (dict(a=1, b=1), "a < b"),
    (dict(a=1, b=0), "a < b"),
    (dict(a=float("nan")), "конечными"),
    (dict(b=float("inf")), "конечными"),
    (dict(a=-1e308, b=1e308), "длина отрезка"),
    (dict(a=1, b=1 + 1e-15), "слишком короткий"),
    (dict(eps=0), "eps > 0"),
    (dict(eps=-0.1), "eps > 0"),
    (dict(eps=float("nan")), "eps > 0"),
    (dict(eps=float("inf")), "eps > 0"),
    (dict(L=0), "> 0"),
    (dict(L=-1), "> 0"),
    (dict(L=float("inf")), "> 0"),
    (dict(L=float("nan")), "> 0"),
    (dict(mode="extremum"), "mode"),
    (dict(max_iter=1), "max_iter"),
    (dict(max_iter=2.5), "max_iter"),
    (dict(grid=5), "grid"),
    (dict(reserve=0.5), "reserve"),
])
def test_s14_bad_parameters(kwargs, message):
    """S14: некорректные параметры -> ошибка."""
    params = dict(function="x^2", a=-1, b=1, eps=0.01) | kwargs
    with pytest.raises(ValueError, match=message):
        minimize(**params)


@pytest.mark.parametrize("text, expected", [
    ("0,01", 0.01), ("0.01", 0.01), ("-2", -2), ("1e-3", 1e-3), ("-2*pi", -2 * np.pi),
])
def test_s15_numbers(text, expected):
    """S15: числа с запятой и выражения с pi, e."""
    assert parse_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "inf", "1e400", "x", "1/0"])
def test_s15_bad_numbers(text):
    with pytest.raises(ValueError):
        parse_number(text)


# --- Интерфейс ----------------------------------------------------------

def test_s16_cli(tmp_path, capsys):
    """S16: командная строка печатает ответ и сохраняет график."""
    picture = tmp_path / "plot.png"
    code = main([RASTRIGIN, "-5,12", "5,12", "0,01", "--plot", str(picture)])
    assert code == 0 and picture.stat().st_size > 0
    assert "Глобальный минимум" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["x^2", "1", "0", "0.01"],           # a > b
    ["x^2", "-1", "1", "0"],             # eps = 0
    ["foo(x)", "-1", "1", "0.01"],       # неизвестная функция
    ["x^2", "-1", "1", "0.01", "--L", "abc"],
])
def test_s16_cli_errors(argv, capsys):
    """S16: ошибка ввода -> код выхода 2."""
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_s17_plot():
    """S17: график любой итерации и для максимума."""
    r = minimize(EXCEL, -2, 1, 0.01, mode="max")
    plot(r, n=1)
    plot(r)
    with pytest.raises(ValueError):
        plot(r, n=r.iterations + 1)


def test_s9_gap_nonnegative_after_last_L_increase():
    """S9: L выросла на последней разрешённой итерации -> gap пересчитан, не < 0."""
    f = "x-exp(-((x-1/12)/0.000001)^2)"
    with pytest.warns(UserWarning):
        r = minimize(f, 0, 1, 0.01, max_iter=2)
    assert r.status == "max_iter" and r.L > 1.2
    assert r.gap >= 0
    assert r.value - r.gap <= true_extremum(f, 0, 1) + 1e-12
