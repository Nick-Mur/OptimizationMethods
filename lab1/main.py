"""Двухфазный симплекс-метод в формате лекционных таблиц.

Решает задачу c·x -> min (или max) при Ax {<=, =, >=} b и x >= 0.

Все вычисления идут в точных дробях (Fraction), поэтому таблицы можно
сравнивать с ручным решением без ошибок округления.

Ограничения: переменные считаются неотрицательными.
"""

from fractions import Fraction as F
from typing import NamedTuple

# При умножении неравенства на -1 его знак меняется на противоположный.
FLIP = {"<=": ">=", ">=": "<=", "=": "="}


class Result(NamedTuple):
    """Итог решения.

    Attributes:
        status: "optimal" (оптимум найден), "infeasible" (ограничения
            противоречат друг другу) или "unbounded" (цель можно улучшать
            бесконечно).
        x: Оптимальная точка (исходные переменные). Пусто, если оптимума нет.
        value: Значение целевой функции в x для исходного направления
            (min или max). None, если оптимума нет.
    """

    status: str
    x: list[F]
    value: F | None


def fmt(value: F) -> str:
    return str(value)


def to_fraction(value) -> F:
    """Переводит число в точную дробь.

    float берётся по десятичной записи (0.1 станет 1/10, а не двоичным
    приближением), иначе сравнения с нулём в таблицах работали бы неточно.

    Raises:
        ValueError: если значение нельзя прочитать как число.
    """
    if isinstance(value, float):
        value = str(value)
    try:
        return F(value)
    except (ValueError, TypeError, ZeroDivisionError):
        raise ValueError(f"не удалось прочитать число: {value!r}") from None


def index(name: str) -> int:
    """Номер переменной по её имени: "x7" -> 7."""
    return int(name[1:])


class Tableau:
    """Симплекс-таблица

    Каждая строка читается как уравнение: базисная переменная плюс сумма
    (коэффициент * свободная переменная) равна b. Свободные переменные
    считаются нулями, поэтому в текущей точке базисная переменная равна
    своему b.

    Attributes:
        cols: Имена свободных переменных (подписи столбцов).
        rows: Имена базисных переменных (подписи строк).
        M: Матрица из m + 1 строки и n + 1 столбца. Первые m строк это
            ограничения, последний столбец это правые части b. Последняя
            строка относится к целевой функции: в ней стоят оценки p_j
            (на сколько изменится цель, если увеличить x_j на 1), а в
            правом нижнем углу лежит -Q, где Q текущее значение цели.
        art: Имена искусственных переменных. После выхода из базиса они
            не нужны, и их столбцы удаляются.
        verbose: Печатать ли таблицы.
    """

    def __init__(
            self,
            cols: list[str],
            rows: list[str],
            M: list[list[F]],
            art: set[str],
            verbose: bool = True,
    ):
        self.cols = cols
        self.rows = rows
        self.M = M
        self.art = art
        self.verbose = verbose

    def show(self, title: str) -> None:
        """Печатает таблицу с заголовком (если включён verbose)."""
        if not self.verbose:
            return

        print(f"\n{title}")
        print(" " * 5 + "".join(f"{name:>7}" for name in self.cols + ["b"]))
        for name, row in zip(self.rows + ["p"], self.M):
            print(f"{name:>4} " + "".join(f"{fmt(value):>7}" for value in row))

    def set_costs(self, cost: dict[str, F]) -> None:
        """Заполняет строку цели для заданных стоимостей переменных.

        Целевая функция выражается через свободные переменные: подставляем
        вместо каждой базисной переменной её выражение из своей строки.
        Получается p_j = c_j - сумма(c_базисной * a_ij) для каждого столбца
        и -Q = -сумма(c_базисной * b_i) в правом нижнем углу.

        Args:
            cost: Стоимость каждой переменной в целевой функции.
                Переменные, которых нет в словаре, считаются стоимостью 0.
        """
        *body, objective = self.M
        for j, name in enumerate(self.cols):
            objective[j] = cost.get(name, F(0)) - sum(
                (cost.get(basic, F(0)) * row[j]
                 for basic, row in zip(self.rows, body)),
                F(0),
            )
        objective[-1] = -sum(
            (cost.get(basic, F(0)) * row[-1]
             for basic, row in zip(self.rows, body)),
            F(0),
        )

    def entering(self, bland: bool = False) -> int | None:
        """Выбирает разрешающий столбец.

        Обычно берётся наименьшая (самая отрицательная) оценка p_j: по ней
        цель убывает быстрее всего. Если таких несколько, берётся первая
        слева. По правилу Бленда (bland=True) берётся отрицательная оценка
        у переменной с наименьшим номером: это защищает от зацикливания.

        Returns:
            Номер столбца или None, если отрицательных оценок нет
            (то есть оптимум достигнут).
        """
        estimates = self.M[-1][:-1]
        negative = [j for j, p in enumerate(estimates) if p < 0]
        if not negative:
            return None
        if bland:
            return min(negative, key=lambda j: index(self.cols[j]))
        return min(negative, key=lambda j: estimates[j])

    def leaving(self, j: int, bland: bool = False) -> int | None:
        """Выбирает разрешающую строку для столбца j.

        Смотрим только на положительные элементы столбца: именно они
        ограничивают рост переменной (базисная переменная доходит до нуля при
        росте на b/a). Берём строку с наименьшим отношением b/a, чтобы после
        шага ни одна переменная не стала отрицательной. При равенстве
        отношений берётся строка выше, а по правилу Бленда (bland=True)
        строка с базисной переменной наименьшего номера.

        Returns:
            Номер строки или None, если в столбце нет положительных
            элементов (переменную можно увеличивать бесконечно).
        """
        ratios = [
            (row[-1] / row[j], index(self.rows[i]) if bland else i, i)
            for i, row in enumerate(self.M[:-1])
            if row[j] > 0
        ]
        return min(ratios)[2] if ratios else None

    def pivot(self, i: int, j: int) -> None:
        """Делает шаг: меняет местами базисную и свободную переменные.

        Пересчёт идёт по формулам лекции, одинаково для всей матрицы
        (включая строку цели и столбец b):
          - обычная клетка: a - a_строки_i * a_столбца_j / a_разр;
          - разрешающая строка: a / a_разр;
          - разрешающий столбец: -a / a_разр;
          - разрешающий элемент: 1 / a_разр.

        Если из базиса вышла искусственная переменная, её столбец удаляется:
        возвращаться в базис ей незачем.

        Args:
            i: Номер разрешающей строки.
            j: Номер разрешающего столбца.
        """
        matrix = self.M
        element = matrix[i][j]
        width = range(len(matrix[0]))

        # Сначала считаем все клетки по общей формуле, затем перезаписываем
        # те, для которых формулы особые (строка, столбец, сам элемент).
        updated = [
            [matrix[r][c] - matrix[i][c] * matrix[r][j] / element for c in width]
            for r in range(len(matrix))
        ]
        updated[i] = [value / element for value in matrix[i]]
        for r in range(len(matrix)):
            updated[r][j] = -matrix[r][j] / element
        updated[i][j] = 1 / element

        self.M = updated
        # Названия меняются местами: в столбце теперь вышедшая переменная,
        # в строке вошедшая.
        self.cols[j], self.rows[i] = self.rows[i], self.cols[j]

        if self.cols[j] in self.art:
            del self.cols[j]
            self.M = [row[:j] + row[j + 1:] for row in self.M]

    def optimize(self, title: str) -> bool:
        """Делает шаги, пока цель можно улучшить, и печатает таблицы.

        Если шаг не уменьшает цель (разрешающая строка с b = 0), следующие
        шаги выбираются по правилу Бленда, пока цель снова не уменьшится.
        Иначе при вырожденности таблицы могли бы повторяться бесконечно.

        Args:
            title: Название этапа для заголовков таблиц.

        Returns:
            True, если оптимум найден; False, если цель неограниченна
            (выбран столбец с отрицательной оценкой, но без положительных
            элементов).
        """
        self.show(f"{title}: таблица 0")
        step = 0
        stalled = False

        while (j := self.entering(stalled)) is not None:
            i = self.leaving(j, stalled)
            if i is None:
                return False
            stalled = self.M[i][-1] == 0

            if self.verbose:
                print(
                    f"\nРазрешающий столбец: {self.cols[j]}, "
                    f"строка: {self.rows[i]}, элемент: {fmt(self.M[i][j])}"
                )

            self.pivot(i, j)
            step += 1
            self.show(f"{title}: таблица {step}")

        return True


def solve(
        c: list,
        A: list[list],
        rel: list[str],
        b: list,
        maximize: bool = False,
        verbose: bool = True,
) -> Result:
    """Решает задачу c·x -> min (max) при Ax {<=, =, >=} b и x >= 0.

    Сначала приводит задачу к каноническому виду, затем решает
    вспомогательную задачу (если нет готового базиса) и после неё основную.
    Числа можно задавать целыми, float, Fraction или строками вида "1/3":
    всё переводится в точные дроби (float по десятичной записи).
    Все переменные неотрицательные (x >= 0), свободные по знаку
    переменные не поддерживаются.

    Args:
        c: Коэффициенты целевой функции.
        A: Матрица ограничений (список строк).
        rel: Знаки ограничений: "<=", "=" или ">=" (по одному на строку A).
        b: Правые части ограничений.
        maximize: True для максимизации, иначе минимизация.
        verbose: Печатать ли симплекс-таблицы.

    Returns:
        Result со статусом, оптимальной точкой и значением цели.

    Raises:
        ValueError: если размеры c, A, rel, b не согласованы, указан
            неизвестный знак ограничения или число нельзя прочитать.
    """
    n = len(c)
    if n == 0:
        raise ValueError("целевая функция пуста: в c нет коэффициентов")
    if not len(A) == len(rel) == len(b):
        raise ValueError(
            f"число строк не совпадает: в A {len(A)}, в rel {len(rel)}, "
            f"в b {len(b)}"
        )
    for k, row in enumerate(A, start=1):
        if len(row) != n:
            raise ValueError(
                f"в строке {k} матрицы A {len(row)} коэффициентов, "
                f"а в c их {n}"
            )
    for k, relation in enumerate(rel, start=1):
        if relation not in FLIP:
            raise ValueError(
                f"неизвестный знак {relation!r} в ограничении {k}: "
                f"допустимы '<=', '=', '>='"
            )

    # Максимум c·x это минимум (-c)·x, поэтому всегда минимизируем,
    # а в конце возвращаем знак обратно.
    sign = -1 if maximize else 1
    names = [f"x{j + 1}" for j in range(n)]
    cost = {name: sign * to_fraction(value) for name, value in zip(names, c)}

    # Правая часть должна быть неотрицательной: иначе умножаем строку на -1
    # (знак неравенства при этом меняется).
    rels = list(rel)
    body = []
    for k, (row, rhs) in enumerate(zip(A, b)):
        row = [to_fraction(value) for value in row]
        rhs = to_fraction(rhs)
        if rhs < 0:
            row = [-value for value in row]
            rhs = -rhs
            rels[k] = FLIP[rels[k]]
        body.append(row + [rhs])

    # Неравенство превращаем в равенство дополнительной переменной:
    # "<=" получает запас (+1), ">=" теряет излишек (-1). Нумерация идёт
    # дальше x1..xn, как в ручном решении.
    for k, relation in enumerate(rels):
        if relation != "=":
            names.append(f"x{len(names) + 1}")
            coefficient = 1 if relation == "<=" else -1
            for i, row in enumerate(body):
                row.insert(-1, F(coefficient if i == k else 0))

    # Строка может сразу получить базисную переменную, если в её столбце
    # стоит 1, а во всех остальных строках 0 (так выходит у запасных
    # переменных "<=").
    basis = {}
    for i in range(len(body)):
        for j in range(len(names)):
            column = [row[j] for row in body]
            if column[i] == 1 and column.count(0) == len(column) - 1:
                basis[i] = j
                break

    # Строкам без готового базиса временно даём искусственную переменную.
    # Нумерация продолжается после всех остальных переменных.
    artificial = set()
    rows = []
    for i in range(len(body)):
        if i in basis:
            rows.append(names[basis[i]])
        else:
            name = f"x{len(names) + len(artificial) + 1}"
            artificial.add(name)
            rows.append(name)

    # В таблице остаются только свободные переменные: столбцы базисных
    # переменных в ней не хранятся (их роль играют подписи строк).
    free = [j for j in range(len(names)) if j not in basis.values()]
    matrix = [[row[j] for j in free] + [row[-1]] for row in body]
    matrix.append([F(0)] * (len(free) + 1))  # строка цели, заполняется ниже
    tableau = Tableau([names[j] for j in free], rows, matrix, artificial, verbose)

    # Вспомогательная задача: минимизируем сумму искусственных переменных.
    if artificial:
        tableau.set_costs({name: F(1) for name in artificial})
        tableau.optimize("Вспомогательная задача")
        # В углу лежит -Q. Если Q > 0, искусственные переменные не удалось
        # обнулить, и допустимых точек у задачи нет.
        if tableau.M[-1][-1] != 0:
            return Result("infeasible", [], None)

        # Искусственная переменная может остаться в базисе с нулевым b.
        # Тогда меняем её на любую свободную с ненулевым элементом в этой
        # строке (точка при этом не меняется, т.к. b = 0). Если вся строка
        # нулевая, ограничение лишнее, и строку можно удалить.
        for i in reversed(range(len(tableau.rows))):
            if tableau.rows[i] in artificial:
                j = next(
                    (j for j, value in enumerate(tableau.M[i][:-1]) if value),
                    None,
                )
                if j is None:
                    del tableau.rows[i]
                    del tableau.M[i]
                else:
                    tableau.pivot(i, j)

    # Основная задача: возвращаем настоящую цель. Таблица остаётся прежней,
    # меняется только строка цели (пересчитывается для текущего базиса).
    tableau.set_costs(cost)
    if not tableau.optimize("Основная задача"):
        return Result("unbounded", [], None)

    # Базисные переменные равны своему b, свободные равны нулю.
    values = {
        name: row[-1]
        for name, row in zip(tableau.rows, tableau.M[:-1])
    }
    x = [values.get(name, F(0)) for name in names[:n]]
    # В углу лежит -Q, где Q минимум sign * Z. Отсюда Z = sign * Q.
    return Result("optimal", x, -sign * tableau.M[-1][-1])


if __name__ == "__main__":
    # Вариант 20: Z = x1 + 2x2 + 4x3 + x4 -> min.
    result = solve(
        c=[1, 2, 4, 1],
        A=[[1, 1, 1, 0], [0, 1, 2, 1], [1, 0, 0, 1]],
        rel=["<=", "=", ">="],
        b=[10, 6, 2],
    )
    print(f"\nСтатус: {result.status}")
    if result.status == "optimal":
        print("x* =", tuple(fmt(value) for value in result.x))
        print("Z(x*) =", fmt(result.value))
