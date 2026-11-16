"""Данные задания и два действия с портфелем: сделка и изменение рынка."""

from dataclasses import dataclass
from fractions import Fraction


def numbers(text):
    """Точные десятичные числа без промежуточного преобразования во float."""
    return tuple(Fraction(x) for x in text.split())


@dataclass(frozen=True)
class State:
    assets: tuple[Fraction, Fraction, Fraction]
    cash: Fraction

    @property
    def capital(self):
        return sum(self.assets) + self.cash


@dataclass(frozen=True)
class Problem:
    # Денежные величины — непосредственно в д.е., без масштабирования.
    initial: State = State(numbers("100 800 400"), Fraction(600))
    packets: tuple[int, int, int] = (25, 200, 100)
    fees: tuple[Fraction, ...] = numbers("0.04 0.07 0.05")
    minimums: tuple[Fraction, ...] = numbers("30 150 100")
    probabilities: tuple = (
        numbers("0.60 0.30 0.10"),
        numbers("0.30 0.20 0.50"),
        numbers("0.40 0.40 0.20"),
    )
    # Этап -> ситуация (благоприятная, нейтральная, негативная) -> три актива.
    rates: tuple = (
        (numbers("1.20 1.10 1.07"), numbers("1.05 1.02 1.03"), numbers("0.80 0.95 1.00")),
        (numbers("1.40 1.15 1.01"), numbers("1.05 1.00 1.00"), numbers("0.60 0.90 1.00")),
        (numbers("1.15 1.12 1.05"), numbers("1.05 1.01 1.01"), numbers("0.70 0.94 1.00")),
    )

    def __post_init__(self):
        if not 1 <= len(self.rates) <= 3 or len(self.rates) != len(self.probabilities):
            raise ValueError("Число этапов должно быть от 1 до 3")
        if any(len(x) != 3 for x in (self.initial.assets, self.packets, self.fees, self.minimums)):
            raise ValueError("В портфеле должно быть три актива")
        if self.initial.cash < 0 or any(x < 0 for x in self.initial.assets):
            raise ValueError("Начальные остатки не могут быть отрицательными")
        if any(type(q) is not int or q <= 0 for q in self.packets):
            raise ValueError("Пакеты должны быть положительными целыми числами")
        if any(not 0 <= f < 1 for f in self.fees) or any(m < 0 for m in self.minimums):
            raise ValueError("Комиссии должны лежать в [0,1), минимумы — быть неотрицательными")
        for probabilities, rates in zip(self.probabilities, self.rates):
            if len(probabilities) != 3 or min(probabilities) <= 0 or sum(probabilities) != 1:
                raise ValueError("Три положительные вероятности должны давать сумму 1")
            if len(rates) != 3 or any(len(row) != 3 for row in rates):
                raise ValueError("На этапе нужны три ситуации для трёх активов")
            if any(not rates[0][i] >= rates[1][i] >= rates[2][i] > 0 for i in range(3)):
                raise ValueError("Множители должны убывать от благоприятной ситуации к негативной")


def trade(problem, state, action):
    """u > 0 — покупка, u < 0 — продажа; комиссия удерживается с обеих сторон."""
    if len(action) != 3 or any(type(u) is not int for u in action):
        raise ValueError("Управление — три целых числа пакетов")
    changes = tuple(q * u for q, u in zip(problem.packets, action))
    commission = sum(f * abs(d) for f, d in zip(problem.fees, changes))
    assets = tuple(x + d for x, d in zip(state.assets, changes))
    cash = state.cash - sum(changes) - commission
    if cash < 0 or any(x < m for x, m in zip(assets, problem.minimums)):
        raise ValueError("Сделка нарушает минимальные остатки или требует кредита")
    return State(assets, cash)


def market(problem, state, stage, event):
    """После сделки рынок изменяет активы; свободные деньги не изменяются."""
    rates = problem.rates[stage][event]
    return State(tuple(x * r for x, r in zip(state.assets, rates)), state.cash)
