"""Печать условия, дерева решений и сделок на выбранной рыночной траектории."""

from decimal import Decimal, localcontext
from fractions import Fraction

from .method import CRITERIA, MINIMUM_RULES
from .model import market, trade

EVENTS = ("благоприятный", "нейтральный", "негативный")


def number(value, places=2):
    value = Fraction(value)
    with localcontext() as context:
        context.prec = 40
        return f"{Decimal(value.numerator) / Decimal(value.denominator):.{places}f}"


def table(headers, rows):
    rows = [[str(x) for x in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = [" | ".join(x.ljust(w) for x, w in zip(row, widths)) for row in rows]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def branches(result):
    """Прямой ход по всей стратегии: 13 узлов решений и 27 конечных исходов."""
    nodes, leaves = [], []
    p = result.problem

    def walk(stage, state, history, probability):
        action = result.policy[stage, state]
        after = trade(p, state, action)
        nodes.append((history, state, action, after))
        for j in range(3):
            following = market(p, after, stage, j)
            next_history = history + (EVENTS[j],)
            next_probability = probability * p.probabilities[stage][j]
            if stage + 1 == len(p.rates):
                leaves.append((next_history, following, next_probability))
            else:
                walk(stage + 1, following, next_history, next_probability)

    walk(0, p.initial, (), Fraction(1))
    return nodes, leaves


def scenario(result, events):
    """Показывает наблюдаемую последовательность: сделка, затем исход рынка."""
    state = result.problem.initial
    lines = ["\nПРИМЕР ХОДА РЕШЕНИЯ: " + " -> ".join(events)]
    for stage, label in enumerate(events):
        action = result.policy[stage, state]
        after = trade(result.problem, state, action)
        following = market(result.problem, after, stage, EVENTS.index(label))
        rows = [
            ("До сделки", *map(number, state.assets), number(state.cash), number(state.capital)),
            ("После сделки", *map(number, after.assets), number(after.cash), number(after.capital)),
            (f"После рынка {label}", *map(number, following.assets),
             number(following.cash), number(following.capital)),
        ]
        lines.extend([
            f"\nЭтап {stage + 1}. Пакеты: {action}; комиссия: {number(state.capital - after.capital)}.",
            table(["Состояние", "ЦБ1", "ЦБ2", "Депозит", "Свободно", "Капитал"], rows),
        ])
        state = following
    lines.append(f"\nДоход на этой траектории: {number(state.capital - result.problem.initial.capital)} д.е.")
    return "\n".join(lines)


def render(result, brief=False, events=None):
    p = result.problem
    action = result.policy[0, p.initial]
    after = trade(p, p.initial, action)
    lines = [
        "ЛР4. Пошаговое управление инвестиционным портфелем",
        f"Критерий: {CRITERIA[result.criterion]}"
        + (f", вес оптимизма {number(result.alpha)}" if result.criterion == "hurwicz" else ""),
        f"Правило минимумов: {result.minimum_rule} — {MINIMUM_RULES[result.minimum_rule]}.",
    ]
    if result.minimum_rule == "trade":
        lines.append("После рынка стоимость актива может опуститься ниже минимума; "
                     "перед следующим рынком его нужно восстановить сделкой.")
    if not brief:
        rows = [(name, number(x), q, number(100 * f) + "%", number(m))
                for name, x, q, f, m in zip(
                    ("ЦБ1", "ЦБ2", "Депозит"), p.initial.assets, p.packets, p.fees, p.minimums)]
        lines.extend([
            "\nИСХОДНЫЕ ДАННЫЕ",
            table(["Актив", "Стоимость", "Пакет", "Комиссия", "Минимум"], rows),
            f"Свободные средства: {number(p.initial.cash)}. Начальный капитал: {number(p.initial.capital)}.",
            "\nПЕРВОЕ РЕШЕНИЕ",
            table(["Актив", "Пакеты", "Изменение стоимости"], [
                (name, f"{u:+d}", number(q * u))
                for name, q, u in zip(("ЦБ1", "ЦБ2", "Депозит"), p.packets, action)]),
            f"Комиссия: {number(p.initial.capital - after.capital)}. "
            f"Свободные деньги после сделки: {number(after.cash)}.",
        ])
        nodes, _ = branches(result)
        lines.append("\nДАЛЬНЕЙШАЯ СТРАТЕГИЯ (+ покупка, - продажа)")
        lines.append("В истории перечислены исходы рынка по порядку раундов.")
        for stage in range(1, len(p.rates)):
            rows = [(" -> ".join(history), *[f"{u:+d}" for u in action], number(after.cash))
                    for history, _, action, after in nodes if len(history) == stage]
            lines.extend([
                "\nПеред этапом 2: известен исход первого этапа." if stage == 1
                else "\nПеред этапом 3: известны исходы первых двух этапов.",
                table(["История", "ЦБ1, пак.", "ЦБ2, пак.", "Депозит, пак.", "Свободно"], rows),
            ])
    _, leaves = branches(result)
    capitals = [state.capital for _, state, _ in leaves]
    lines.extend([
        "\nОТВЕТ",
        f"Первое управление: {action}.",
        f"Конечный капитал по критерию: {number(result.value, 8)} д.е.",
        f"Чистый доход по критерию: {number(result.value - p.initial.capital, 8)} д.е.",
        f"Капитал в {len(leaves)} исходах: от {number(min(capitals))} до {number(max(capitals))} д.е.",
        "Ожидаемый доход не является гарантией прибыли." if result.criterion == "bayes"
        else "Оценка по выбранному критерию не является математическим ожиданием по заданным вероятностям.",
    ])
    if events:
        lines.append(scenario(result, events))
    return "\n".join(lines)
