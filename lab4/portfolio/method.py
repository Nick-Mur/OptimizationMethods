"""Обратный ход Беллмана и восстановление стратегии для каждого наблюдаемого состояния."""

from dataclasses import dataclass
from fractions import Fraction

from .model import Problem, State, market, trade

CRITERIA = {"bayes": "Байеса", "wald": "Вальда", "laplace": "Лапласа", "hurwicz": "Гурвица"}
MINIMUM_RULES = {
    "trade": "только после сделки",
    "round": "после сделки и после каждого возможного исхода рынка",
}


@dataclass(frozen=True)
class Result:
    problem: Problem
    criterion: str
    alpha: Fraction
    minimum_rule: str
    value: Fraction
    policy: dict[tuple[int, State], tuple[int, int, int]]


def evaluate(values, probabilities, criterion, alpha):
    if criterion == "bayes":
        return sum(p * v for p, v in zip(probabilities, values))
    if criterion == "wald":
        return min(values)
    if criterion == "laplace":
        return sum(values, Fraction(0)) / 3
    return alpha * max(values) + (1 - alpha) * min(values)


def required_holdings(problem, stage, minimum_rule="round"):
    """Порог после сделки по выбранному правилу проверки минимумов."""
    if minimum_rule == "trade":
        return problem.minimums
    if minimum_rule != "round":
        raise ValueError("Неизвестное правило проверки минимумов")
    return tuple(max(m, m / r) for m, r in zip(problem.minimums, problem.rates[stage][2]))


def lower_actions(problem, state, required):
    return tuple(-((x - m) // q) for x, m, q in zip(state.assets, required, problem.packets))


def actions(problem, state, stage, minimum_rule="round"):
    """Все допустимые покупки/продажи; продажи можно выполнить перед покупками."""
    required = required_holdings(problem, stage, minimum_rule)
    lower = lower_actions(problem, state, required)
    upper = tuple((state.capital - sum(required) + required[i] - state.assets[i]) // q
                  for i, q in enumerate(problem.packets))
    q1, q2, q3 = problem.packets
    fee1, fee2, fee3 = (q * f for q, f in zip(problem.packets, problem.fees))
    for u2 in range(lower[1], upper[1] + 1):
        cash2 = state.cash - q2 * u2 - fee2 * abs(u2)
        for u3 in range(lower[2], upper[2] + 1):
            cash3 = cash2 - q3 * u3 - fee3 * abs(u3)
            # Стоимость сделки с ЦБ1 монотонна по u1, поэтому получаем верхний
            # предел из запрета кредита, а не задаём искусственный лимит пакетов.
            limit = cash3 // (q1 + fee1 if cash3 >= 0 else q1 - fee1)
            for u1 in range(lower[0], min(upper[0], limit) + 1):
                yield (u1, u2, u3), State(
                    (state.assets[0] + q1 * u1,
                     state.assets[1] + q2 * u2,
                     state.assets[2] + q3 * u3),
                    cash3 - q1 * u1 - fee1 * abs(u1),
                )


def upper_bounds(problem, criterion, alpha):
    """Линейные оценки сверху: снимаем минимумы и целочисленность сделок.

    Если продолжение <= a*c + sum(b_i*x_i), оценка после текущей сделки —
    a*cash + sum(A(r_i)*b_i*y_i). Ситуации упорядочены по всем активам,
    поэтому каждый из четырёх критериев для этой оценки является линейным.
    """
    continuation = (Fraction(1),) * 4  # терминальный капитал
    bounds = [None] * len(problem.rates)
    for stage in range(len(problem.rates) - 1, -1, -1):
        cash_weight = continuation[0]
        asset_weights = tuple(evaluate(
            [continuation[i + 1] * row[i] for row in problem.rates[stage]],
            problem.probabilities[stage], criterion, alpha) for i in range(3))
        bounds[stage] = (cash_weight, *asset_weights)
        # Цена покупки единицы — 1+f, выручка от продажи — 1-f.
        liquidity = max(cash_weight, *(a / (1 + f)
                                       for a, f in zip(asset_weights, problem.fees)))
        continuation = (liquidity, *(max(a, liquidity * (1 - f))
                                    for a, f in zip(asset_weights, problem.fees)))
    return bounds


def score(state, weights):
    return weights[0] * state.cash + sum(w * x for w, x in zip(weights[1:], state.assets))


def solve(problem=Problem(), criterion="bayes", alpha=Fraction(1, 2), minimum_rule="round"):
    if criterion not in CRITERIA:
        raise ValueError("Неизвестный критерий")
    if minimum_rule not in MINIMUM_RULES:
        raise ValueError("Неизвестное правило проверки минимумов")
    alpha = Fraction(alpha)
    if not 0 <= alpha <= 1:
        raise ValueError("Вес оптимизма должен лежать от 0 до 1")
    bounds = upper_bounds(problem, criterion, alpha)
    memo, decisions = {}, {}

    def bellman(stage, state):
        if stage == len(problem.rates):
            return state.capital
        key = (stage, state)
        if key in memo:
            return memo[key]
        weights = bounds[stage]
        last = stage == len(problem.rates) - 1
        # На последнем этапе оценка точна. Если покупка невыгодна, выбираем
        # минимум обязательных покупок и выгодные продажи независимо по активам.
        if last and all(a <= 1 + f for a, f in zip(weights[1:], problem.fees)):
            lower = lower_actions(problem, state, required_holdings(problem, stage, minimum_rule))
            action = tuple(lower[i] if a < 1 - problem.fees[i] else max(0, lower[i])
                           for i, a in enumerate(weights[1:]))
            try:
                after = trade(problem, state, action)
            except ValueError:
                pass  # обязательные покупки нужно финансировать другими продажами
            else:
                memo[key], decisions[key] = score(after, weights), action
                return memo[key]
        candidates = [(score(after, weights), action, after)
                      for action, after in actions(problem, state, stage, minimum_rule)]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        best, best_action = None, None
        for bound, action, after in candidates:
            if best is not None and bound <= best:
                break  # остальные границы не больше: они не могут улучшить ответ
            if last:
                value = bound
            else:
                children = []
                order = (2, 1, 0) if criterion == "wald" else (0, 1, 2)
                for j in order:
                    child = bellman(stage + 1, market(problem, after, stage, j))
                    if child is None:
                        break
                    children.append(child)
                    # Для минимума достаточно одного плохого исхода, чтобы
                    # отвергнуть сделку; остальные ветви уже не улучшат оценку.
                    if criterion == "wald" and best is not None and child <= best:
                        break
                if len(children) != 3:
                    continue
                value = evaluate(children, problem.probabilities[stage], criterion, alpha)
            if best is None or value > best:
                best, best_action = value, action
        memo[key], decisions[key] = best, best_action
        return best

    value = bellman(0, problem.initial)
    if value is None:
        raise ValueError("Нет стратегии, допустимой при всех рыночных ситуациях")
    policy = {}

    def restore(stage, state):
        action = decisions[stage, state]
        policy[stage, state] = action
        if stage + 1 < len(problem.rates):
            after = trade(problem, state, action)
            for j in range(3):
                restore(stage + 1, market(problem, after, stage, j))

    restore(0, problem.initial)
    return Result(problem, criterion, alpha, minimum_rule, value, policy)
