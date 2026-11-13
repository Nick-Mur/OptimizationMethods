"""Обратный и прямой ход Беллмана для табличной функции выигрыша."""

from decimal import Decimal, localcontext

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    name: str
    rewards: tuple[Decimal, ...]
    allowed: tuple[int, ...]


@dataclass(frozen=True)
class Problem:
    budget: int
    teams: tuple[Team, ...]
    title: str = "Распределение ресурса"


@dataclass(frozen=True)
class Result:
    problem: Problem
    value: Decimal
    plan: tuple[int, ...]
    count: int
    # None означает недостижимое состояние; choices содержит ВСЕ лучшие управления.
    values: tuple[tuple[Decimal | None, ...], ...]
    choices: tuple[tuple[tuple[int, ...], ...], ...]


class NoSolution(ValueError):
    """Входные данные корректны, но распределить весь ресурс невозможно."""


def solve(problem: Problem) -> Result:
    from .io import parse_problem
    # Проверка данных при программном вызове.
    problem = parse_problem({
        'title': problem.title, 'budget': problem.budget,
        'teams': [{'name': t.name, 'rewards': list(t.rewards),
                   'allowed': list(t.allowed)} for t in problem.teams],
    })
    n, budget = len(problem.teams), problem.budget
    values = [[None] * (budget + 1) for _ in range(n + 1)]
    choices = [[()] * (budget + 1) for _ in range(n)]
    counts = [[0] * (budget + 1) for _ in range(n + 1)]
    values[n][0], counts[n][0] = Decimal(0), 1

    # 100 десятичных знаков + 101 целая цифра + запас на сумму 50 слагаемых.
    with localcontext() as context:
        context.prec = 210
        for k in range(n - 1, -1, -1):
            team = problem.teams[k]
            for s in range(budget + 1):
                for u in team.allowed:
                    if u > s:
                        break
                    tail = values[k + 1][s - u]
                    if tail is None:
                        continue
                    candidate = team.rewards[u] + tail
                    best = values[k][s]
                    if best is None or candidate > best:
                        values[k][s] = candidate
                        choices[k][s] = (u,)
                        counts[k][s] = counts[k + 1][s - u]
                    elif candidate == best:
                        choices[k][s] += (u,)
                        counts[k][s] += counts[k + 1][s - u]

    if values[0][budget] is None:
        raise NoSolution('допустимого плана нет: весь ресурс нельзя распределить с заданными ограничениями')
    plan, s = [], budget
    for k in range(n):
        u = choices[k][s][0]
        plan.append(u)
        s -= u
    return Result(problem, values[0][budget], tuple(plan), counts[0][budget],
                  tuple(map(tuple, values)), tuple(map(tuple, choices)))


def optimal_plans(result: Result, limit: int = 20) -> list[tuple[int, ...]]:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('число выводимых планов должно быть целым от 1 до 1000')
    plans = []

    def walk(k, s, prefix):
        if len(plans) >= limit:
            return
        if k == len(result.problem.teams):
            plans.append(prefix)
            return
        for u in result.choices[k][s]:
            walk(k + 1, s - u, prefix + (u,))
            if len(plans) >= limit:
                break

    walk(0, result.problem.budget, ())
    return plans
