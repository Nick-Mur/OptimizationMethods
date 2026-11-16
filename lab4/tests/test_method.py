"""Проверка Беллмана полным перебором и контроль финансовых ограничений."""

from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import random

import pytest

from portfolio.method import actions, evaluate, required_holdings, score, solve, upper_bounds
from portfolio.model import Problem, State, market, numbers, trade
from portfolio.output import EVENTS, branches, number, render, scenario


def enumerate_all(problem, criterion, alpha, minimum_rule="round"):
    """Независимый перебор: без генератора actions, оценок сверху и ускорений."""
    @lru_cache(None)
    def visit(stage, state):
        if stage == len(problem.rates):
            return state.capital
        ranges = [range(-x // q, state.capital // q + 1)
                  for x, q in zip(state.assets, problem.packets)]
        best = None
        for action in product(*ranges):
            changes = [q * u for q, u in zip(problem.packets, action)]
            assets = tuple(x + d for x, d in zip(state.assets, changes))
            cash = state.cash - sum(changes) - sum(
                abs(d) * f for d, f in zip(changes, problem.fees))
            if cash < 0 or any(x < m for x, m in zip(assets, problem.minimums)):
                continue
            values = []
            for row in problem.rates[stage]:
                following = tuple(x * r for x, r in zip(assets, row))
                if minimum_rule == "round" and any(x < m for x, m in zip(following, problem.minimums)):
                    break
                value = visit(stage + 1, State(following, cash))
                if value is None:
                    break
                values.append(value)
            if len(values) != 3:
                continue
            if criterion == "bayes":
                value = sum(p * v for p, v in zip(problem.probabilities[stage], values))
            elif criterion == "wald":
                value = min(values)
            elif criterion == "laplace":
                value = sum(values) / 3
            else:
                value = alpha * max(values) + (1 - alpha) * min(values)
            if best is None or value > best:
                best = value
        return best

    return visit(0, problem.initial)


def small_problem(seed, horizon):
    rng = random.Random(seed)
    rates = []
    for _ in range(horizon):
        columns = [sorted(rng.sample(numbers("0.6 0.8 1 1.1 1.3 1.5"), 3), reverse=True)
                   for _ in range(3)]
        rates.append(tuple(zip(*columns)))
    return Problem(
        initial=State(numbers("2 2 2"), F(1)), packets=(2, 2, 2),
        fees=tuple(rng.choice(numbers("0 0.04 0.07")) for _ in range(3)),
        minimums=numbers("1 1 1"),
        probabilities=(numbers("0.3 0.4 0.3"),) * horizon, rates=tuple(rates),
    )


@pytest.mark.parametrize("criterion,alpha", [
    ("bayes", F(1, 2)), ("wald", F(1, 2)), ("laplace", F(1, 2)),
    ("hurwicz", F(0)), ("hurwicz", F(1, 2)), ("hurwicz", F(1)),
])
@pytest.mark.parametrize("minimum_rule", ["trade", "round"])
def test_matches_exhaustive_search(criterion, alpha, minimum_rule):
    for horizon in (1, 2, 3):
        for seed in range(3):
            problem = small_problem(seed, horizon)
            expected = enumerate_all(problem, criterion, alpha, minimum_rule)
            if expected is None:
                with pytest.raises(ValueError, match="Нет стратегии"):
                    solve(problem, criterion, alpha, minimum_rule)
            else:
                assert solve(problem, criterion, alpha, minimum_rule).value == expected


@pytest.mark.parametrize("minimum_rule,criterion,action,value", [
    ("round", "bayes", (23, 0, 0), "2033.7038838"),
    ("round", "wald", (-2, -3, 0), "1759.54"),
    ("round", "laplace", (0, 0, 5), F(7122274, 3600)),
    ("round", "hurwicz", (0, 0, 5), "1994.5264375"),
    ("trade", "bayes", (23, 0, 0), "2033.9888838"),
    ("trade", "wald", (-2, -3, 0), "1775.04"),
    ("trade", "laplace", (0, 0, 5), F(7122274, 3600)),
    ("trade", "hurwicz", (-1, 0, 5), "1994.9639375"),
])
def test_coursework_and_every_branch(minimum_rule, criterion, action, value):
    result = solve(criterion=criterion, minimum_rule=minimum_rule)
    p = result.problem
    assert result.minimum_rule == minimum_rule
    assert result.value == F(value)
    assert result.policy[0, p.initial] == action
    nodes, leaves = branches(result)
    assert len(nodes) == 13 and len(leaves) == 27
    assert sum(probability for _, _, probability in leaves) == 1
    for _, before, u, after in nodes:
        fee = sum(abs(q * n) * f for q, n, f in zip(p.packets, u, p.fees))
        assert after.capital == before.capital - fee
        assert after.cash >= 0
        assert all(x >= m for x, m in zip(after.assets, p.minimums))
        if minimum_rule == "round":
            assert all(x >= m for x, m in zip(before.assets, p.minimums))
    if minimum_rule == "round":
        for _, state, _ in leaves:
            assert all(x >= m for x, m in zip(state.assets, p.minimums))
    if criterion == "bayes":
        assert sum(s.capital * probability for _, s, probability in leaves) == result.value
        limits = ("1519.76", "2405.934") if minimum_rule == "round" else ("1533.26", "2389.684")
        assert min(s.capital for _, s, _ in leaves) == F(limits[0])
        assert max(s.capital for _, s, _ in leaves) == F(limits[1])
    if criterion == "wald":
        assert min(s.capital for _, s, _ in leaves) == result.value


def test_source_data_and_first_trade():
    p = Problem()
    assert p.initial.capital == 1900
    assert p.packets == (25, 200, 100)
    assert p.fees == numbers("0.04 0.07 0.05")
    assert p.minimums == numbers("30 150 100")
    assert p.probabilities[1] == numbers("0.3 0.2 0.5")
    assert p.rates[1][2] == numbers("0.6 0.9 1")
    after = trade(p, p.initial, (23, 0, 0))
    assert after == State(numbers("675 800 400"), F(2))
    assert market(p, after, 0, 0) == State(numbers("810 880 428"), F(2))


def test_commissions_and_sales_finance_purchases():
    p = Problem()
    assert trade(p, p.initial, (-2, 0, 1)) == State(numbers("50 800 500"), F(543))
    assert trade(p, State(p.initial.assets, F(0)), (1, -1, 0)).cash == 160


@pytest.mark.parametrize("action", [(24, 0, 0), (-3, 0, 0), (0, -4, 0), (0, 0, -4), (0.5, 0, 0)])
def test_rejects_forbidden_trades(action):
    with pytest.raises(ValueError):
        trade(Problem(), Problem().initial, action)


def test_generator_and_first_stage_certificate():
    p = Problem()
    candidates = list(actions(p, p.initial, 0))
    assert len(candidates) == 1502
    weights = upper_bounds(p, "bayes", F(1, 2))[0]
    ordered = sorted((score(after, weights), action) for action, after in candidates)[::-1]
    assert ordered[0][1] == (23, 0, 0)
    assert ordered[5][0] == F("2033.6898432") < solve(p).value
    for action, after in candidates:
        assert trade(p, p.initial, action) == after
        for j in range(3):
            assert all(x >= m for x, m in zip(market(p, after, 0, j).assets, p.minimums))


def test_mandatory_purchase_needs_funding():
    p = Problem(
        initial=State(numbers("1 5 2"), F(0)), packets=(1, 1, 1), minimums=numbers("2 1 1"),
        probabilities=(numbers("0.3 0.4 0.3"),),
        rates=((numbers("1 1 1"),) * 3,),
    )
    result = solve(p)
    assert result.value == enumerate_all(p, "bayes", F(1, 2)) == F("7.84")
    assert result.policy[0, p.initial] == (1, -1, -1)


def test_minimum_rule_changes_admissible_sale():
    p = Problem()
    state = State(numbers("810 880 428"), F(2))
    assert required_holdings(p, 1, "trade")[0] == 30
    assert required_holdings(p, 1, "round")[0] == 50
    trade_actions = {action for action, _ in actions(p, state, 1, "trade")}
    round_actions = {action for action, _ in actions(p, state, 1, "round")}
    assert (-31, 0, 0) in trade_actions and (-31, 0, 0) not in round_actions
    assert (-30, 0, 0) in trade_actions & round_actions
    after = trade(p, state, (-31, 0, 0))
    assert after.assets[0] == 35
    assert market(p, after, 1, 2).assets[0] == 21


def test_trade_rule_allows_terminal_drop_but_requires_next_trade_to_be_feasible():
    p = Problem(
        initial=State(numbers("40 150 100"), F(0)),
        probabilities=(numbers("0.3 0.2 0.5"),),
        rates=((numbers("1.4 1 1"), numbers("1.05 1 1"), numbers("0.6 1 1")),),
    )
    result = solve(p, minimum_rule="trade")
    assert result.value == F("287.2")
    assert min(state.assets[0] for _, state, _ in branches(result)[1]) == 24
    with pytest.raises(ValueError, match="Нет стратегии"):
        solve(p, minimum_rule="round")
    # Если есть ещё один этап, 24 нужно пополнить до 30. Денег на это нет,
    # другие активы продать нельзя, поэтому стратегия недопустима даже в trade.
    next_stage = (numbers("1 1 1"),) * 3
    longer = replace(p, probabilities=p.probabilities * 2, rates=(*p.rates, next_stage))
    with pytest.raises(ValueError, match="Нет стратегии"):
        solve(longer, minimum_rule="trade")


def test_invalid_problem_and_no_strategy():
    with pytest.raises(ValueError, match="Нет стратегии"):
        solve(replace(Problem(), minimums=numbers("1000 1000 1000")))
    with pytest.raises(ValueError):
        replace(Problem(), fees=numbers("1 0 0"))
    with pytest.raises(ValueError):
        replace(Problem(), probabilities=(numbers("0.5 0.5 0.5"),) * 3)
    with pytest.raises(ValueError):
        solve(criterion="unknown")
    with pytest.raises(ValueError):
        solve(alpha=F(2))
    with pytest.raises(ValueError, match="правило"):
        solve(minimum_rule="unknown")


def test_exact_evaluation_and_rounding_only_in_output():
    assert evaluate([F(1), F(2), F(2)], numbers("0.6 0.3 0.1"), "laplace", F(1, 2)) == F(5, 3)
    assert number(F(1, 3), 4) == "0.3333"


def test_cli_and_demonstration(capsys):
    from main import main
    assert main(["--brief", "--scenario", "благоприятный", "негативный", "нейтральный"]) == 0
    text = capsys.readouterr().out
    assert "133.70388380" in text
    assert "Правило минимумов: round" in text
    assert "После сделки" in text and "После рынка негативный" in text
    assert "Доход на этой траектории: 92.25" in text
    assert main(["--alpha", "nan"]) == 2
    assert main(["--alpha", "2"]) == 2


def test_cli_trade_rule(capsys):
    from main import main
    assert main(["--brief", "--minimum-rule", "trade"]) == 0
    text = capsys.readouterr().out
    assert "Правило минимумов: trade" in text
    assert "133.98888380" in text
    assert "может опуститься ниже минимума" in text
    with pytest.raises(SystemExit) as error:
        main(["--minimum-rule", "unknown"])
    assert error.value.code == 2


@pytest.fixture(scope="module")
def bayes_result():
    return solve()


def test_named_histories_and_report_trajectory(bayes_result):
    assert EVENTS == ("благоприятный", "нейтральный", "негативный")
    nodes, leaves = branches(bayes_result)
    assert nodes[0][0] == ()
    assert {history for history, _, _ in leaves} == set(product(EVENTS, repeat=3))
    history = ("благоприятный", "негативный", "нейтральный")
    _, final, probability = next(leaf for leaf in leaves if leaf[0] == history)
    assert final == State(numbers("64.05 799.92 432.28"), F(696))
    assert probability == F("0.12")
    text = scenario(bayes_result, history)
    assert "благоприятный -> негативный -> нейтральный" in text
    assert text.count("До сделки") == text.count("После сделки") == 3
    assert "Доход на этой траектории: 92.25" in text


def test_full_and_brief_output(bayes_result):
    full = render(bayes_result)
    brief = render(bayes_result, brief=True)
    for heading in ("ИСХОДНЫЕ ДАННЫЕ", "ПЕРВОЕ РЕШЕНИЕ", "ДАЛЬНЕЙШАЯ СТРАТЕГИЯ"):
        assert heading in full and heading not in brief
    assert "благоприятный -> негативный" in full
    for text in (full, brief):
        assert "Первое управление: (23, 0, 0)." in text
        assert "133.70388380" in text
        assert "ПРИМЕР ХОДА РЕШЕНИЯ" not in text
    assert "ПРИМЕР ХОДА РЕШЕНИЯ" in render(bayes_result, True, EVENTS)


@pytest.mark.parametrize("args", [
    ["--scenario", "Б", "П", "Н"],
    ["--scenario", "благоприятный", "нейтральный"],
    ["--scenario", "благоприятный", "нейтральный", "неизвестный"],
    ["--scenario", *EVENTS, "негативный"],
    ["--criterion", "unknown"],
])
def test_cli_rejects_invalid_arguments(args):
    from main import main
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_cli_help_lists_current_arguments(capsys):
    from main import main
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    text = capsys.readouterr().out
    for argument in ("--criterion", "--alpha", "--minimum-rule", "--brief", "--scenario"):
        assert argument in text
    assert all(event in text for event in EVENTS)
    assert "ОТВЕТ" not in text
