"""Независимый полный перебор, точные дроби и восстановление всех оптимумов."""

from decimal import Decimal
from itertools import product
import random
import json

from main import main
from allocation.io import load_problem

import pytest

from allocation.method import NoSolution, optimal_plans, solve
from allocation.io import variant20
from allocation.io import render
from allocation.io import parse_problem


def brute(problem):
    plans = [u for u in product(*(t.allowed for t in problem.teams)) if sum(u) == problem.budget]
    if not plans:
        return None, []
    values = [sum(t.rewards[x] for t, x in zip(problem.teams, u)) for u in plans]
    best = max(values)
    return best, [u for u, value in zip(plans, values) if value == best]


def test_variant20():
    result = solve(variant20())
    assert result.value == 36
    assert result.plan == (3, 2, 3)
    assert result.count == 1
    assert brute(result.problem) == (Decimal(36), [(3, 2, 3)])
    assert result.values[1][5] == 23
    assert result.choices[1][5] == (2,)
    assert 'Максимальная результативность: 36' in render(result)


def test_random_tables_against_independent_brute_force():
    rng = random.Random(20)
    for _ in range(150):
        data = {'budget': rng.randrange(10), 'teams': []}
        for k in range(rng.randrange(1, 5)):
            data['teams'].append({'name': str(k), 'rewards': [rng.randrange(-9, 10) for _ in range(5)],
                                  'allowed': rng.sample(range(5), rng.randrange(1, 6))})
        problem = parse_problem(data)
        best, plans = brute(problem)
        if best is None:
            with pytest.raises(NoSolution):
                solve(problem)
        else:
            result = solve(problem)
            assert result.value == best
            assert result.count == len(plans)
            assert optimal_plans(result, 1000) == sorted(plans)


def test_decimal_ties_are_exact():
    p = parse_problem({'budget': 2, 'teams': [
        {'name': 'A', 'rewards': [0, '0,1', '0.3'], 'allowed': [1, 2]},
        {'name': 'B', 'rewards': [0, '0.2'], 'allowed': [0, 1]},
    ]})
    result = solve(p)
    assert result.value == Decimal('0.3')
    assert optimal_plans(result) == [(1, 1), (2, 0)]


def test_zero_budget_and_negative_values():
    p = parse_problem({'budget': 0, 'teams': [{'name': 'A', 'rewards': [-3], 'allowed': [0]}]})
    assert solve(p).value == -3
    assert solve(p).plan == (0,)


def test_limit_does_not_change_count():
    p = parse_problem({'budget': 4, 'teams': [
        {'name': str(i), 'rewards': [0] * 5, 'allowed': list(range(5))} for i in range(4)]})
    result = solve(p)
    assert result.count == 35
    assert len(optimal_plans(result, 2)) == 2
    assert 'Показаны первые 2 планов из 35' in render(result, limit=2)


def test_large_decimal_and_zero_exponent_format():
    p = parse_problem({'budget': 0, 'teams': [{'name': 'A', 'rewards': ['1e100'], 'allowed': [0]}]})
    assert '1' + '0' * 100 in render(solve(p))
    p = parse_problem({'budget': 0, 'teams': [{'name': 'A', 'rewards': ['0e999999'], 'allowed': [0]}]})
    assert solve(p).value == 0


def example():
    return {'budget': 1, 'teams': [{'name': 'A', 'rewards': [0, 1], 'allowed': [0, 1]}]}


@pytest.mark.parametrize('budget', [-1, 1001, True, 1.5, '1', None])
def test_bad_budget(budget):
    data = example()
    data['budget'] = budget
    with pytest.raises(ValueError, match='budget'):
        parse_problem(data)


@pytest.mark.parametrize('change', [
    {'teams': []}, {'teams': {}}, {'extra': 1}, {'title': ''}, {'title': 'a\nb'},
    {'teams': [{'name': 'A', 'rewards': [], 'allowed': [0]}]},
    {'teams': [{'name': 'A', 'rewards': [0], 'allowed': []}]},
    {'teams': [{'name': 'A', 'rewards': [0], 'allowed': [0, 0]}]},
    {'teams': [{'name': 'A', 'rewards': [0], 'allowed': [1]}]},
    {'teams': [{'name': 'A', 'rewards': [0], 'allowed': [True]}]},
    {'teams': [{'name': 'A', 'rewards': [True], 'allowed': [0]}]},
    {'teams': [{'name': 'A', 'rewards': ['NaN'], 'allowed': [0]}]},
    {'teams': [{'name': 'A', 'rewards': ['1e101'], 'allowed': [0]}]},
    {'teams': [{'name': 'A', 'rewards': ['1e-101'], 'allowed': [0]}]},
    {'teams': [{'name': 'A', 'rewards': [''], 'allowed': [0]}]},
])
def test_bad_structure(change):
    with pytest.raises(ValueError):
        parse_problem(example() | change)


def test_duplicate_names_and_missing_fields():
    data = example()
    data['teams'] *= 2
    with pytest.raises(ValueError, match='повторяется'):
        parse_problem(data)
    with pytest.raises(ValueError, match='отсутствуют'):
        parse_problem({})


@pytest.mark.parametrize('content', ['{', '{"budget":1,"budget":2}', '[]', '{"a":NaN}', '\ufeff{'])
def test_bad_json(tmp_path, content):
    path = tmp_path / 'bad.json'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        load_problem(str(path))


def test_json_files(tmp_path):
    path = tmp_path / 'ok.json'
    path.write_text(json.dumps(example()), encoding='utf-8-sig')
    assert load_problem(str(path)).budget == 1
    path.write_bytes(b'\xff\xfe\x00')
    with pytest.raises(ValueError):
        load_problem(str(path))
    with pytest.raises(ValueError):
        load_problem(str(tmp_path))
    with pytest.raises(ValueError):
        load_problem(str(tmp_path / 'absent'))
    path.write_bytes(b' ' * 2_000_001)
    with pytest.raises(ValueError, match='2 МБ'):
        load_problem(str(path))



@pytest.mark.parametrize('argv', [[], ['--brief']])
def test_command_line(argv, capsys):
    assert main(argv) == 0
    assert 'ОТВЕТ' in capsys.readouterr().out


@pytest.mark.parametrize('argv', [['--plans', '0'], ['--plans', '1001'], ['--config', 'absent.json']])
def test_command_line_error(argv, capsys):
    assert main(argv) == 2
    assert 'Ошибка:' in capsys.readouterr().err


def test_command_line_custom_data(tmp_path, capsys):
    file = tmp_path / 'task.json'
    file.write_text(json.dumps(example()), encoding='utf-8')
    assert main(['--config', str(file)]) == 0
    assert 'Максимальная результативность: 1' in capsys.readouterr().out
