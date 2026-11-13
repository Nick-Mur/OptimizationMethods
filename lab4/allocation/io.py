"""Ввод, проверка данных и вывод решения задачи распределения ресурса."""

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .method import Problem, Team

MAX_BUDGET = 1000
MAX_TEAMS = 50
MAX_WORK = 5_000_000


def integer(value, field: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field}: нужно целое число от {low} до {high}")
    return value


def number(value, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{field}: требуется конечное число")
    text = str(value).strip().replace(",", ".")
    if not text or len(text) > 100:
        raise ValueError(f"{field}: число пустое или длиннее 100 символов")
    try:
        result = Decimal(text)
    except InvalidOperation:
        raise ValueError(f"{field}: не удалось прочитать число {text!r}") from None
    if not result.is_finite() or result.copy_abs() > Decimal('1e100'):
        raise ValueError(f"{field}: нужно конечное число по модулю не больше 1e100")
    if result and result.as_tuple().exponent < -100:
        raise ValueError(f"{field}: допускается не более 100 знаков после запятой")
    return result if result else Decimal(0)


def object_fields(data, allowed: set[str], required: set[str], field: str):
    if not isinstance(data, dict):
        raise ValueError(f"{field}: требуется JSON-объект")
    extra, missing = set(data) - allowed, required - set(data)
    if extra:
        raise ValueError(f"{field}: неизвестные поля {', '.join(map(str, extra))}")
    if missing:
        raise ValueError(f"{field}: отсутствуют поля {', '.join(sorted(missing))}")


def name(value, field: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 100
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"{field}: нужна непустая строка до 100 символов без управляющих знаков")
    return value.strip()


def parse_problem(data) -> Problem:
    object_fields(data, {'title', 'budget', 'teams'}, {'budget', 'teams'}, 'задача')
    budget = integer(data['budget'], 'budget', 0, MAX_BUDGET)
    title = name(data.get('title', 'Распределение ресурса'), 'title')
    raw = data['teams']
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_TEAMS:
        raise ValueError(f"teams: нужен список от 1 до {MAX_TEAMS} участников")
    teams, seen = [], set()
    for i, item in enumerate(raw, 1):
        field = f'teams[{i}]'
        object_fields(item, {'name', 'rewards', 'allowed'},
                      {'name', 'rewards', 'allowed'}, field)
        label = name(item['name'], field + '.name')
        if label in seen:
            raise ValueError(f"{field}: название {label!r} повторяется")
        seen.add(label)
        rewards = item['rewards']
        if not isinstance(rewards, list) or not 1 <= len(rewards) <= MAX_BUDGET + 1:
            raise ValueError(f"{field}.rewards: нужен непустой список до {MAX_BUDGET + 1} чисел")
        rewards = tuple(number(x, f'{field}.rewards[{j}]') for j, x in enumerate(rewards))
        allowed = item['allowed']
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(f"{field}.allowed: нужен непустой список разрешённых объёмов")
        allowed = tuple(integer(x, field + '.allowed', 0, len(rewards) - 1) for x in allowed)
        if len(set(allowed)) != len(allowed):
            raise ValueError(f"{field}.allowed: объёмы не должны повторяться")
        teams.append(Team(label, rewards, tuple(sorted(allowed))))
    if (budget + 1) * sum(len(t.allowed) for t in teams) > MAX_WORK:
        raise ValueError('слишком большая таблица: уменьшите бюджет или число управлений')
    return Problem(budget, tuple(teams), title)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'JSON: поле {key!r} задано повторно')
        result[key] = value
    return result


def read_json(path: str):
    try:
        file = Path(path)
        if file.stat().st_size > 2_000_000:
            raise ValueError('JSON-файл должен быть не больше 2 МБ')
        return json.loads(file.read_text(encoding='utf-8-sig'),
                          parse_float=Decimal, object_pairs_hook=unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f'не удалось прочитать JSON {path}: {exc}') from None


def load_problem(path: str) -> Problem:
    return parse_problem(read_json(path))


def variant20() -> Problem:
    return parse_problem({
        'title': 'Вариант 20. Грантовые пакеты', 'budget': 8,
        'teams': [
            {'name': 'Команда А', 'allowed': [1, 2, 3],
             'rewards': [0, 4, 9, 13, 16, 18, 19, 20, 21]},
            {'name': 'Команда Б', 'allowed': [1, 2, 3],
             'rewards': [0, 5, 9, 12, 14, 15, 16, 17, 18]},
            {'name': 'Команда В', 'allowed': [1, 2, 3],
             'rewards': [0, 3, 8, 14, 19, 22, 24, 25, 26]},
        ],
    })


def table(headers, rows) -> str:
    rows = [[str(x) for x in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = [' | '.join(x.ljust(w) for x, w in zip(row, widths)) for row in rows]
    return '\n'.join([lines[0], '-+-'.join('-' * w for w in widths), *lines[1:]])


def fmt(value) -> str:
    if value is None:
        return '—'
    text = format(value, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def render(result, show_tables=True, limit=20) -> str:
    p = result.problem
    sections = [p.title, f'Ресурс: {p.budget} пакетов. Участников: {len(p.teams)}.',
                '1. ДОПУСТИМЫЕ УПРАВЛЕНИЯ\n' + table(
                    ['Участник', 'Пакеты'],
                    [(t.name, ', '.join(map(str, t.allowed))) for t in p.teams])]
    if show_tables:
        if p.budget <= 30 and len(p.teams) <= 10:
            sections.append('2. ОБРАТНЫЙ ХОД\n'
                            'F_k(s): лучший результат команд k..n при остатке s.\n'
                            '—: состояние недостижимо. u*: все лучшие решения.\n\n' + table(
                ['s', *[f'F{k + 1}; u*' for k in range(len(p.teams))]],
                [[s, *[fmt(result.values[k][s]) + '; ' +
                       (','.join(map(str, result.choices[k][s])) or '—')
                       for k in range(len(p.teams))]] for s in range(p.budget + 1)]))
        else:
            sections.append('2. ОБРАТНЫЙ ХОД\nТаблица большая; ниже показан оптимальный путь.')
    s, rows = p.budget, []
    for k, (t, u) in enumerate(zip(p.teams, result.plan)):
        rows.append([k + 1, t.name, s, u, s - u, fmt(t.rewards[u])])
        s -= u
    sections.append('3. ПРЯМОЙ ХОД\n' + table(
        ['Шаг', 'Участник', 'Было', 'Выделено', 'Осталось', 'Выигрыш'], rows))
    from .method import optimal_plans
    plans = optimal_plans(result, limit)
    sections.append('4. ОТВЕТ\n' +
                    f'Максимальная результативность: {fmt(result.value)}.\n' +
                    f'Оптимальных планов: {result.count}.\n' +
                    '\n'.join('  (' + ', '.join(map(str, plan)) + ')' for plan in plans))
    if result.count > len(plans):
        sections.append(f'Показаны первые {len(plans)} планов из {result.count}.')
    sections.append(f'Проверка: распределено {sum(result.plan)} из {p.budget}; '
                    'все управления разрешены; остаток 0.')
    return '\n\n'.join(sections)
