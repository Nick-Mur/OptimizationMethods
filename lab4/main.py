"""Командная строка лабораторной работы №4."""

import argparse
import sys

from allocation import io, method


def main(argv=None):
    parser = argparse.ArgumentParser(description='ЛР4. Динамическое программирование, вариант 20.')
    parser.add_argument('--config', metavar='JSON', help='файл исходных данных')
    parser.add_argument('--brief', action='store_true', help='скрыть таблицу Беллмана')
    parser.add_argument('--plans', type=int, default=20, help='число выводимых оптимальных планов (1..1000)')
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.plans <= 1000:
            raise ValueError('--plans: требуется целое число от 1 до 1000')
        problem = io.load_problem(args.config) if args.config else io.variant20()
        print(io.render(method.solve(problem), not args.brief, args.plans))
    except (ValueError, OverflowError, MemoryError) as exc:
        print(f'Ошибка: {exc or "недостаточно памяти"}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Расчёт прерван пользователем.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
