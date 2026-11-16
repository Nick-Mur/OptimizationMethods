"""ЛР4. Пошаговое управление инвестиционным портфелем."""

import argparse
from fractions import Fraction
import sys

from portfolio.method import CRITERIA, MINIMUM_RULES, solve
from portfolio.model import Problem
from portfolio.output import EVENTS, render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterion", choices=CRITERIA,
                        default="bayes", help="критерий (по умолчанию — Байеса)")
    parser.add_argument("--alpha", default="0.5", help="вес оптимизма Гурвица, от 0 до 1")
    parser.add_argument("--minimum-rule", choices=MINIMUM_RULES, default="round",
                        help="trade — минимумы только после сделки; round — также после рынка (по умолчанию)")
    parser.add_argument("--brief", action="store_true", help="не печатать дерево стратегии")
    parser.add_argument("--scenario", nargs=3, choices=EVENTS,
                        help="показать сделки и состояния для трёх заданных исходов рынка")
    args = parser.parse_args(argv)
    try:
        result = solve(Problem(), args.criterion, Fraction(args.alpha), args.minimum_rule)
        print(render(result, args.brief, args.scenario))
    except (ValueError, ZeroDivisionError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Расчёт прерван пользователем.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
