"""Запуск: python main.py "x + sin(3.14159*x)" -2 2 0.01 --plot result.png"""

import argparse

import matplotlib

from piyavsky import minimize, parse_number, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Глобальный экстремум липшицевой функции методом ломаных.",
        epilog='Пример: python main.py "x + sin(3.14159*x)" -2 2 0.01. '
               'Если отрицательное выражение принято за опцию, '
               'заключите его в скобки: "(-2*pi)".',
    )
    parser.add_argument("function", help='функция от x, например "x + sin(3.14159*x)"')
    parser.add_argument("a", help="левый конец отрезка")
    parser.add_argument("b", help="правый конец отрезка")
    parser.add_argument("eps", help="точность, например 0,01")
    parser.add_argument("--L", help="константа Липшица (по умолчанию оценивается)")
    parser.add_argument("--max", action="store_true", help="искать максимум")
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--plot", metavar="FILE",
                        help="сохранить график в файл (иначе открыть окно)")
    args = parser.parse_args(argv)

    try:
        a, b, eps = (parse_number(v) for v in (args.a, args.b, args.eps))
        L = None if args.L is None else parse_number(args.L)
        result = minimize(args.function, a, b, eps, L=L,
                          mode="max" if args.max else "min",
                          max_iter=args.max_iter)
    except ValueError as error:
        parser.error(str(error))  # сообщение и код выхода 2

    print(summary(result))
    if args.plot:
        matplotlib.use("Agg")  # без окна
    import matplotlib.pyplot as plt
    from piyavsky.plotting import plot

    plot(result)
    plt.tight_layout()
    if args.plot:
        plt.savefig(args.plot, dpi=150)
        print(f"График: {args.plot}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
