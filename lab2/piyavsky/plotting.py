"""График: функция, вспомогательные функции g_i, ломаная."""

import matplotlib.pyplot as plt
import numpy as np

from .method import Result
from .validation import parse_function


def plot(result: Result, n: int | None = None, ax=None):
    """Рисует W, g_i, ломаную и x* после итерации n (по умолчанию последней)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    sign = 1 if result.mode == "min" else -1
    n = result.iterations if n is None else n
    if not 1 <= n <= result.iterations:
        raise ValueError(f"n должно быть от 1 до {result.iterations}")
    L = result.L
    (a, _), (b, _) = result.points[:2]
    us, zs = result.points[: n + 1, 0], sign * result.points[: n + 1, 1]

    # Вершины ломаной: верхние — вычисленные точки, нижние — между ними.
    k = np.argsort(us)
    xs, ys = us[k], zs[k]
    low_x = (xs[:-1] + xs[1:]) / 2 + (ys[:-1] - ys[1:]) / (2 * L)
    low_y = (ys[:-1] + ys[1:]) / 2 - L * np.diff(xs) / 2
    line_x = np.insert(xs, range(1, len(xs)), low_x)
    line_y = np.insert(ys, range(1, len(ys)), low_y)

    grid = np.linspace(a, b, 2000)  # только для рисования
    curve = parse_function(result.function)(grid)
    for u, z in zip(us, zs):
        g = [z - L * (u - a), z, z - L * (b - u)]
        ax.plot([a, u, b], sign * np.array(g), color="gray", lw=0.6, alpha=0.5)
    ax.plot(grid, curve, color="tab:blue", lw=2, label="W(x)")
    name = "миноранта" if result.mode == "min" else "мажоранта"
    ax.plot(line_x, sign * line_y, color="tab:green", lw=1.5,
            label=f"ломаная p_{n} ({name}), серые — g_i(x)")
    ax.scatter(us, sign * zs, color="black", s=12, zorder=3, label="u_k")

    best = int(np.argmin(zs))
    ax.scatter([us[best]], [sign * zs[best]], color="red", s=60, zorder=4,
               label=f"x* = {us[best]:.4f}, W = {sign * zs[best]:.4f}")
    ax.axvline(us[best], color="red", ls="--", lw=0.8)

    # Ось Y — по функции и ломаной, иначе g_i растягивают график.
    shown = np.concatenate([curve, sign * line_y])
    pad = 0.1 * (np.ptp(shown) or 1)
    ax.set_ylim(shown.min() - pad, shown.max() + pad)
    ax.set_xlim(a, b)
    ax.set_xlabel("x")
    ax.set_title(f"{result.function}, итерация n = {n}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return ax
