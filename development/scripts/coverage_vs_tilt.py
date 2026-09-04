"""График покрытия салона в зависимости от угла наклона камеры.

Статус: расчёт на бумаге, не измерение.
Дата: 2026-09-03
Authority: подчиняется ../../GROUND_TRUTH.md и ../CAMERA_GEOMETRY.md

Зачем
-----
CAMERA_GEOMETRY.md показывает две таблицы по углам наклона: геометрическую
дальность и пиксели на голову. График показывает то же самое как одну
картинку и добавляет то, чего в таблицах нет: чувствительность к высоте
потолка. Высота потолка в репозитории является предположением, поэтому
любой вывод, который держится только на 2.30 м, ненадёжен по определению.

Вся оптика и тригонометрия берутся из camera_geometry.py. Здесь нет своей
модели объектива: дублировать её означало бы завести второй источник истины.
Собственного здесь только композиция, то есть обрезка зоны видимости
границами салона и порогом разрешения.

Что график НЕ показывает
------------------------
Окклюзию. Именно она, а не пиксели, ломает счёт на дальнем конце полного
автобуса (CAMERA_GEOMETRY.md раздел 4). Посчитать её на бумаге нельзя.
Поэтому верхняя часть кривых это ГЕОМЕТРИЧЕСКИЙ потолок, а не ожидаемое
качество.

Запуск:
    python development/scripts/coverage_vs_tilt.py
    python development/scripts/coverage_vs_tilt.py --outdir <путь>
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_geometry as cg  # noqa: E402

# =====================================================================
# ПАРАМЕТРЫ
# Всё, что не пришло из camera_geometry.py, объявлено здесь.
# =====================================================================

# Высоты потолка для проверки чувствительности, м.
# Все три ПРЕДПОЛОЖЕНИЕ. 2.30 это значение из CAMERA_GEOMETRY.md, две
# другие показывают, насколько выводы держатся на нём.
CEILING_HEIGHTS_M = (2.10, 2.30, 2.50)

# Углы, на которые физически рассчитана испытательная оснастка
# development/cad/camera_test_rig.scad
RIG_TILTS_DEG = (0, 30, 45, 60)

TILT_MIN_DEG = 0.0
TILT_MAX_DEG = 80.0
TILT_STEP_DEG = 0.5

FIG_W_IN = 11.0
FIG_H_IN = 8.0
FIG_DPI = 150

OUT_SUBDIR = "camera_geometry"
PNG_NAME = "coverage_vs_tilt.png"
CSV_NAME = "coverage_vs_tilt.csv"

CAPTION = (
    "Расчёт, не измерение. Модель объектива equidistant fisheye из "
    "опубликованных 160 град по диагонали.\n"
    "Размеры салона предположены и не измерены. Окклюзия не моделируется, "
    "поэтому кривые это геометрический потолок, а не ожидаемое качество.\n"
    "Вертикальные красные линии: углы 0, 30, 45 и 60 град, на которые "
    "рассчитана испытательная оснастка development/cad/camera_test_rig.scad."
)

# =====================================================================


def horizontal_limit_by_pixels(lens: cg.Lens, drop_m: float) -> float:
    """Горизонтальная дальность, на которой голова падает до порога.

    Порог и ширина головы берутся из camera_geometry.py. Возвращает 0,
    если камера уже ближе порога по одному только вертикальному сносу.
    """
    d_max = cg.max_useful_distance(lens)
    if d_max <= drop_m:
        return 0.0
    return math.sqrt(d_max**2 - drop_m**2)


def coverage_at_tilt(
    lens: cg.Lens, drop_m: float, tilt_deg: float, cabin_len_m: float
) -> tuple[float, float, float]:
    """Покрытие прохода при заданном наклоне.

    Камера считается установленной у переднего торца салона, отсчёт вдоль
    прохода идёт от точки под камерой. Возвращает
    (покрытая длина, дальняя граница, пикселей на голову на дальней границе).
    """
    near, far = cg.oblique_reach(lens, drop_m, tilt_deg)
    px_limit = horizontal_limit_by_pixels(lens, drop_m)
    start = max(near, 0.0)
    end = min(far, cabin_len_m, px_limit)
    if end <= start:
        return 0.0, start, float("nan")
    slant = cg.slant_distance(drop_m, end)
    return end - start, end, cg.head_px(lens, slant)


def sweep(lens: cg.Lens) -> dict:
    tilts = []
    t = TILT_MIN_DEG
    while t <= TILT_MAX_DEG + 1e-9:
        tilts.append(round(t, 3))
        t += TILT_STEP_DEG

    rows = {}
    for ceiling in CEILING_HEIGHTS_M:
        drop = ceiling - cg.STANDING_HEAD_M
        cov, far, px = [], [], []
        for tilt in tilts:
            c, f, p = coverage_at_tilt(lens, drop, tilt, cg.CABIN_LENGTH_M)
            cov.append(c)
            far.append(f)
            px.append(p)
        rows[ceiling] = {"cov": cov, "far": far, "px": px, "drop": drop}
    return {"tilts": tilts, "by_ceiling": rows}


def write_csv(data: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "tilt_deg",
                "ceiling_m",
                "camera_drop_above_standing_head_m",
                "covered_aisle_len_m",
                "far_edge_m",
                "head_px_at_far_edge",
            ]
        )
        for ceiling, series in data["by_ceiling"].items():
            for i, tilt in enumerate(data["tilts"]):
                w.writerow(
                    [
                        f"{tilt:.1f}",
                        f"{ceiling:.2f}",
                        f"{series['drop']:.2f}",
                        f"{series['cov'][i]:.3f}",
                        f"{series['far'][i]:.3f}",
                        "" if math.isnan(series["px"][i]) else f"{series['px'][i]:.1f}",
                    ]
                )


def plot(data: dict, lens: cg.Lens, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W_IN, FIG_H_IN), sharex=True)
    tilts = data["tilts"]

    for ceiling, series in data["by_ceiling"].items():
        label = f"потолок {ceiling:.2f} м"
        style = "-" if ceiling == 2.30 else "--"
        ax1.plot(tilts, series["cov"], style, label=label)
        ax2.plot(tilts, series["px"], style, label=label)

    ax1.axhline(cg.CABIN_LENGTH_M, color="k", lw=1.0, ls=":")
    ax1.annotate(
        f"длина салона {cg.CABIN_LENGTH_M:.1f} м (предположение)",
        xy=(TILT_MAX_DEG, cg.CABIN_LENGTH_M),
        xytext=(-6, 4),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    ax1.set_ylabel("покрытая длина прохода, м")
    ax1.set_title(
        "Покрытие салона и разрешение на дальнем конце против угла наклона\n"
        "плоскость голов стоящих, IMX219-160, длинная ось сенсора вдоль салона",
        fontsize=12,
    )
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.axhline(cg.MIN_HEAD_PX, color="k", lw=1.0, ls=":")
    ax2.annotate(
        f"порог {cg.MIN_HEAD_PX:.0f} px на голову (правило, не измерение)",
        xy=(TILT_MAX_DEG, cg.MIN_HEAD_PX),
        xytext=(-6, 4),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    ax2.set_ylabel("пикселей на голову на дальней границе")
    ax2.set_xlabel("наклон от надира, град")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="upper right", fontsize=9)

    for ax in (ax1, ax2):
        for tilt in RIG_TILTS_DEG:
            ax.axvline(tilt, color="tab:red", lw=0.8, alpha=0.5)
    fig.text(0.01, 0.005, CAPTION, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def print_rig_table(data: dict) -> None:
    print()
    print("Значения в точках фиксации оснастки")
    print("-" * 35)
    header = f"{'наклон':>8}{'потолок':>10}{'покрытие м':>13}{'дальняя м':>12}{'px/голова':>12}"
    print(header)
    for ceiling, series in data["by_ceiling"].items():
        for tilt in RIG_TILTS_DEG:
            i = data["tilts"].index(float(tilt))
            px = series["px"][i]
            px_s = "-" if math.isnan(px) else f"{px:.0f}"
            print(
                f"{tilt:>6} °{ceiling:>10.2f}{series['cov'][i]:>13.2f}"
                f"{series['far'][i]:>12.2f}{px_s:>12}"
            )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=repo_root / "outputs" / OUT_SUBDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    lens = cg.Lens.from_diagonal(cg.DIAGONAL_FOV_DEG, cg.SENSOR_W_PX, cg.SENSOR_H_PX)
    data = sweep(lens)

    png = args.outdir / PNG_NAME
    csv_path = args.outdir / CSV_NAME
    plot(data, lens, png)
    write_csv(data, csv_path)

    print_rig_table(data)
    print()
    print(f"график: {png}")
    print(f"числа:  {csv_path}")
    print()
    print("Окклюзия не моделируется. Кривые это геометрический потолок.")


if __name__ == "__main__":
    main()
