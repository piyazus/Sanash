"""Демо-анимация Sanas для встречи с Innoforce/Avtobys. Локальный CPU.

Что делает:
  1. Берёт 4 окна записи (эпизодные чанки из data/apc/apc_manifest.json),
     в каждом есть посадка и/или высадка, которые ловит 8x8-вариант (mz8).
  2. Прогоняет НАСТОЯЩИЙ конвейер из development/src/sanas/door_apc.py (то же 60-град.
     кадрирование, тот же p20-пулинг в 8x8, тот же трекер) — ничего не
     переписано и не заскриптовано руками; времена событий сверяются с
     research/paper/results/apc_validation.json и при расхождении скрипт падает.
  3. Рендерит MP4 (ffmpeg) или GIF (fallback, pillow) + статичные кадры
     для слайдов в stills/.

Запуск из корня репозитория или откуда угодно:
  python business/demo/make_demo_animation.py

Данные: 2 215 локальных depth-кадров под data/apc/ (gitignored). RGB в
демо нет и не может быть — изображения не существует, в этом и суть.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "development" / "src"))

from sanas.door_apc import (  # noqa: E402
    ApcConfig,
    NearestNeighbourTracker,
    build_multizone_model,
    extract_centroids,
    multizone_background,
    multizone_foreground_points,
    multizone_ranges,
)
from sanas.depth_occupancy import (  # noqa: E402
    load_depth_metres,
    load_extrinsics,
    load_intrinsics,
)

# ==========================================================================
# ПАРАМЕТРЫ (всё настраиваемое — здесь, не в теле функций)
# ==========================================================================

DATA = REPO / "data" / "apc"
CAMERA = "center_left_depth"
FRAME_DIR = DATA / "cameras" / "depth" / CAMERA / "images"
MANIFEST = DATA / "apc_manifest.json"
VALIDATION_JSON = REPO / "research" / "paper" / "results" / "apc_validation.json"
OUT_DIR = Path(__file__).resolve().parent
STILLS_DIR = OUT_DIR / "stills"

# Индексы эпизодных чанков манифеста, попадающие в ролик (по порядку показа).
# 0  = посадка (pid 1); 1 = высадка + посадка (pid 1);
# 11 = высадка + посадка (pid 119); 15 = высадка + посадка (pid 135, случай
# "завис на линии", который полное разрешение пропускает, а 8x8 ловит).
CHOSEN_CHUNKS = [0, 1, 11, 15]
WINDOW_DESC = {
    0: "пассажир входит в автобус",
    1: "пассажир выходит и входит снова",
    11: "пассажир выходит и входит снова",
    15: "пассажир выходит и входит снова",
}

# Тайминги, секунды
INTRO_S = 6.0  # заставка
MINICARD_S = 1.8  # карточка перед каждым окном (склейка-«cut»)
FREEZE_S = 1.1  # пауза на каждом событии (+1/−1)
FLASH_TAIL_S = 1.0  # сколько ещё горит подпись после паузы
FINAL_S = 12.0  # финальная честная заставка

# Видео
MP4_FPS = 15  # 1 видеокадр = 1 реальный кадр (~7.7 Гц) => ~2x реального времени
GIF_FPS = 10  # fallback без ffmpeg
GIF_PLAY_EVERY = 2  # в GIF показываем каждый 2-й кадр (скорость ~2.6x)
FIGSIZE = (12.8, 7.2)  # 1280x720 при dpi 100
DPI = 100

# Тепловая карта 8x8: ярче = ближе. Диапазон датчика 0.3–3.0 м (ApcConfig),
# для контраста растягиваем яркость по фактическому рабочему окну сцены.
GRID_NEAR_M = 0.8
GRID_FAR_M = 3.0

# Цвета
C_BG_CARD = "#101622"  # фон заставок
C_BG_PLAY = "#f7f8fa"  # фон рабочих кадров
C_TEXT = "#16181d"
C_TEXT_LIGHT = "#f2f4f8"
C_MUTED = "#6a7180"
C_BOARD = "#1d9e50"  # посадка
C_ALIGHT = "#e05a1e"  # высадка
C_STEP = "#155a9c"  # линия счётчика
C_GRID_PANEL = "#0d1017"  # фон панели датчика
C_GRID_LINE = "#3a4150"
C_CURSOR = "#c8102e"
C_WINDOW_SHADE = "#9ec7e8"

# Тексты (обязательные формулировки — не менять без согласования)
TITLE_SENSOR = "Что видит датчик: 64 расстояния."
TITLE_SENSOR2 = "Изображения не существует."
FINAL_LINE1 = "Симуляция датчика на публичных записях реального автобуса."
FINAL_LINE2 = "Валидация: {tp} из {n} событий на {n}-событийном наборе."
FINAL_LINE3 = "Следующий шаг — пилот с реальным датчиком."

NS = 1_000_000_000


# ==========================================================================
# Данные: прогон настоящего конвейера на выбранных окнах
# ==========================================================================


def frame_path(t: int) -> str:
    return str(FRAME_DIR / f"{t}.png")


def mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


def run_pipeline() -> dict:
    """Пулинг в 8x8 и трекинг на выбранных чанках, сверка с валидацией."""
    with open(MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    cfg = ApcConfig()
    intr = load_intrinsics(
        str(DATA / "cameras" / "depth" / CAMERA / "camera_info.yaml")
    )
    T = load_extrinsics(str(DATA), CAMERA)
    mz8 = build_multizone_model(intr, 8, cfg)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice")
        bg8 = multizone_background(
            [
                multizone_ranges(load_depth_metres(frame_path(t)), mz8, cfg)
                for t in man["background_ts"]
            ]
        )

    chunks = [man["episode_chunks"][i] for i in CHOSEN_CHUNKS]
    missing = [
        t for ch in chunks for t in ch["ts"] if not os.path.exists(frame_path(t))
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} кадров выбранных окон нет на диске "
            f"(первый: {missing[0]}) — анимацию собрать нельзя"
        )

    # Покадровый прогон: тот же цикл, что run_chunk(), но с покадровой
    # привязкой событий (нужно знать, НА каком кадре сработал счётчик).
    # Логика целиком из door_apc: multizone_ranges -> foreground -> centroids
    # -> NearestNeighbourTracker. Трекер сбрасывается на каждом чанке,
    # как в конвейере.
    windows = []
    for wi, (ci, ch) in enumerate(zip(CHOSEN_CHUNKS, chunks)):
        ts = ch["ts"]
        if ch.get("full_rate"):  # spot-check чанки прорежаем до ~7.7 Гц,
            ts = [t for i, t in enumerate(ts) if i % 2 == 0]  # как process_all()
        tracker = NearestNeighbourTracker(cfg)
        frames = []
        for t in ts:
            depth = load_depth_metres(frame_path(t))
            r8 = multizone_ranges(depth, mz8, cfg)
            pts = multizone_foreground_points(r8, bg8, mz8, T, cfg)
            cents = extract_centroids(
                pts, cfg.mz_cluster_cell_m, cfg.mz_min_cluster_points
            )
            evs = tracker.step(t, cents)
            frames.append(
                {"t_ns": t, "grid": r8.reshape(8, 8), "events": [e.kind for e in evs]}
            )
        windows.append(
            {"win": wi, "chunk": ci, "start_ns": ch["start_ns"], "frames": frames}
        )

    # Сверка с research/paper/results/apc_validation.json: события должны совпасть
    # с валидированными первыми событиями эпизодов этих чанков 1-в-1.
    with open(VALIDATION_JSON, encoding="utf-8") as fh:
        val = json.load(fh)
    mz8v = val["variants"]["mz8"]
    bounds = [(ch["start_ns"], ch["end_ns"]) for ch in chunks]
    expected = sorted(
        (ep["first_event_t_ns"], ep["first_event_kind"])
        for ep in mz8v["per_episode"]
        if ep["first_event_t_ns"] is not None
        and any(a <= ep["first_event_t_ns"] <= b for a, b in bounds)
    )
    got = sorted(
        (f["t_ns"], k) for w in windows for f in w["frames"] for k in f["events"]
    )
    if [(t, k) for t, k in expected] != got:
        raise SystemExit(
            "Прогон разошёлся с apc_validation.json:\n"
            f"  ожидалось: {expected}\n  получено:  {got}"
        )
    print(f"события сверены с apc_validation.json: {len(got)} шт., совпадение 1-в-1")

    # Временная шкала всей сессии + маркеры всех 25 эпизодов
    all_chunks = man["episode_chunks"] + man["negative_chunks"]
    t0 = min(c["start_ns"] for c in all_chunks)
    t_end = max(c["end_ns"] for c in all_chunks)
    episodes = [
        {
            "mid_s": ((ep["start_ns"] + ep["end_ns"]) / 2 - t0) / NS,
            "kind": ep["kind"],
            "detected": ep["detected"],
        }
        for ep in mz8v["per_episode"]
    ]
    return {
        "windows": windows,
        "session_t0": t0,
        "session_len_s": (t_end - t0) / NS,
        "episodes": episodes,
        "tp": mz8v["true_positives"],
        "n_ep": mz8v["episodes"],
    }


# ==========================================================================
# Раскадровка
# ==========================================================================


def build_playback(data: dict) -> dict:
    """Плоский список воспроизводимых кадров + счётчик + линия-ступенька."""
    t0 = data["session_t0"]
    flat = []
    shown = 0.0
    n = 0
    for w in data["windows"]:
        w_first = w["frames"][0]["t_ns"]
        for i, f in enumerate(w["frames"]):
            if i > 0:
                shown += (f["t_ns"] - w["frames"][i - 1]["t_ns"]) / NS
            for k in f["events"]:
                n += 1 if k == "board" else -1
            flat.append(
                {
                    "win": w["win"],
                    "t_ns": f["t_ns"],
                    "rec_s": (f["t_ns"] - t0) / NS,
                    "win_rec_s": (w_first - t0) / NS,
                    "shown_s": shown,
                    "grid": f["grid"],
                    "events": f["events"],
                    "n_after": n,
                }
            )
    joins = []  # границы окон на оси показанного времени
    for a, b in zip(flat, flat[1:]):
        if b["win"] != a["win"]:
            joins.append((a["shown_s"] + 0.2, b["win"]))
    return {"flat": flat, "joins": joins, "shown_total": shown}


def build_specs(data: dict, pb: dict, fps: int, play_every: int) -> list[dict]:
    """Список видеокадров: карточки, кадры окна, паузы на событиях."""
    specs: list[dict] = []
    specs += [{"mode": "intro"}] * int(round(INTRO_S * fps))
    flat = pb["flat"]
    freeze_n = int(round(FREEZE_S * fps))
    tail_n = int(round(FLASH_TAIL_S * fps))
    cur_win = -1
    flash: tuple[str, int] | None = None  # (kind, кадров осталось)
    i = 0
    while i < len(flat):
        f = flat[i]
        if f["win"] != cur_win:
            cur_win = f["win"]
            flash = None
            specs += [
                {
                    "mode": "minicard",
                    "win": cur_win,
                    "rec_s": f["win_rec_s"],
                    "desc": WINDOW_DESC[CHOSEN_CHUNKS[cur_win]],
                }
            ] * int(round(MINICARD_S * fps))
        # события с прореженных кадров (GIF) не теряем: собираем до шага
        kinds = [
            k
            for j in range(i, min(i + play_every, len(flat)))
            for k in flat[j]["events"]
        ]
        f_show = flat[min(i + play_every - 1, len(flat) - 1)] if kinds else f
        if kinds:
            k = kinds[-1]  # на кадре максимум одно событие; последнее — актуальное
            specs += [
                {"mode": "play", "f": f_show, "flash": k, "freeze": True}
            ] * freeze_n
            flash = (k, tail_n)
        else:
            fl = None
            if flash and flash[1] > 0:
                fl = flash[0]
                flash = (flash[0], flash[1] - 1)
            specs.append({"mode": "play", "f": f, "flash": fl, "freeze": False})
        i += play_every
    specs += [{"mode": "final"}] * int(round(FINAL_S * fps))
    return specs


# ==========================================================================
# Отрисовка
# ==========================================================================


def _card_bg(fig):
    fig.clf()
    fig.patch.set_facecolor(C_BG_CARD)


def draw_intro(fig, data: dict) -> None:
    _card_bg(fig)
    fig.text(
        0.5,
        0.72,
        "Sanas",
        ha="center",
        color=C_TEXT_LIGHT,
        fontsize=52,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.60,
        "Подсчёт пассажиров датчиком-дальномером",
        ha="center",
        color=C_TEXT_LIGHT,
        fontsize=22,
    )
    fig.text(
        0.5,
        0.44,
        "Симуляция дешёвого 8×8-датчика на записях из настоящего\n"
        "городского автобуса (открытый научный датасет, Германия)",
        ha="center",
        color="#aeb6c4",
        fontsize=16,
        linespacing=1.5,
    )
    fig.text(
        0.5,
        0.30,
        "Датчик видит только 64 расстояния. Изображения не существует.",
        ha="center",
        color=C_TEXT_LIGHT,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.14,
        "воспроизведение ≈2× быстрее реального времени · пауза на каждом событии",
        ha="center",
        color=C_MUTED,
        fontsize=12,
    )


def draw_minicard(fig, win: int, rec_s: float, desc: str) -> None:
    _card_bg(fig)
    fig.text(
        0.5,
        0.60,
        f"Окно {win + 1} из {len(CHOSEN_CHUNKS)}",
        ha="center",
        color=C_TEXT_LIGHT,
        fontsize=34,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.46,
        f"{mmss(rec_s)} от начала записи",
        ha="center",
        color="#aeb6c4",
        fontsize=19,
    )
    fig.text(0.5, 0.35, desc, ha="center", color=C_TEXT_LIGHT, fontsize=21)


def draw_final(fig, data: dict) -> None:
    _card_bg(fig)
    fig.text(
        0.5, 0.80, "Sanas", ha="center", color=C_MUTED, fontsize=20, fontweight="bold"
    )
    fig.text(0.5, 0.62, FINAL_LINE1, ha="center", color=C_TEXT_LIGHT, fontsize=21)
    fig.text(
        0.5,
        0.50,
        FINAL_LINE2.format(tp=data["tp"], n=data["n_ep"]),
        ha="center",
        color=C_TEXT_LIGHT,
        fontsize=25,
        fontweight="bold",
    )
    fig.text(0.5, 0.38, FINAL_LINE3, ha="center", color=C_TEXT_LIGHT, fontsize=21)


def draw_play(fig, data: dict, pb: dict, spec: dict, speed_label: str) -> None:
    fig.clf()
    fig.patch.set_facecolor(C_BG_PLAY)
    f = spec["f"]
    flash = spec["flash"]
    freeze = spec["freeze"]

    # --- левая панель: 8x8 --------------------------------------------------
    ax = fig.add_axes([0.045, 0.21, 0.46, 0.615])
    ax.set_facecolor(C_GRID_PANEL)
    grid = f["grid"]
    b = np.clip((GRID_FAR_M - grid) / (GRID_FAR_M - GRID_NEAR_M), 0.0, 1.0) ** 0.9
    gray = 0.12 + 0.82 * b
    rgb = np.stack([gray * 0.96, gray * 0.98, gray], axis=-1)
    rgb[grid <= 0] = (0.05, 0.06, 0.10)  # нет отражения — вне диапазона
    ax.imshow(rgb, extent=(0, 8, 8, 0), interpolation="nearest")
    for k in range(9):
        ax.plot([0, 8], [k, k], color=C_GRID_LINE, lw=1.0)
        ax.plot([k, k], [0, 8], color=C_GRID_LINE, lw=1.0)
    for r in range(8):
        for c in range(8):
            v = grid[r, c]
            txt = f"{v:.1f}".replace(".", ",") if v > 0 else "–"
            dark_cell = gray[r, c] < 0.55 or v <= 0
            ax.text(
                c + 0.5,
                r + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=10.5,
                color="#c9ccd4" if dark_cell else "#10131a",
            )
    ax.set_xlim(0, 8)
    ax.set_ylim(8, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(C_GRID_LINE)
    ax.set_title(
        f"{TITLE_SENSOR}\n{TITLE_SENSOR2}",
        fontsize=16,
        fontweight="bold",
        color=C_TEXT,
        pad=12,
    )
    fig.text(
        0.275,
        0.165,
        "числа — расстояние в метрах · ярче = ближе · «–» — вне диапазона (до 3 м)",
        ha="center",
        fontsize=11,
        color=C_MUTED,
    )

    # --- правая панель: счётчик --------------------------------------------
    n = f["n_after"]
    box_ec = {"board": C_BOARD, "alight": C_ALIGHT}.get(flash, "#c8cdd6")
    fig.text(0.755, 0.915, "В салоне:", ha="center", fontsize=21, color=C_TEXT)
    fig.text(
        0.755,
        0.765,
        str(n),
        ha="center",
        va="center",
        fontsize=70,
        fontweight="bold",
        color=C_TEXT,
        bbox=dict(
            boxstyle="round,pad=0.3", fc="white", ec=box_ec, lw=4 if flash else 1.5
        ),
    )
    fig.text(
        0.755,
        0.60,
        "отсчёт с начала ролика · на конечной — сброс в 0",
        ha="center",
        fontsize=10.5,
        color=C_MUTED,
    )
    if flash == "board":
        fig.text(
            0.755,
            0.535,
            "+1 посадка",
            ha="center",
            fontsize=28,
            fontweight="bold",
            color=C_BOARD,
        )
    elif flash == "alight":
        fig.text(
            0.755,
            0.535,
            "−1 высадка",
            ha="center",
            fontsize=28,
            fontweight="bold",
            color=C_ALIGHT,
        )

    # --- линия-ступенька -----------------------------------------------------
    axs = fig.add_axes([0.575, 0.235, 0.385, 0.245])
    axs.set_facecolor("white")
    flat = pb["flat"]
    idx = next(i for i, ff in enumerate(flat) if ff is f)
    xs = [ff["shown_s"] for ff in flat[: idx + 1]]
    ys = [ff["n_after"] for ff in flat[: idx + 1]]
    axs.plot(xs, ys, color=C_STEP, lw=2.5, drawstyle="steps-post")
    for ff in flat[: idx + 1]:
        for k in ff["events"]:
            axs.plot(
                ff["shown_s"],
                ff["n_after"],
                marker="^" if k == "board" else "v",
                ms=9,
                color=C_BOARD if k == "board" else C_ALIGHT,
                zorder=5,
            )
    for jx, jw in pb["joins"]:
        axs.axvline(jx, color="#b9bfc9", lw=1, ls="--")
    axs.set_xlim(0, pb["shown_total"] + 1.5)
    axs.set_ylim(-0.4, 2.4)
    axs.set_yticks([0, 1, 2])
    axs.set_xlabel("показанное время записи, с", fontsize=11, color=C_MUTED)
    axs.tick_params(labelsize=10, colors=C_MUTED)
    axs.grid(True, color="#e3e6ea", lw=0.8)
    for s in axs.spines.values():
        s.set_color("#c8cdd6")

    # --- нижняя полоса: таймлайн --------------------------------------------
    axt = fig.add_axes([0.045, 0.035, 0.925, 0.085])
    axt.set_facecolor("white")
    sess = data["session_len_s"]
    for w in data["windows"]:
        a = (w["frames"][0]["t_ns"] - data["session_t0"]) / NS
        bnd = (w["frames"][-1]["t_ns"] - data["session_t0"]) / NS
        axt.axvspan(a, bnd, color=C_WINDOW_SHADE, alpha=0.85)
        axt.text(
            (a + bnd) / 2,
            0.5,
            str(w["win"] + 1),
            ha="center",
            va="center",
            fontsize=8,
            color=C_STEP,
            fontweight="bold",
        )
    for ep in data["episodes"]:
        axt.plot(
            ep["mid_s"],
            0.68 if ep["kind"] == "board" else 0.32,
            marker="^" if ep["kind"] == "board" else "v",
            ms=6,
            color=C_BOARD if ep["kind"] == "board" else C_ALIGHT,
            mec="none",
            alpha=0.95,
            zorder=4,
        )
    axt.axvline(f["rec_s"], color=C_CURSOR, lw=2, zorder=6)
    axt.set_xlim(0, sess)
    axt.set_ylim(0, 1)
    axt.set_yticks([])
    ticks = np.arange(0, sess + 1, 300)
    axt.set_xticks(ticks)
    axt.set_xticklabels([mmss(t) for t in ticks], fontsize=9, color=C_MUTED)
    for s in axt.spines.values():
        s.set_color("#c8cdd6")
    fig.text(
        0.045,
        0.135,
        f"Окно {f['win'] + 1} из {len(CHOSEN_CHUNKS)}",
        fontsize=12,
        color=C_TEXT,
        fontweight="bold",
    )
    fig.text(
        0.97,
        0.135,
        f"Запись: {mmss(f['rec_s'])} из {mmss(sess)}   ▲ посадка · ▼ высадка",
        ha="right",
        fontsize=11,
        color=C_MUTED,
    )
    fig.text(0.985, 0.968, speed_label, ha="right", fontsize=11, color=C_MUTED)
    if freeze:
        fig.text(0.985, 0.935, "пауза", ha="right", fontsize=11, color=C_CURSOR)


def draw(fig, data: dict, pb: dict, spec: dict, speed_label: str) -> None:
    if spec["mode"] == "intro":
        draw_intro(fig, data)
    elif spec["mode"] == "minicard":
        draw_minicard(fig, spec["win"], spec["rec_s"], spec["desc"])
    elif spec["mode"] == "final":
        draw_final(fig, data)
    else:
        draw_play(fig, data, pb, spec, speed_label)


# ==========================================================================
# Сборка
# ==========================================================================


def main() -> int:
    t_start = time.time()
    stills_only = "--stills-only" in sys.argv[1:]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STILLS_DIR.mkdir(parents=True, exist_ok=True)

    data = run_pipeline()
    pb = build_playback(data)

    have_ffmpeg = shutil.which("ffmpeg") is not None
    fps = MP4_FPS if have_ffmpeg else GIF_FPS
    play_every = 1 if have_ffmpeg else GIF_PLAY_EVERY
    # фактическая скорость чистого воспроизведения (без пауз на событиях)
    n_play = sum(1 for _ in range(0, len(pb["flat"]), play_every))
    speed = pb["shown_total"] / (n_play / fps)
    speed_label = f"≈{speed:.0f}× реального времени"
    specs = build_specs(data, pb, fps, play_every)
    dur = len(specs) / fps
    print(
        f"окна: {len(data['windows'])} шт., показанного времени записи "
        f"{pb['shown_total']:.1f} с, видеокадров {len(specs)}, "
        f"длительность ролика {dur:.1f} с при {fps} fps (скорость {speed:.2f}x)"
    )

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    # --- статичные кадры для слайдов ----------------------------------------
    flat = pb["flat"]
    ev_idx = [i for i, f in enumerate(flat) if f["events"]]
    first_board = next(i for i in ev_idx if flat[i]["events"][0] == "board")
    w3_alight = next(
        i for i in ev_idx if flat[i]["events"][0] == "alight" and flat[i]["win"] == 2
    )
    stills = [
        (
            "still_01_sensor_view.png",
            {
                "mode": "play",
                "f": flat[max(0, first_board - 1)],
                "flash": None,
                "freeze": False,
            },
        ),
        (
            "still_02_boarding.png",
            {"mode": "play", "f": flat[first_board], "flash": "board", "freeze": True},
        ),
        (
            "still_03_alighting.png",
            {"mode": "play", "f": flat[w3_alight], "flash": "alight", "freeze": True},
        ),
        ("still_04_final_card.png", {"mode": "final"}),
    ]
    for name, spec in stills:
        draw(fig, data, pb, spec, speed_label)
        fig.savefig(STILLS_DIR / name, dpi=DPI, facecolor=fig.get_facecolor())
        print(f"кадр для слайда: {STILLS_DIR / name}")

    # --- видео ---------------------------------------------------------------
    if stills_only:
        plt.close(fig)
        print("--stills-only: видео не пересобиралось")
        return 0
    if have_ffmpeg:
        out = OUT_DIR / "sanas_demo.mp4"
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"],
        )
    else:
        out = OUT_DIR / "sanas_demo.gif"
        writer = animation.PillowWriter(fps=fps)
    with writer.saving(fig, str(out), DPI):
        for i, spec in enumerate(specs):
            draw(fig, data, pb, spec, speed_label)
            writer.grab_frame()
            if i % 150 == 0:
                print(f"  рендер {i}/{len(specs)} кадров...")
    plt.close(fig)
    size_mb = out.stat().st_size / 1e6
    print(
        f"готово: {out} ({size_mb:.1f} МБ, {dur:.1f} с) "
        f"за {time.time() - t_start:.0f} с"
    )
    if size_mb > 25:
        print("ВНИМАНИЕ: файл больше 25 МБ — уменьшите fps или размер кадра")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
