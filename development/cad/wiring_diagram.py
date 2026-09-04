"""Генератор схемы подключения прототипа Sanas в SVG.

Статус: proposed. Ни один узел не куплен, не собран и не измерен.
Дата: 2026-09-03
Authority: подчиняется ../../GROUND_TRUTH.md раздел 4 и
../TECH_DATA_OPTIONS.md раздел "Compute/recording risks".

Почему SVG, а не KiCad
----------------------
KiCad в рабочем окружении не установлен, поэтому по правилу fallback
схема собирается как самодостаточный SVG на стандартной библиотеке.
Здесь и нет схемотехники в смысле цепей и номиналов: это схема
межблочных соединений готовых модулей, для неё netlist и footprint не
нужны. Когда появится собственная плата или PD-модуль на своей плате,
схему надо переносить в KiCad, а не расширять этот файл.

Что схема утверждает и чего не утверждает
-----------------------------------------
Она фиксирует ЦЕПЬ и ОТКРЫТЫЕ ВОПРОСЫ по каждому узлу, а не
работоспособность. Проверенным считается только то, что помечено
статусом "проверено" и имеет ссылку на первичный источник в
wiring_diagram.md.

Запуск:
    python development/cad/wiring_diagram.py
    python development/cad/wiring_diagram.py -o путь/файл.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

# =====================================================================
# ПАРАМЕТРЫ
# =====================================================================

TITLE = "Sanas: схема подключения прототипа (proposed)"
SUBTITLE = (
    "Источник цепи: GROUND_TRUTH.md раздел 4. Ни один узел не куплен и не "
    "проверен на стенде."
)
DATE = "2026-09-03"

CANVAS_W = 1000  # px
CANVAS_H = 720  # px
FONT = "DejaVu Sans, Verdana, sans-serif"

# Цвет по статусу. Статусы те же, что в GROUND_TRUTH.md раздел 1.
STATUS_FILL = {
    "проверено": "#dff0d8",
    "решено": "#e8eef7",
    "кандидат": "#fdf3d8",
    "открыто": "#fbe3e3",
}
STATUS_STROKE = {
    "проверено": "#4a7a3a",
    "решено": "#3c5a86",
    "кандидат": "#a8842a",
    "открыто": "#a83a3a",
}

# Узлы: id -> (x, y, w, h, заголовок, статус, строки описания)
NODES = {
    "bank": (
        30,
        60,
        210,
        96,
        "USB-C PD power bank",
        "решено",
        ["65 W+, модель не выбрана", "ёмкость Wh не выбрана"],
    ),
    "trigger": (
        290,
        60,
        200,
        96,
        "USB-C PD trigger",
        "открыто",
        ["12 V или 15 V, не выбрано", "+ ответный barrel plug"],
    ),
    "meter": (
        540,
        60,
        180,
        96,
        "Ваттметр в разрыв",
        "кандидат",
        ["только на стенде", "нужен для бюджета питания"],
    ),
    "jetson": (
        640,
        250,
        300,
        210,
        "Jetson Orin Nano Super Dev Kit",
        "решено",
        [
            "DC barrel jack, вход 9-20 V (проверено)",
            "USB-C: данные и recovery, НЕ питание",
            "CSI CAM0 / CAM1",
            "M.2 Key M под NVMe",
            "ambient 0-35 C (проверено)",
            "NVENC отсутствует (проверено)",
        ],
    ),
    "camera": (
        30,
        250,
        250,
        96,
        "Waveshare IMX219-160",
        "кандидат",
        ["8 MP, 160 град по диагонали", "драйвер и FPS на Orin не проверены"],
    ),
    "nvme": (
        30,
        390,
        250,
        96,
        "NVMe SSD",
        "открыто",
        ["модель, объём, ресурс записи", "и retention не выбраны"],
    ),
    "host": (
        30,
        530,
        250,
        96,
        "Хост для прошивки",
        "открыто",
        ["рабочая машина под Windows 11", "SDK Manager требует Linux"],
    ),
}

# Рёбра: список точек ломаной, подпись, подпись второй строкой, стиль
EDGES = [
    ([(240, 108), (290, 108)], "USB-C, PD", "", "solid"),
    ([(490, 108), (540, 108)], "DC 12/15 V", "", "solid"),
    (
        [(720, 108), (790, 108), (790, 250)],
        "DC barrel jack",
        "9-20 V, полярность и размер штекера не проверены",
        "solid",
    ),
    (
        [(280, 298), (640, 298)],
        "CSI-2, шлейф FFC",
        "число контактов и шаг не сверены",
        "solid",
    ),
    (
        [(280, 438), (460, 438), (460, 390), (640, 390)],
        "M.2 Key M",
        "ключ и длина 2280 не сверены",
        "solid",
    ),
    (
        [(280, 578), (520, 578), (520, 440), (640, 440)],
        "USB-C, только данные",
        "питание сюда не подавать",
        "dashed",
    ),
]

LEGEND_X = 30
LEGEND_Y = 655
LEGEND_STEP = 170

NOTES = [
    "Открытый блокер: USB-C на dev kit это порт данных и recovery. Питание идёт только в barrel jack,",
    "поэтому power bank не питает кит напрямую и PD trigger обязателен (GROUND_TRUTH.md раздел 4).",
]

# =====================================================================


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def node_svg(node_id: str) -> list[str]:
    x, y, w, h, title, status, lines = NODES[node_id]
    fill = STATUS_FILL[status]
    stroke = STATUS_STROKE[status]
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>',
        f'<text x="{x + 12}" y="{y + 24}" font-family="{FONT}" font-size="14" '
        f'font-weight="bold" fill="#111">{esc(title)}</text>',
        f'<text x="{x + w - 12}" y="{y + 24}" font-family="{FONT}" font-size="11" '
        f'text-anchor="end" fill="{stroke}">{esc(status)}</text>',
    ]
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x + 12}" y="{y + 44 + i * 16}" font-family="{FONT}" '
            f'font-size="11.5" fill="#333">{esc(line)}</text>'
        )
    return out


def edge_svg(points, label, sublabel, style) -> list[str]:
    dash = ' stroke-dasharray="7 5"' if style == "dashed" else ""
    path = " ".join(f"{px},{py}" for px, py in points)
    out = [
        f'<polyline points="{path}" fill="none" stroke="#333" '
        f'stroke-width="2"{dash} marker-end="url(#arrow)"/>'
    ]
    mid_i = len(points) // 2
    mx = (points[mid_i - 1][0] + points[mid_i][0]) / 2
    my = (points[mid_i - 1][1] + points[mid_i][1]) / 2
    if label:
        out.append(
            f'<text x="{mx}" y="{my - 8}" font-family="{FONT}" font-size="11.5" '
            f'text-anchor="middle" fill="#111">{esc(label)}</text>'
        )
    if sublabel:
        out.append(
            f'<text x="{mx}" y="{my + 16}" font-family="{FONT}" font-size="10.5" '
            f'text-anchor="middle" fill="#a83a3a">{esc(sublabel)}</text>'
        )
    return out


def legend_svg() -> list[str]:
    out = []
    for i, status in enumerate(STATUS_FILL):
        x = LEGEND_X + i * LEGEND_STEP
        out.append(
            f'<rect x="{x}" y="{LEGEND_Y - 12}" width="16" height="16" rx="3" '
            f'fill="{STATUS_FILL[status]}" stroke="{STATUS_STROKE[status]}"/>'
        )
        out.append(
            f'<text x="{x + 24}" y="{LEGEND_Y + 1}" font-family="{FONT}" '
            f'font-size="12" fill="#333">{esc(status)}</text>'
        )
    return out


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" '
        f'height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" '
        'refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#333"/>'
        "</marker></defs>",
        f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#ffffff"/>',
        f'<text x="30" y="32" font-family="{FONT}" font-size="18" '
        f'font-weight="bold" fill="#111">{esc(TITLE)}</text>',
        f'<text x="30" y="50" font-family="{FONT}" font-size="11.5" '
        f'fill="#555">{esc(SUBTITLE)}</text>',
        f'<text x="{CANVAS_W - 30}" y="32" font-family="{FONT}" font-size="11.5" '
        f'text-anchor="end" fill="#555">{esc(DATE)}</text>',
    ]
    for edge in EDGES:
        parts += edge_svg(*edge)
    for node_id in NODES:
        parts += node_svg(node_id)
    parts += legend_svg()
    for i, note in enumerate(NOTES):
        parts.append(
            f'<text x="30" y="{CANVAS_H - 40 + i * 15}" font-family="{FONT}" '
            f'font-size="11.5" fill="#a83a3a">{esc(note)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    default_out = Path(__file__).resolve().parent / "wiring_diagram.svg"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=default_out)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_svg(), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
