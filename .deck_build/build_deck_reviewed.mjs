import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:\\Users\\User\\OneDrive\\Desktop\\me\\Sanash";
const skillDir = "C:\\Users\\User\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.904.11930\\skills\\presentations";
const buildDir = path.join(workspaceDir, ".deck_build");
const assetsDir = path.join(buildDir, "assets");
const outputDir = path.join(workspaceDir, "deliverables");
const finalName = process.env.FINAL_NAME || "Sanash_technical_pitch_ru_reviewed.pptx";
const finalPath = path.join(outputDir, finalName);
const runtimePython = "C:\\Users\\User\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";

const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
const fontFamily = resolvePresentationFont({ fontFamily: "Aptos" });

const C = {
  ink: "#07131F", navy: "#0B2136", blue: "#1759D1", cyan: "#19D7C2",
  green: "#30C982", lime: "#A8E063", amber: "#F4B942", red: "#F0645A",
  white: "#F7FAFC", paper: "#F2F5F7", slate: "#526273", pale: "#DCE6ED",
  grid: "#CBD6DE", mid: "#8FA7BA", darkGrid: "#29455A",
};

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const asset = async (name) => new Uint8Array(await fs.readFile(path.join(assetsDir, name)));

function rect(slide, left, top, width, height, fill, radius = 0, line = null) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: line || { fill: "none", width: 0 },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function textBox(slide, text, left, top, width, height, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: opts.fill || "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: opts.typeface || fontFamily,
    fontSize: opts.fontSize ?? 24,
    bold: opts.bold ?? false,
    color: opts.color || C.ink,
    alignment: opts.align || "left",
    verticalAlignment: opts.vAlign || "top",
    autoFit: "none",
    wrap: "square",
    lineSpacing: opts.lineSpacing ?? 1,
    insets: opts.insets || { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return box;
}

function image(slide, bytes, type, alt, position, fit = "cover", radius = 0, crop = null) {
  return slide.images.add({
    blob: bytes, contentType: type, alt, fit, position,
    ...(radius ? { geometry: "roundRect", borderRadius: radius } : {}),
    ...(crop ? { crop } : {}),
  });
}

function header(slide, title, subtitle, dark = false) {
  textBox(slide, title, 64, 38, 830, 58, { fontSize: 35, bold: true, color: dark ? C.white : C.ink });
  rect(slide, 64, 104, 72, 5, C.cyan, 3);
  if (subtitle) textBox(slide, subtitle, 910, 45, 300, 42, { fontSize: 14, color: dark ? C.pale : C.slate, align: "right", lineSpacing: 1.05 });
}

function footer(slide, n, dark = false, label = "SANASH · технический питч") {
  textBox(slide, label, 64, 682, 680, 18, { fontSize: 11, bold: true, color: dark ? C.mid : "#788997" });
  textBox(slide, String(n).padStart(2, "0"), 1160, 680, 48, 18, { fontSize: 11, bold: true, color: dark ? C.cyan : C.blue, align: "right" });
}

function chip(slide, label, left, top, fill, color = C.ink, width = 140) {
  rect(slide, left, top, width, 30, fill, 15);
  textBox(slide, label, left + 10, top + 7, width - 20, 16, { fontSize: 11, bold: true, color, align: "center", vAlign: "middle" });
}

function rule(slide, x1, y1, x2, y2, color = C.grid, width = 2) {
  return slide.shapes.add({ geometry: "line", position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 }, fill: "none", line: { fill: color, width } });
}

function notes(slide, lines) { slide.speakerNotes.textFrame.setText(lines); }

// 1 — Cover
{
  const slide = deck.slides.add();
  slide.background.fill = C.ink;
  const buses = await asset("almaty_bus_fleet_hq.png");
  image(slide, buses, "image/png", "Парк зелёных автобусов Алматы", { left: 565, top: 0, width: 715, height: 720 }, "cover", 0, { left: 0.08, top: 0, right: 0.02, bottom: 0 });
  rect(slide, 0, 0, 615, 720, C.ink);
  rect(slide, 555, 0, 60, 720, { color: C.ink, transparency: 0.15 });
  textBox(slide, "SANASH", 72, 58, 320, 28, { fontSize: 18, bold: true, color: C.cyan });
  textBox(slide, "Загрузка салона —\nвидна до посадки", 72, 162, 480, 150, { fontSize: 52, bold: true, color: C.white, lineSpacing: 0.92 });
  textBox(slide, "Предложение ограниченного технического пилота для Avtobys", 74, 350, 430, 76, { fontSize: 23, color: C.pale, lineSpacing: 1.12 });
  chip(slide, "СТАТУС: ПРОТОТИП", 74, 482, C.amber, C.ink, 190);
  textBox(slide, "Полевая валидация в автобусе ещё не проведена", 74, 527, 420, 56, { fontSize: 18, color: C.white });
  textBox(slide, "Алматы · 2026", 74, 634, 300, 24, { fontSize: 14, bold: true, color: C.mid });
  notes(slide, [
    "Фото: парк автобусов Алматы. Источник: Almaty TV.",
    "https://almaty.tv/ru/news/transport/1328-kak-v-almaty-moderniziruyut-marshrutnuyu-set",
    "Статус продукта: локальный README.md проекта Sanash; полевая валидация не заявляется.",
  ]);
}

// 2 — Survey signal
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  header(slide, "Исследовательский сигнал, а не доказанный эффект", "опрос заявленных предпочтений · волна 1");
  const chart = slide.charts.add("bar", {
    position: { left: 54, top: 155, width: 675, height: 330 },
    categories: ["Переполнен", "Есть места стоя"],
    series: [{
      name: "Выберут следующий автобус",
      values: [0.689, 0.428], valuesFormatCode: "0.0%", fill: C.blue,
      points: [{ idx: 0, fill: C.red }, { idx: 1, fill: C.cyan }],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 52 },
    hasLegend: false,
    xAxis: { visible: true, min: 0, max: 0.8, majorUnit: 0.2, numberFormatCode: "0%", majorGridlines: { style: "solid", fill: C.grid, width: 1 }, line: { fill: C.grid, width: 1 } },
    yAxis: { visible: true, line: { fill: "none", width: 0 }, textStyle: { typeface: fontFamily, fontSize: 16, fill: C.ink } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: fontFamily, fontSize: 18, fill: C.ink, bold: true } },
  });
  applyPresentationChartFont(chart, { fontFamily });
  textBox(slide, "n = 208", 603, 195, 95, 22, { fontSize: 13, bold: true, color: C.slate, align: "right" });
  textBox(slide, "n = 209", 462, 342, 95, 22, { fontSize: 13, bold: true, color: C.slate, align: "right" });
  textBox(slide, "+26,1 п.п.", 70, 505, 280, 54, { fontSize: 42, bold: true, color: C.blue });
  textBox(slide, "при одинаковом ожидании 5 минут и одинаковом часе пик", 72, 561, 540, 46, { fontSize: 18, color: C.slate });
  const crowd = await asset("almaty_crowded_bus.png");
  image(slide, crowd, "image/png", "Переполненный салон автобуса Алматы", { left: 770, top: 146, width: 442, height: 468 }, "cover", 22, { left: 0.14, top: 0.03, right: 0.08, bottom: 0.02 });
  rect(slide, 794, 474, 394, 116, { color: C.ink, transparency: 0.06 }, 16);
  textBox(slide, "7,96 мин", 816, 492, 180, 35, { fontSize: 29, bold: true, color: C.cyan });
  textBox(slide, "оценочная готовность ждать; 95% ДИ [5,89; 11,09]", 816, 536, 335, 38, { fontSize: 15, color: C.white, lineSpacing: 1.05 });
  rect(slide, 64, 630, 1145, 36, "#E7EDF1", 8);
  textBox(slide, "Ограничение выборки: 77,7% — 14–24 года; 73,5% — студенты. Это не причинный эффект показа Sanash.", 80, 641, 1110, 18, { fontSize: 14, bold: true, color: C.slate });
  footer(slide, 2);
  notes(slide, [
    "Данные: research/survey/outputs/descriptives.txt и model_summary.txt.",
    "Сценарии: одинаковые 5 минут ожидания и час пик; n=209 для переполненного, n=208 для состояния со стоячими местами.",
    "68,9% против 42,8%; WTW 7,96 мин, bootstrap 95% CI [5,89; 11,09], 1000 повторов.",
    "Ограничения: stated preference, 77,7% в возрасте 14–24, 73,5% студенты; не причинная оценка продукта.",
    "Фото: Tengrinews. https://tengrinews.kz/article/vlezt-i-vyzit-cto-ne-tak-s-almatinskimi-avtobusami-3255/",
  ]);
}

// 3 — Product output
{
  const slide = deck.slides.add();
  slide.background.fill = C.ink;
  header(slide, "Пассажиру нужен один понятный сигнал", "не видеопоток · не карта людей · не персональные данные", true);
  textBox(slide, "Sanash переводит оценку загрузки салона в пять устойчивых уровней — чтобы решить: садиться сейчас или ждать следующий автобус.", 64, 142, 1090, 70, { fontSize: 25, color: C.white, lineSpacing: 1.12 });
  const levels = [
    ["1", "Свободно", C.green], ["2", "Низкая загрузка", "#80D96B"], ["3", "Средняя загрузка", "#F4D35E"],
    ["4", "Высокая загрузка", "#F4A261"], ["5", "Переполнено", C.red],
  ];
  levels.forEach((l, i) => {
    const x = 64 + i * 226;
    rect(slide, x, 270, 198, 112, l[2], 18);
    textBox(slide, l[0], x + 16, 285, 40, 50, { fontSize: 35, bold: true, color: C.ink, align: "center" });
    textBox(slide, l[1], x + 12, 338, 174, 24, { fontSize: 15, bold: true, color: C.ink, align: "center" });
  });
  rule(slide, 64, 446, 1210, 446, C.darkGrid, 2);
  const fields = [
    ["Уровень", "1…5"], ["Оценка", "0…1"], ["Уверенность", "0…1"],
    ["Время", "метка UTC"], ["Состояние", "норма / нет данных"],
  ];
  fields.forEach((f, i) => {
    const x = 64 + i * 225;
    textBox(slide, f[0], x, 482, 180, 22, { fontSize: 14, bold: true, color: C.cyan });
    textBox(slide, f[1], x, 515, 185, 30, { fontSize: 21, color: C.white });
  });
  rect(slide, 64, 588, 1145, 50, "#102E47", 12);
  textBox(slide, "Граница приватности: наружу уходит только агрегированный статус; сырой кадр остаётся на бортовом устройстве.", 84, 603, 1105, 22, { fontSize: 17, bold: true, color: C.pale });
  footer(slide, 3, true);
  notes(slide, [
    "Источник: корневой README.md и development/README.md — пять порядковых уровней и непрерывная оценка 0..1.",
    "Черновой uplink передаёт агрегированный статус и метаданные; продукт не требует передачи сырого видео.",
    "Состояния отказа должны быть отдельными от уровней загрузки; два теста сейчас фиксируют открытые противоречия спецификации состояний.",
  ]);
}

// 4 — System and statuses
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  header(slide, "Что именно проверяет технический пилот", "каждый узел имеет явный статус");
  const stages = [
    ["01", "Кадр", "камера в салоне", "КАНДИДАТ", C.amber],
    ["02", "Оценка", "счёт + уверенность", "ПРОТОТИП", C.amber],
    ["03", "Стабилизация", "уровень + состояние", "ЧЕРНОВИК", C.cyan],
    ["04", "Сообщение", "минимальный JSON", "ЛОКАЛЬНО", C.green],
    ["05", "Avtobys", "приём и показ", "ОТКРЫТО", C.red],
  ];
  stages.forEach((s, i) => {
    const x = 54 + i * 244;
    rect(slide, x, 170, 214, 252, C.white, 20, { fill: C.grid, width: 1 });
    textBox(slide, s[0], x + 20, 190, 44, 24, { fontSize: 14, bold: true, color: C.blue });
    textBox(slide, s[1], x + 20, 238, 174, 34, { fontSize: 24, bold: true, color: C.ink });
    textBox(slide, s[2], x + 20, 285, 170, 48, { fontSize: 16, color: C.slate, lineSpacing: 1.06 });
    chip(slide, s[3], x + 20, 358, s[4], C.ink, 174);
    if (i < stages.length - 1) {
      rule(slide, x + 214, 292, x + 244, 292, C.blue, 3);
      rect(slide, x + 234, 287, 10, 10, C.blue, 5);
    }
  });
  const checks = [
    ["Качество кадра", "темнота · смаз · закрытый объектив"],
    ["Свежесть", "время последнего валидного сообщения"],
    ["Отказ", "явно «нет данных», а не ложный зелёный"],
  ];
  checks.forEach((c, i) => {
    const x = 64 + i * 385;
    textBox(slide, c[0], x, 494, 330, 25, { fontSize: 18, bold: true, color: C.blue });
    textBox(slide, c[1], x, 531, 330, 40, { fontSize: 16, color: C.slate });
  });
  rect(slide, 64, 608, 1145, 46, "#E2ECF2", 12);
  textBox(slide, "Критерий перехода: не подключать показ пассажирам, пока камера, качество счёта и состояния отказа не пройдут полевой тест.", 84, 621, 1105, 22, { fontSize: 16, bold: true, color: C.ink });
  footer(slide, 4);
  notes(slide, [
    "Источник: development/README.md, разделы pipeline, uplink schema и state machine.",
    "Локальный baseline и валидация сообщения существуют; транспорт, авторизация, привязка устройства к автобусу и TTL требуют согласования с Innoforce/Avtobys.",
    "Статусы на слайде намеренно разделяют реализацию, кандидатов и открытые интерфейсы.",
  ]);
}

// 5 — Camera test
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  header(slide, "Главный риск обзора решается в автобусе", "расчёт указывает на риск · выбор угла — только по измерению");
  textBox(slide, "Строго вниз камера видит лишь локальную зону прохода. Вместо ложной точности — короткий сравнительный тест четырёх углов.", 64, 139, 1110, 56, { fontSize: 23, color: C.ink, lineSpacing: 1.08 });
  const angles = ["0°", "30°", "45°", "60°"];
  angles.forEach((a, i) => {
    const x = 64 + i * 281;
    rect(slide, x, 246, 246, 155, i === 0 ? "#E8EDF0" : "#E6F8F5", 18, { fill: i === 0 ? C.grid : C.cyan, width: 1.5 });
    textBox(slide, a, x + 18, 267, 210, 54, { fontSize: 42, bold: true, color: i === 0 ? C.slate : C.blue, align: "center" });
    textBox(slide, i === 0 ? "базовая точка" : "кандидат", x + 20, 338, 206, 24, { fontSize: 15, bold: true, color: C.slate, align: "center" });
  });
  const metrics = [
    ["Видимая зона", "доля прохода без слепых участков"],
    ["Перекрытия", "двери, поручни, стоящие пассажиры"],
    ["Ошибка счёта", "сравнение с ручной разметкой"],
    ["Приватность", "минимизация распознаваемых лиц"],
  ];
  metrics.forEach((m, i) => {
    const x = 64 + i * 281;
    textBox(slide, m[0], x, 458, 245, 24, { fontSize: 18, bold: true, color: C.blue });
    textBox(slide, m[1], x, 494, 245, 48, { fontSize: 16, color: C.slate, lineSpacing: 1.05 });
  });
  rect(slide, 64, 582, 1145, 66, C.ink, 14);
  textBox(slide, "До теста с партнёром фиксируем пороги приёмки. После теста выбираем угол или останавливаем концепт.", 88, 602, 1095, 26, { fontSize: 19, bold: true, color: C.white, align: "center" });
  footer(slide, 5);
  notes(slide, [
    "Источник: development/README.md, CAMERA_GEOMETRY consolidated section.",
    "Бумажный расчёт для строго нижнего направления показывает локальный охват около 2,46 м при принятой длине салона 11,5 м; это не полевое измерение и поэтому не вынесено как точная метрика.",
    "Углы 0/30/45/60 — план сравнительного теста, не результат. Пороги ошибки и допустимой видимой зоны должны быть согласованы до теста.",
  ]);
}

// 6 — CAD mount
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  const cad = await asset("cad_camera_rig_concept.png");
  image(slide, cad, "image/png", "Концептуальная визуализация крепления камеры", { left: 470, top: 0, width: 810, height: 720 }, "cover", 0, { left: 0.02, top: 0, right: 0, bottom: 0 });
  rect(slide, 0, 0, 522, 720, C.paper);
  textBox(slide, "Крепление — объект проверки, не готовое изделие", 64, 48, 390, 104, { fontSize: 36, bold: true, color: C.ink, lineSpacing: 0.96 });
  textBox(slide, "Параметрическая оснастка нужна, чтобы быстро менять угол и проверить монтаж на реальном поручне.", 64, 171, 380, 78, { fontSize: 20, color: C.slate, lineSpacing: 1.08 });
  const tests = [
    ["01", "Совместимость", "диаметр поручня · доступ к крепежу"],
    ["02", "Безопасность", "вибрация · фиксация · страховочный трос"],
    ["03", "Кабель", "радиус изгиба · разгрузка натяжения"],
  ];
  tests.forEach((t, i) => {
    const y = 305 + i * 92;
    textBox(slide, t[0], 64, y, 42, 30, { fontSize: 16, bold: true, color: C.cyan });
    textBox(slide, t[1], 124, y, 280, 28, { fontSize: 20, bold: true, color: C.ink });
    textBox(slide, t[2], 124, y + 36, 298, 35, { fontSize: 15, color: C.slate });
  });
  rect(slide, 64, 612, 354, 44, C.red, 10);
  textBox(slide, "КОНЦЕПТ · НЕ ИЗГОТОВЛЕНО", 78, 626, 326, 18, { fontSize: 13, bold: true, color: C.white, align: "center" });
  rect(slide, 826, 615, 398, 52, { color: C.ink, transparency: 0.03 }, 12);
  textBox(slide, "ВИЗУАЛИЗАЦИЯ СОЗДАНА ИИ — НЕ ФОТО", 844, 632, 362, 18, { fontSize: 12, bold: true, color: C.white, align: "center" });
  footer(slide, 6, false, "SANASH · механическая концепция");
  notes(slide, [
    "Источник требований: development/cad/camera_test_rig.scad и development/README.md.",
    "Визуализация создана ИИ для коммуникации концепта; это не фотография, не точный рендер SCAD и не доказательство изготовления.",
    "Текущий статус: proposed; нет STL/STEP, фото сборки и физической проверки на автобусе.",
  ]);
}

// 7 — Hardware chain
{
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  header(slide, "Аппаратная цепь: пять узлов, три блокера", "схема для стендовой проверки, не спецификация поставки");
  const nodes = [
    ["Камера", "IMX219-160", "КАНДИДАТ", C.amber],
    ["Вычислитель", "Jetson Orin Nano", "ВЫБРАН", C.cyan],
    ["Накопитель", "NVMe", "ОТКРЫТО", C.red],
    ["Связь", "модем / сеть", "ОТКРЫТО", C.red],
    ["Avtobys", "приём статуса", "ОТКРЫТО", C.red],
  ];
  nodes.forEach((n, i) => {
    const x = 50 + i * 244;
    rect(slide, x, 180, 210, 158, C.paper, 18, { fill: C.grid, width: 1 });
    textBox(slide, n[0], x + 18, 203, 174, 26, { fontSize: 20, bold: true, color: C.ink, align: "center" });
    textBox(slide, n[1], x + 18, 245, 174, 27, { fontSize: 16, color: C.slate, align: "center" });
    chip(slide, n[2], x + 25, 290, n[3], C.ink, 160);
    if (i < nodes.length - 1) {
      rule(slide, x + 210, 258, x + 244, 258, C.blue, 3);
      rect(slide, x + 234, 253, 10, 10, C.blue, 5);
    }
  });
  textBox(slide, "ПИТАНИЕ", 64, 402, 150, 24, { fontSize: 15, bold: true, color: C.blue });
  rect(slide, 64, 440, 1145, 72, "#E8F4F2", 14);
  textBox(slide, "Аккумулятор 65 Вт+ → триггер USB PD 12/15 В → разъём питания Jetson", 88, 457, 790, 30, { fontSize: 20, bold: true, color: C.ink });
  chip(slide, "НЕ ПРОВЕРЕНО", 970, 461, C.amber, C.ink, 200);
  const blockers = [
    ["1", "Питание Jetson", "режим, разъём и время работы"],
    ["2", "Хранение", "объём, шифрование, очистка"],
    ["3", "Интеграция", "транспорт, авторизация, привязка к рейсу"],
  ];
  blockers.forEach((b, i) => {
    const x = 64 + i * 385;
    textBox(slide, b[0], x, 560, 34, 38, { fontSize: 26, bold: true, color: C.red, align: "center" });
    textBox(slide, b[1], x + 52, 556, 286, 25, { fontSize: 18, bold: true, color: C.ink });
    textBox(slide, b[2], x + 52, 590, 290, 38, { fontSize: 15, color: C.slate });
  });
  footer(slide, 7);
  notes(slide, [
    "Источник: development/cad/wiring_diagram.svg и development/README.md.",
    "USB-C на Jetson dev kit используется для данных/recovery; питание предполагается через barrel jack с PD trigger. Схема proposed, узлы не проверены на стенде.",
    "Камера IMX219-160 и NVMe указаны как кандидаты/открытые решения. Транспорт, авторизация и device-to-vehicle mapping требуют согласования с Innoforce/Avtobys.",
  ]);
}

// 8 — What exists
{
  const slide = deck.slides.add();
  slide.background.fill = C.ink;
  header(slide, "Работает каркас, а не продуктовая модель", "инженерная база отделена от продуктовых доказательств", true);
  textBox(slide, "456", 70, 164, 210, 80, { fontSize: 70, bold: true, color: C.cyan });
  textBox(slide, "тестов проходят", 76, 249, 250, 28, { fontSize: 19, color: C.white });
  textBox(slide, "2", 365, 164, 120, 80, { fontSize: 70, bold: true, color: C.red });
  textBox(slide, "фиксируют известные\nпротиворечия спецификации", 370, 249, 320, 58, { fontSize: 18, color: C.pale, lineSpacing: 1.05 });
  textBox(slide, "прогон: 11.09.2026 · a4099566", 76, 304, 380, 20, { fontSize: 12, bold: true, color: C.mid });
  rule(slide, 64, 346, 1210, 346, C.darkGrid, 2);
  const yes = [
    ["✓", "Загрузка и подготовка DISCO", "воспроизводимый локальный путь"],
    ["✓", "Скалярный счёт из карты плотности", "техническая проверка запуска"],
    ["✓", "Черновик схемы сообщения", "валидация структуры локально"],
  ];
  const no = [
    ["×", "Нет кадров из салона автобуса", "доменные данные отсутствуют"],
    ["×", "Нет целевой обученной модели", "ResNet-18 случайно инициализирован"],
    ["×", "Нет замеров на Jetson", "точность и задержка не заявляются"],
  ];
  textBox(slide, "ЕСТЬ", 64, 383, 400, 24, { fontSize: 14, bold: true, color: C.green });
  textBox(slide, "НЕТ ПОКА", 662, 383, 400, 24, { fontSize: 14, bold: true, color: C.red });
  yes.forEach((r, i) => {
    const y = 430 + i * 70;
    textBox(slide, r[0], 64, y, 30, 26, { fontSize: 22, bold: true, color: C.green });
    textBox(slide, r[1], 108, y, 460, 24, { fontSize: 18, bold: true, color: C.white });
    textBox(slide, r[2], 108, y + 30, 470, 24, { fontSize: 14, color: C.mid });
  });
  no.forEach((r, i) => {
    const y = 430 + i * 70;
    textBox(slide, r[0], 662, y, 30, 26, { fontSize: 22, bold: true, color: C.red });
    textBox(slide, r[1], 706, y, 455, 24, { fontSize: 18, bold: true, color: C.white });
    textBox(slide, r[2], 706, y + 30, 470, 24, { fontSize: 14, color: C.mid });
  });
  footer(slide, 8, true);
  notes(slide, [
    "Источник: development/README.md и корневой README.md.",
    "Текущий прогон: 456 passed, 2 failed; оба падения фиксируют открытые противоречия в спецификации состояний.",
    "Baseline: публичный наружный DISCO, случайно инициализированный ResNet-18, smoke run; это не качество Sanash на автобусах.",
    "Нет собственных кадров салона, целевой модели, метрик точности и бенчмарка на Jetson.",
  ]);
}

// 9 — Latency plan
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  header(slide, "Цель задержки ≤30 с. Замеров на Jetson пока нет", "значения ниже — проектные лимиты");
  const chart = slide.charts.add("bar", {
    position: { left: 56, top: 168, width: 690, height: 350 },
    categories: ["Захват", "Инференс", "Стабилизация", "Публикация / сеть"],
    series: [{
      name: "Проектный лимит, секунды", values: [0.1, 1.0, 10.0, 2.0], valuesFormatCode: "0.0", fill: C.cyan,
      points: [{ idx: 0, fill: C.green }, { idx: 1, fill: C.cyan }, { idx: 2, fill: C.amber }, { idx: 3, fill: C.blue }],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 42 },
    hasLegend: false,
    xAxis: { visible: true, min: 0, max: 12, majorUnit: 2, numberFormatCode: "0.0\" с\"", majorGridlines: { style: "solid", fill: C.grid, width: 1 }, line: { fill: C.grid, width: 1 } },
    yAxis: { visible: true, line: { fill: "none", width: 0 }, textStyle: { typeface: fontFamily, fontSize: 16, fill: C.ink } },
    dataLabels: { showValue: false },
  });
  applyPresentationChartFont(chart, { fontFamily });
  textBox(slide, "≤2,0", 279, 210, 70, 24, { fontSize: 16, bold: true, color: C.ink });
  textBox(slide, "≤10,0", 634, 284, 78, 24, { fontSize: 16, bold: true, color: C.ink });
  textBox(slide, "≤1,0", 234, 358, 70, 24, { fontSize: 16, bold: true, color: C.ink });
  textBox(slide, "≤0,1", 194, 432, 70, 24, { fontSize: 16, bold: true, color: C.ink });
  chip(slide, "ПРОЕКТНЫЙ БЮДЖЕТ · НЕ ЗАМЕР", 820, 157, C.red, C.white, 300);
  textBox(slide, "≤13,1 с +", 840, 218, 300, 60, { fontSize: 45, bold: true, color: C.blue });
  textBox(slide, "задержка Avtobys", 846, 285, 300, 34, { fontSize: 23, bold: true, color: C.red });
  textBox(slide, "приём и отображение\nна стороне приложения", 848, 343, 300, 60, { fontSize: 18, color: C.ink, lineSpacing: 1.05 });
  rect(slide, 812, 500, 360, 104, C.ink, 16);
  textBox(slide, "Как измеряем", 836, 520, 160, 22, { fontSize: 15, bold: true, color: C.cyan });
  textBox(slide, "сквозная метка времени: кадр → статус → экран приложения", 836, 552, 310, 38, { fontSize: 17, color: C.white, lineSpacing: 1.05 });
  rect(slide, 64, 621, 1105, 38, "#E7EDF1", 9);
  textBox(slide, "Решение о переходе: только по измеренному 95-му перцентилю сквозной задержки и согласованному порогу качества.", 82, 632, 1070, 18, { fontSize: 15, bold: true, color: C.slate });
  footer(slide, 9);
  notes(slide, [
    "Источник: development/README.md, раздел latency budget.",
    "Цель продукта: <=30 секунд до появления в приложении.",
    "0,1 / 1,0 / до 10 / 2 секунды — кандидаты для проектного бюджета, а не измерения на Jetson. Их сумма 13,1 с не включает неизвестную задержку на стороне Avtobys.",
    "Приёмочный показатель предлагается измерять сквозной меткой времени и 95-м перцентилем; конкретный порог качества модели согласуется до теста.",
  ]);
}

// 10 — Pilot decision
{
  const slide = deck.slides.add();
  slide.background.fill = C.ink;
  const terminal = await asset("avtobys_terminal_bus.jpg");
  image(slide, terminal, "image/jpeg", "Терминал Avtobys в автобусе", { left: 895, top: 0, width: 385, height: 720 }, "cover", 0, { left: 0.08, top: 0, right: 0.06, bottom: 0 });
  rect(slide, 0, 0, 930, 720, C.ink);
  textBox(slide, "Решение на встрече", 64, 47, 670, 60, { fontSize: 38, bold: true, color: C.white });
  rect(slide, 64, 111, 72, 5, C.cyan, 3);
  textBox(slide, "Этап 1 · один автобус", 64, 158, 360, 30, { fontSize: 21, bold: true, color: C.cyan });
  textBox(slide, "Доступ для обмера и съёмки тестовых кадров", 64, 201, 730, 36, { fontSize: 27, bold: true, color: C.white });
  textBox(slide, "Результат: выбранный угол и крепление, стенд питания, размеченная выборка, измеренная задержка, решение о переходе.", 64, 252, 742, 70, { fontSize: 19, color: C.pale, lineSpacing: 1.08 });
  rule(slide, 64, 352, 820, 352, C.darkGrid, 2);
  textBox(slide, "Этап 2 · только после решения о переходе", 64, 385, 560, 30, { fontSize: 21, bold: true, color: C.amber });
  textBox(slide, "Несколько последовательных автобусов", 64, 428, 710, 36, { fontSize: 27, bold: true, color: C.white });
  textBox(slide, "Теневой режим → контроль / показ → измерение фактического выбора пассажира. Один автобус для такого эксперимента недостаточен.", 64, 480, 742, 76, { fontSize: 19, color: C.pale, lineSpacing: 1.08 });
  rect(slide, 64, 590, 758, 72, C.cyan, 16);
  textBox(slide, "Согласуем этап 1 и ответственных с обеих сторон?", 88, 611, 710, 30, { fontSize: 23, bold: true, color: C.ink, align: "center" });
  footer(slide, 10, true, "SANASH · запрос на технический пилот");
  notes(slide, [
    "Источник пилотной рамки: business/README.md, INNOFORCE_RTCI_PILOT_BRIEF.",
    "Этап 1: один автобус достаточен для обмера, тестовых кадров, стенда и теневой технической проверки.",
    "Этап 2: пользовательский эксперимент требует нескольких последовательных оборудованных автобусов или альтернативного дизайна; один автобус не поддерживает обещанный выбор между текущим и следующим рейсом.",
    "Фото терминала Avtobys: официальный портал gov.kz. https://www.gov.kz/memleket/entities/almaty/press/news/details/1157055?lang=kk",
  ]);
}

for (let i = 0; i < deck.slides.items.length; i++) {
  const slide = deck.slides.items[i];
  console.log(`rendering slide ${i + 1}`);
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(buildDir, `reviewed-slide-${i + 1}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(buildDir, `reviewed-slide-${i + 1}.layout.json`), await layout.text());
}

const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, `candidate-${path.parse(finalName).name}.pptx`);
await (await PresentationFile.exportPptx(deck)).save(candidatePath);

const result = await finalizePresentation({
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [2, 9],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: true,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  fontPolicy: { basis: "design", families: [fontFamily] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${finalName}.validation.json`),
});

console.log(JSON.stringify({ finalPath, result }, null, 2));
