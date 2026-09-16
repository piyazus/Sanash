import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:\\Users\\User\\OneDrive\\Desktop\\me\\Sanash";
const SKILL_DIR = "C:\\Users\\User\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.904.11930\\skills\\presentations";
const TMP_DIR = path.join(workspaceDir, ".deck_build");
const assetsDir = path.join(TMP_DIR, "assets");
const outputDir = path.join(workspaceDir, "deliverables");
const finalName = process.env.FINAL_NAME || "Sanash_technical_pitch_ru_v1.pptx";
const FINAL_PPTX = path.join(outputDir, finalName);
const RUNTIME_PYTHON = "C:\\Users\\User\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";

const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
const fontFamily = resolvePresentationFont({ fontFamily: "Aptos" });

const C = {
  ink: "#07131F",
  navy: "#0B2136",
  blue: "#1759D1",
  cyan: "#16D9C4",
  green: "#2FCC84",
  lime: "#A8E063",
  amber: "#F4B942",
  red: "#F0645A",
  white: "#F7FAFC",
  paper: "#F3F6F8",
  slate: "#526273",
  pale: "#DCE6ED",
  grid: "#CBD6DE",
};

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

const asset = async (name) => new Uint8Array(await fs.readFile(path.join(assetsDir, name)));
const localAsset = async (rel) => new Uint8Array(await fs.readFile(path.join(workspaceDir, rel)));

function addRect(slide, left, top, width, height, fill, radius = 0, line = "none") {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: line === "none" ? { fill: "none", width: 0 } : line,
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function addText(slide, text, left, top, width, height, opts = {}) {
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
    autoFit: opts.autoFit || "none",
    wrap: "square",
    lineSpacing: opts.lineSpacing ?? 1.0,
    insets: opts.insets || { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return box;
}

function addTitle(slide, title, dark = false, subtitle = null) {
  addText(slide, title, 64, 38, subtitle ? 820 : 1090, 58, { fontSize: 36, bold: true, color: dark ? C.white : C.ink });
  addRect(slide, 64, 104, 72, 5, C.cyan, 3);
  if (subtitle) addText(slide, subtitle, 920, 48, 290, 40, { fontSize: 14, color: dark ? C.pale : C.slate, align: "right" });
}

function addFooter(slide, n, dark = false, text = "SANASH · технический питч") {
  addText(slide, text, 64, 683, 700, 20, { fontSize: 12, bold: true, color: dark ? "#8FA7BA" : "#7B8B99" });
  addText(slide, String(n).padStart(2, "0"), 1160, 681, 48, 20, { fontSize: 12, bold: true, color: dark ? C.cyan : C.blue, align: "right" });
}

function addImage(slide, bytes, contentType, alt, position, fit = "cover", radius = 0, crop) {
  return slide.images.add({
    blob: bytes,
    contentType,
    alt,
    fit,
    position,
    ...(radius ? { geometry: "roundRect", borderRadius: radius } : {}),
    ...(crop ? { crop } : {}),
  });
}

function styleChart(chart, dark = false) {
  applyPresentationChartFont(chart, { fontFamily });
}

// 1. Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  const buses = await asset("almaty_green_buses.webp");
  addImage(slide, buses, "image/webp", "Автобусы Алматы", { left: 624, top: 0, width: 656, height: 720 }, "cover");
  addRect(slide, 0, 0, 650, 720, C.ink);
  addRect(slide, 624, 0, 12, 720, C.cyan);
  addText(slide, "SANASH", 72, 62, 360, 36, { fontSize: 18, bold: true, color: C.cyan });
  addText(slide, "Загрузка автобуса\nв реальном времени", 72, 178, 520, 190, { fontSize: 54, bold: true, color: C.white, lineSpacing: 0.92 });
  addText(slide, "Потолочная RGB-камера, edge-обработка и пять уровней загрузки для Avtobys", 74, 404, 480, 94, { fontSize: 22, color: C.pale, lineSpacing: 1.12 });
  addText(slide, "Технический питч · Алматы · 2026", 74, 620, 450, 30, { fontSize: 15, bold: true, color: "#8FA7BA" });
  slide.speakerNotes.textFrame.setText([
    "Фото: автобусы Алматы. Источник: Orda.kz, фото gov.kz.",
    "https://kaz.orda.kz/almatyda-avtobys-jelektrondy-zholaky-toleu-zhyjesi-iske-kosylady-137363/",
    "Основные продуктовые утверждения: локальный README.md проекта Sanas/Sanash.",
  ]);
}

// 2. Evidence of passenger demand
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "Переполненность меняет решение пассажира", false, "опрос заявленных предпочтений · n = 215");

  const chart = slide.charts.add("bar", {
    position: { left: 58, top: 160, width: 690, height: 360 },
    categories: ["Переполнен", "Есть места стоя"],
    series: [{
      name: "Готовы ждать следующий автобус",
      values: [0.689, 0.428],
      valuesFormatCode: "0%",
      points: [{ idx: 0, fill: C.red }, { idx: 1, fill: C.cyan }],
      fill: C.blue,
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 54 },
    hasLegend: false,
    xAxis: { visible: true, min: 0, max: 0.8, majorUnit: 0.2, numberFormatCode: "0%", majorGridlines: { style: "solid", fill: C.grid, width: 1 }, line: { fill: C.grid, width: 1 } },
    yAxis: { visible: true, line: { fill: "none", width: 0 }, textStyle: { typeface: fontFamily, fontSize: 16, fill: C.ink } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: fontFamily, fontSize: 18, fill: C.ink, bold: true } },
  });
  styleChart(chart, false);

  addText(slide, "+26,1 п.п.", 80, 532, 300, 62, { fontSize: 42, bold: true, color: C.blue });
  addText(slide, "разрыв при одинаковом ожидании 5 минут и одинаковом часе пик", 82, 590, 560, 62, { fontSize: 18, color: C.slate, lineSpacing: 1.08 });

  const interior = await asset("almaty_bus_interior.webp");
  addImage(slide, interior, "image/webp", "Салон автобуса Алматы", { left: 790, top: 150, width: 420, height: 470 }, "cover", 24, { left: 0.05, top: 0, right: 0, bottom: 0 });
  addRect(slide, 820, 500, 360, 88, C.ink, 14);
  addText(slide, "7,96 мин", 842, 514, 160, 34, { fontSize: 28, bold: true, color: C.cyan });
  addText(slide, "оценочная готовность ждать, чтобы избежать переполненности", 842, 551, 310, 38, { fontSize: 14, color: C.white, lineSpacing: 1.05 });
  addFooter(slide, 2, false);
  slide.speakerNotes.textFrame.setText([
    "Источник данных: research/survey/outputs/descriptives.txt и model_summary.txt.",
    "Сценарии 2 и 4: ожидание 5 минут, час пик; отличаются только состоянием переполненности.",
    "68,9% против 42,8%, разрыв 26,1 п.п. WTW = 7,96 мин, 95% bootstrap CI [5,89; 11,09].",
    "Ограничение: stated preference, а не причинный эффект реального показа RTCI.",
    "Фото салона: Kazinform. https://www.inform.kz/ru/eshe-dva-novih-prigorodnih-marshruta-zapuskayut-v-almati-d13650",
  ]);
}

// 3. Product architecture
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  addTitle(slide, "Архитектура продукта", true, "обработка на борту · сырое видео не передаётся");
  const stages = [
    ["01", "Камера", "RGB-кадр + метка времени"],
    ["02", "Контроль качества", "темнота · смаз · закрытый объектив"],
    ["03", "Инференс на борту", "число людей + уверенность"],
    ["04", "Сглаживание", "скор 0…1 + уровень"],
    ["05", "Avtobys", "показ пассажиру"],
  ];
  const xs = [64, 306, 548, 790, 1032];
  for (let i = 0; i < stages.length; i++) {
    const [num, name, desc] = stages[i];
    addText(slide, num, xs[i], 170, 70, 28, { fontSize: 16, bold: true, color: C.cyan });
    addText(slide, name, xs[i], 212, 188, 54, { fontSize: 25, bold: true, color: C.white, lineSpacing: 0.95 });
    addText(slide, desc, xs[i], 282, 180, 74, { fontSize: 16, color: "#AFC1CF", lineSpacing: 1.06 });
    if (i < stages.length - 1) {
      addRect(slide, xs[i] + 170, 194, 54, 3, "#355269", 2);
      addRect(slide, xs[i] + 215, 190, 10, 10, C.cyan, 5);
    }
  }

  addText(slide, "Пять уровней для понятного интерфейса", 64, 410, 460, 36, { fontSize: 22, bold: true, color: C.white });
  const levels = [
    ["1", "Свободно", "#2FCC84"],
    ["2", "Низкая", "#80D96B"],
    ["3", "Средняя", "#F4D35E"],
    ["4", "Высокая", "#F4A261"],
    ["5", "Переполнен", "#F0645A"],
  ];
  levels.forEach((l, i) => {
    const x = 64 + i * 225;
    addRect(slide, x, 474, 196, 54, l[2], 12);
    addText(slide, `${l[0]}  ${l[1]}`, x + 14, 490, 168, 24, { fontSize: 16, bold: true, color: C.ink, vAlign: "middle" });
  });
  addText(slide, "Сбой не маскируется цветом: приложение получает отдельное состояние «нет данных» или «ограничено»", 64, 576, 1090, 42, { fontSize: 19, color: C.pale });
  addFooter(slide, 3, true);
  slide.speakerNotes.textFrame.setText([
    "Источник: development/README.md, разделы Pipeline и Состояния; корневой README.md.",
    "Продуктовый выход: пять порядковых уровней и непрерывный score 0..1.",
    "Удалённая передача сырого видео не заявлена как часть продукта.",
  ]);
}

// 4. Camera geometry
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Геометрия камеры определяет объект измерения", false, "расчёт на бумаге · требуется полевой кадр");
  const chart = slide.charts.add("bar", {
    position: { left: 60, top: 160, width: 660, height: 350 },
    categories: ["Поле зрения на уровне голов", "Длина салона"],
    series: [{
      name: "Метры",
      values: [2.46, 11.5],
      valuesFormatCode: "0.0",
      fill: C.blue,
      points: [{ idx: 0, fill: C.cyan }, { idx: 1, fill: C.blue }],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 48 },
    hasLegend: false,
    xAxis: { visible: true, min: 0, max: 12, majorUnit: 2, numberFormatCode: "0.0\" м\"", majorGridlines: { style: "solid", fill: C.grid, width: 1 }, line: { fill: C.grid, width: 1 } },
    yAxis: { visible: true, line: { fill: "none", width: 0 }, textStyle: { typeface: fontFamily, fontSize: 15, fill: C.ink } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: fontFamily, fontSize: 18, fill: C.ink, bold: true } },
  });
  styleChart(chart, false);
  addText(slide, "≈21%", 798, 168, 300, 78, { fontSize: 60, bold: true, color: C.blue });
  addText(slide, "длины салона видит одна камера, направленная строго вниз, на уровне голов стоящих пассажиров", 800, 252, 350, 120, { fontSize: 22, color: C.ink, lineSpacing: 1.08 });
  addRect(slide, 782, 410, 4, 170, C.cyan, 2);
  addText(slide, "Рабочая гипотеза", 810, 410, 320, 30, { fontSize: 18, bold: true, color: C.blue });
  addText(slide, "Наклон 30–45° вдоль прохода выглядит реалистичнее. Альтернативы: несколько камер или счёт только в видимой зоне.", 810, 452, 350, 120, { fontSize: 20, color: C.slate, lineSpacing: 1.12 });
  addFooter(slide, 4, false);
  slide.speakerNotes.textFrame.setText([
    "Источник: development/README.md, CAMERA_GEOMETRY.md consolidated section.",
    "Расчёт: около 2,46 м длины обзора на уровне голов стоящих пассажиров при принятой длине салона 11,5 м.",
    "Это расчёт, а не измерение; угол установки и фактическое поле зрения должны пройти тест в автобусе.",
  ]);
}

// 5. CAD concept
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  const cad = await asset("cad_camera_rig_concept.png");
  addImage(slide, cad, "image/png", "Концептуальная CAD-визуализация регулируемого крепления камеры", { left: 485, top: 0, width: 795, height: 720 }, "cover", 0, { left: 0.02, top: 0, right: 0, bottom: 0 });
  addRect(slide, 0, 0, 520, 720, C.paper);
  addText(slide, "CAD-концепт крепления", 64, 54, 390, 80, { fontSize: 38, bold: true, color: C.ink });
  addText(slide, "Параметрическая оснастка для проверки угла, обзора и кабеля", 64, 150, 380, 74, { fontSize: 21, color: C.slate, lineSpacing: 1.08 });
  const items = [
    ["Регулируемый наклон", "фиксированные положения 0°, 15°, 30° и 45°"],
    ["Монтаж на поручень", "V-блок и стяжка без привязки к одному диаметру"],
    ["Безопасность", "отдельная точка страховочного троса"],
    ["Кабель", "радиус изгиба и разгрузка натяжения CSI"],
  ];
  items.forEach((it, i) => {
    const y = 278 + i * 78;
    addText(slide, it[0], 64, y, 360, 27, { fontSize: 18, bold: true, color: C.blue });
    addText(slide, it[1], 64, y + 30, 370, 43, { fontSize: 16, color: C.ink, lineSpacing: 1.02 });
  });
  addText(slide, "СТАТУС: КОНЦЕПТ", 64, 622, 210, 24, { fontSize: 13, bold: true, color: C.red });
  addText(slide, "не печаталось и не проверялось физически", 64, 650, 350, 26, { fontSize: 14, color: C.slate });
  addFooter(slide, 5, false, "SANASH · испытательная CAD-оснастка");
  slide.speakerNotes.textFrame.setText([
    "Источник механических требований: development/cad/camera_test_rig.scad и development/README.md.",
    "Изображение создано ИИ как визуализация концепта для презентации; это не фотография реального устройства и не точный рендер SCAD.",
    "Текущий статус проекта: proposed, физически не проверено; STL/STEP и реальные фотографии первой сборки отсутствуют.",
  ]);
}

// 6. Hardware and wiring
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Аппаратная цепь прототипа", false, "решённые узлы отделены от кандидатов и открытых пунктов");
  const wiring = await localAsset("development/cad/wiring_diagram.svg");
  addImage(slide, wiring, "image/svg+xml", "Схема подключения прототипа Sanash", { left: 58, top: 120, width: 1164, height: 530 }, "contain");
  addFooter(slide, 6, false);
  slide.speakerNotes.textFrame.setText([
    "Источник: development/cad/wiring_diagram.svg, сохранён без перерисовки.",
    "Ключевой блокер схемы: USB-C на Jetson dev kit используется для данных/recovery, питание идёт через barrel jack; нужен PD trigger.",
    "Схема имеет статус proposed; узлы не куплены и не проверены на стенде.",
  ]);
}

// 7. Latency budget
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  addTitle(slide, "Бюджет задержки", true, "цель: не более 30 секунд до появления в приложении");
  const chart = slide.charts.add("bar", {
    position: { left: 60, top: 165, width: 760, height: 390 },
    categories: ["Захват", "Инференс", "Сглаживание", "Публикация и сеть"],
    series: [{
      name: "Секунды",
      values: [0.1, 1.0, 10.0, 2.0],
      valuesFormatCode: "0.0",
      fill: C.cyan,
      points: [{ idx: 0, fill: C.green }, { idx: 1, fill: C.cyan }, { idx: 2, fill: C.amber }, { idx: 3, fill: C.blue }],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 42 },
    hasLegend: false,
    xAxis: { visible: true, min: 0, max: 12, majorUnit: 2, numberFormatCode: "0.0\" с\"", majorGridlines: { style: "solid", fill: "#29455A", width: 1 }, line: { fill: "#29455A", width: 1 } },
    yAxis: { visible: true, line: { fill: "none", width: 0 }, textStyle: { typeface: fontFamily, fontSize: 16, fill: C.white } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: fontFamily, fontSize: 17, fill: C.white, bold: true } },
  });
  styleChart(chart, true);
  addText(slide, "13,1 с", 895, 185, 280, 80, { fontSize: 60, bold: true, color: C.cyan });
  addText(slide, "известная часть бюджета", 900, 270, 270, 32, { fontSize: 18, color: C.pale });
  addRect(slide, 900, 350, 260, 8, "#29455A", 4);
  addRect(slide, 900, 350, 114, 8, C.cyan, 4);
  addText(slide, "44%", 900, 372, 100, 34, { fontSize: 24, bold: true, color: C.white });
  addText(slide, "целевого лимита уже распределено", 900, 410, 250, 54, { fontSize: 17, color: "#AFC1CF" });
  addText(slide, "Неизвестно", 900, 500, 250, 28, { fontSize: 18, bold: true, color: C.amber });
  addText(slide, "время приёма и отображения на стороне Avtobys", 900, 534, 250, 62, { fontSize: 17, color: C.pale, lineSpacing: 1.05 });
  addFooter(slide, 7, true);
  slide.speakerNotes.textFrame.setText([
    "Источник: development/README.md, раздел Бюджет задержки.",
    "Кандидаты: захват 0,1 с; инференс 1,0 с; сглаживание до 10 с; публикация/сеть 2 с. Итого известное 13,1 с.",
    "Целевой лимит <=30 с. Время на стороне Innoforce/Avtobys пока неизвестно.",
    "Все значения являются кандидатами, а не результатами измерений на Jetson.",
  ]);
}

// 8. Pilot and ask
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  const terminal = await asset("avtobys_terminal_bus.jpg");
  addImage(slide, terminal, "image/jpeg", "Терминал Avtobys в автобусе", { left: 854, top: 0, width: 426, height: 720 }, "cover", 0, { left: 0.05, top: 0, right: 0.08, bottom: 0 });
  addRect(slide, 0, 0, 880, 720, C.paper);
  addTitle(slide, "Ограниченный пилот на одном маршруте", false);
  addText(slide, "Городской запуск не входит в первый этап", 64, 122, 620, 28, { fontSize: 15, color: C.slate });

  const steps = [
    ["1", "Обмер и доступ", "модель автобуса, фото точки установки, питание, разрешение на съёмку"],
    ["2", "Технический режим без показа", "сравнение CV с ручной разметкой до включения функции пассажирам"],
    ["3", "Рандомизированный A/B-тест", "часть пользователей видит загрузку первого и следующего автобуса"],
  ];
  steps.forEach((s, i) => {
    const y = 170 + i * 132;
    addText(slide, s[0], 64, y, 48, 48, { fontSize: 32, bold: true, color: C.cyan, align: "center", vAlign: "middle" });
    addText(slide, s[1], 132, y, 560, 32, { fontSize: 23, bold: true, color: C.ink });
    addText(slide, s[2], 132, y + 40, 610, 58, { fontSize: 17, color: C.slate, lineSpacing: 1.08 });
  });
  addRect(slide, 64, 568, 720, 2, C.grid);
  addText(slide, "Что нужно от партнёра сейчас", 64, 594, 380, 30, { fontSize: 21, bold: true, color: C.blue });
  addText(slide, "доступ к одному автобусу · API и ID рейса · флаг функции · ответственный за приватность", 64, 634, 720, 34, { fontSize: 17, color: C.ink });
  addFooter(slide, 8, false, "SANASH · запрос на технический пилот");
  slide.speakerNotes.textFrame.setText([
    "Источник пилотной рамки: business/README.md, раздел INNOFORCE_RTCI_PILOT_BRIEF.",
    "Предлагаемый scope: один высокочастотный маршрут, один тип автобуса, shadow technical period, затем randomized treatment после go/no-go.",
    "Фото терминала Avtobys: официальный портал gov.kz. https://www.gov.kz/memleket/entities/almaty/press/news/details/1157055?lang=kk",
    "Не обещать production readiness, точность модели или причинный эффект до полевого эксперимента.",
  ]);
}

// Private preview renders for visual inspection.
for (let i = 0; i < presentation.slides.items.length; i++) {
  const slide = presentation.slides.items[i];
  console.log(`rendering slide ${i + 1}`);
  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(TMP_DIR, `slide-${i + 1}.layout.json`), await layout.text());
}

const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, `candidate-${path.parse(finalName).name}.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [2, 4, 7],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: true,
};

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  fontPolicy: { basis: "design", families: [fontFamily] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${finalName}.validation.json`),
});

console.log(JSON.stringify({ finalPath: FINAL_PPTX, result }, null, 2));
