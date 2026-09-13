// Local regression checks. No Google services are called or responses submitted.
// Run from any directory: node research/survey/wave2/verify_local.cjs
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const root = path.resolve(__dirname, '../../..');
const read = p => fs.readFileSync(path.join(root, p), 'utf8');
const plain = value => JSON.parse(JSON.stringify(value));
function load(p, extras = {}) {
  const context = vm.createContext(extras);
  vm.runInContext(read(p), context, { filename: p });
  return context;
}

test('12 router destinations match the 12 source codes; random intervals cover each variant', () => {
  const math = Object.create(Math);
  const router = load('sanas_form_router/Code.gs', { Math: math });
  const master = load('research/survey/wave2/master_responses.gs');
  const urls = [];
  for (const lang of ['RU', 'KK', 'EN']) {
    for (let index = 0; index < 4; index++) {
      for (const sample of [index / 4, (index + 0.999999) / 4]) {
        math.random = () => sample;
        const form = router.getRandomForm(lang);
        assert.equal(form.label, `Block ${index < 2 ? 1 : 2} — Order ${index % 2 ? 'B' : 'A'}`);
        assert.match(form.url, /^https:\/\/docs.google.com\/forms\/d\/e\/[\w-]+\/viewform$/);
      }
      urls.push(router.getRandomForm(lang).url);
    }
  }
  assert.equal(new Set(urls).size, 12);
  assert.equal(new Set(master.RESPONSE_SOURCES.map(s => s.spreadsheetId)).size, 12);
  assert.deepEqual(plain(master.RESPONSE_SOURCES.map(s => s.code)),
    ['RU', 'KK', 'EN'].flatMap(l => [1, 2].flatMap(b => ['A', 'B'].map(o => `${l}-B${b}-${o}`))));
  assert.doesNotThrow(() => router.getRandomForm(' ru '));
  for (const bad of ['', null, '__proto__', 'FR']) assert.throws(() => router.getRandomForm(bad), /Unsupported language/);
});

test('four fielding orders match design.json; attention and repeat tasks stay in positions 5/10', () => {
  const builder = load('research/survey/wave2/build_forms.gs');
  const design = JSON.parse(read('research/survey/wave2/design.json'));
  for (const block of [1, 2]) for (const order of ['A', 'B']) {
    const tasks = plain(builder.situationsFor_(block, order));
    assert.equal(tasks.length, 10);
    assert.deepEqual(tasks[4], { a: 'seated', b: 'packed', wait: 8, peak: 0 });
    assert.deepEqual(tasks[9], tasks[2]);
    const expected = design[block].map(t => ({ a: t.bus_a, b: t.bus_b, wait: t.wait, peak: t.peak }));
    if (order === 'B') expected.reverse();
    assert.deepEqual(tasks.filter((_, i) => i !== 4 && i !== 9), expected);
  }
});

test('D3 is a required checkbox in all three generated languages', () => {
  const builder = load('research/survey/wave2/build_forms.gs');
  for (const lang of ['ru', 'kk', 'en']) {
    const items = [];
    function item(type) {
      const value = { type, setTitle(t) { this.title = t; return this; },
        setChoiceValues(v) { this.values = plain(v); return this; },
        setRequired(r) { this.required = r; return this; } };
      items.push(value); return value;
    }
    builder.addDemographicsPage_({ addPageBreakItem: () => item('page'),
      addMultipleChoiceItem: () => item('radio'), addCheckboxItem: () => item('checkbox') }, builder.TEXT[lang]);
    const d3 = items.find(i => i.title === builder.TEXT[lang].d3.title);
    assert.equal(d3.type, 'checkbox'); assert.equal(d3.required, true); assert.equal(d3.values.length, 4);
  }
});

function ui() {
  const elements = { status: {}, continue: { hidden: true, removeAttribute(k) { delete this[k]; }, focus() {} } };
  const buttons = [{}, {}, {}];
  const callbacks = { calls: 0 };
  const api = { withSuccessHandler(fn) { callbacks.success = fn; return this; },
    withFailureHandler(fn) { callbacks.failure = fn; return this; },
    getRandomForm(lang) { callbacks.calls++; callbacks.language = lang; } };
  const context = vm.createContext({ document: { getElementById: id => elements[id], querySelectorAll: () => buttons },
    google: { script: { run: api } } });
  vm.runInContext(read('sanas_form_router/Index.html').match(/<script>([\s\S]*?)<\/script>/)[1], context);
  return { context, elements, buttons, callbacks };
}
test('router offers a user-activated link and does not assign twice on double click', () => {
  const u = ui(); u.context.start('RU'); u.context.start('RU');
  assert.equal(u.callbacks.calls, 1);
  u.callbacks.success({ url: 'https://docs.google.com/forms/d/e/test_form/viewform' });
  assert.equal(u.elements.continue.hidden, false);
  assert.equal(u.elements.continue.textContent, 'Перейти к опросу');
  assert.match(read('sanas_form_router/Index.html'), /id="continue" target="_top"/);
});
test('RPC failure and malformed destination allow retry without an unsafe link', () => {
  const u = ui(); u.context.start('EN'); u.callbacks.failure();
  assert.ok(u.buttons.every(b => !b.disabled));
  u.context.start('KK'); u.callbacks.success({ url: 'javascript:bad()' });
  assert.equal(u.elements.continue.hidden, true);
  assert.equal(u.elements.continue.href, undefined);
  assert.ok(u.buttons.every(b => !b.disabled));
});

function sheet(name, data = [], formUrl = null, maxRows = 1000, maxColumns = 26) {
  const s = { name, data, rows: maxRows, columns: maxColumns, clears: 0,
    getName: () => name, getFormUrl: () => formUrl,
    getDataRange: () => ({ getDisplayValues: () => data }),
    getMaxRows() { return this.rows; }, getMaxColumns() { return this.columns; },
    insertRowsAfter(_n, count) { this.rows += count; }, insertColumnsAfter(_n, count) { this.columns += count; },
    getFilter: () => null, getBandings: () => [], clear() { this.clears++; this.data = []; },
    clearConditionalFormatRules() {}, setFrozenRows() {}, setRowHeight() {}, setColumnWidth() {},
    getRange(row, col, rows, cols) {
      assert.ok(row + rows - 1 <= this.rows, 'range exceeds grid rows');
      assert.ok(col + cols - 1 <= this.columns, 'range exceeds grid columns');
      const range = { setValues: values => { this.data = plain(values); return range; } };
      for (const key of ['setBackground', 'setFontColor', 'setFontWeight', 'setVerticalAlignment', 'applyRowBanding', 'createFilter', 'setWrap']) range[key] = () => range;
      return range;
    }
  }; return s;
}
function fixture({ failed = false, locked = false } = {}) {
  const tabs = new Map(['Респонденты', 'Все ответы', 'Источники', 'Мои заметки'].map(n => [n, sheet(n, [['PREVIOUS']])]));
  const master = { getSheetByName: n => tabs.get(n), insertSheet: n => { const s = sheet(n); tabs.set(n, s); return s; },
    getUrl: () => 'test-master', getSheets: () => [...tabs.values()], deleteSheet() { throw Error('Must not delete tabs'); } };
  let released = false, reads = 0;
  const data = [['Timestamp', 'D3', 'D3'], ['2026-09-12 10:00', '', 'Morning, Evening'], ['', '', '']];
  const response = sheet('Responses', data, 'https://docs.google.com/forms/d/test/edit');
  const notes = sheet('Extra-wide notes', [['memo']], null, 1000, 100);
  const context = load('research/survey/wave2/master_responses.gs', {
    PropertiesService: { getScriptProperties: () => ({ getProperty: () => 'master' }) },
    SpreadsheetApp: { openById(id) { reads++; if (id === 'master') return master;
      if (failed && id === context.RESPONSE_SOURCES[4].spreadsheetId) throw Error('No access');
      return { getSheets: () => [notes, response] }; }, flush() {}, BandingTheme: { LIGHT_GREY: 'grey' } },
    Utilities: { formatDate: () => '2026-09-12 10:00:00' }, Session: { getScriptTimeZone: () => 'UTC' },
    Logger: { log() {} }, LockService: { getScriptLock: () => ({ tryLock: () => !locked, releaseLock() { released = true; } }) }
  });
  return { context, tabs, released: () => released, reads: () => reads };
}
test('12-source sync preserves multi-select/duplicate-title columns and custom tabs; rerun is idempotent', () => {
  const f = fixture(); f.context.syncMasterResponses();
  assert.equal(f.tabs.get('Респонденты').data.length, 13);
  const answers = f.tabs.get('Все ответы').data;
  assert.equal(answers.length, 13); assert.equal(answers[1][7], 3); assert.equal(answers[1][9], 'Morning, Evening');
  assert.deepEqual(f.tabs.get('Мои заметки').data, [['PREVIOUS']]);
  f.context.syncMasterResponses(); assert.deepEqual(f.tabs.get('Все ответы').data, answers);
  assert.equal(f.released(), true);
});
test('one inaccessible source preserves BOTH previous answer tables, reports unknown count, and releases lock', () => {
  const f = fixture({ failed: true });
  assert.throws(() => f.context.syncMasterResponses(), /Master answers unchanged/);
  for (const name of ['Респонденты', 'Все ответы', 'Мои заметки']) assert.deepEqual(f.tabs.get(name).data, [['PREVIOUS']]);
  const failed = f.tabs.get('Источники').data.find(row => row[0] === 'KK-B1-A');
  assert.equal(failed[4], ''); assert.match(failed[5], /ERROR/); assert.equal(f.released(), true);
});
test('concurrent sync exits before reading or writing', () => {
  const f = fixture({ locked: true }); assert.throws(() => f.context.syncMasterResponses(), /Another master sync/);
  assert.equal(f.reads(), 0); assert.equal(f.released(), false);
});
test('source selection rejects no linked form and ambiguous linked forms', () => {
  const f = fixture();
  for (const sheets of [[], [sheet('notes')], [sheet('a', [], 'a'), sheet('b', [], 'b')]]) {
    assert.throws(() => f.context.getResponseSheet_({ getSheets: () => sheets }), /exactly one/);
  }
});
test('writer grows beyond 1000 rows before clearing existing data', () => {
  const f = fixture(); const s = sheet('Все ответы', [['PREVIOUS']], null, 1000, 5);
  const values = Array.from({ length: 1201 }, (_, i) => Array.from({ length: 10 }, () => String(i)));
  f.context.writeTable_(s, values, Array(10).fill(100));
  assert.equal(s.rows, 1201); assert.equal(s.columns, 10); assert.equal(s.data.length, 1201);
});
test('failed access to configured master does not create a replacement', () => {
  const f = fixture(); let creates = 0;
  f.context.SpreadsheetApp.openById = () => { throw Error('Access denied'); };
  f.context.SpreadsheetApp.create = () => { creates++; };
  assert.throws(() => f.context.installMasterResponses(), /Access denied/); assert.equal(creates, 0);
});
