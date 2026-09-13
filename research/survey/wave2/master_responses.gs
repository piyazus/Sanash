/**
 * SANAS W2 — live master response workbook.
 *
 * HOW TO USE
 * 1. Add this file to the same Apps Script project as the form router.
 * 2. Run installMasterResponses() once and grant access.
 * 3. Open the URL printed in the execution log.
 *
 * The master workbook is refreshed immediately and then every 15 minutes.
 */

var MASTER_PROPERTY = 'SANAS_W2_MASTER_SPREADSHEET_ID';

var RESPONSE_SOURCES = [
  source_('RU-B1-A', 'RU', 1, 'A', '1XeRsxvwjXo3qZE_wWHFcFJyJcd2lFd13ieFJf2xGO04'),
  source_('RU-B1-B', 'RU', 1, 'B', '1KssCHV9U9SzOA0GadMLAZHcYvjy_FsmyN-ZHW1NnK6U'),
  source_('RU-B2-A', 'RU', 2, 'A', '1SBHKanLE7giudVxBGbiAmh37Td4eYYrnwCOH6DD8HuM'),
  source_('RU-B2-B', 'RU', 2, 'B', '1djRFpt68kwS-mQlFxgxWnNlL0fXYW23iZAhG3T5P0DA'),
  source_('KK-B1-A', 'KK', 1, 'A', '1bdcUZD4ZS68BWbD1nX9eO6f9qdY-uVFtXwNKzwn4jC0'),
  source_('KK-B1-B', 'KK', 1, 'B', '1UqO7Zo6Kruhte8PAeN6lJNcUK7mbgg7cRvGkfqifH94'),
  source_('KK-B2-A', 'KK', 2, 'A', '1w-g1em-jp28pcCrdyrg4HxL5aMTXzAq6tBd1dOncokU'),
  source_('KK-B2-B', 'KK', 2, 'B', '1FkmKIm2wAu8tVQnZ-DNYBWamXt3v2o6HS-xL4kc3dvs'),
  source_('EN-B1-A', 'EN', 1, 'A', '1GACjOsTLTZfMjq3cW99sVos3XtI9twXZB0NHSkXlb78'),
  source_('EN-B1-B', 'EN', 1, 'B', '1S1MSBkLwvv6fcCn0ISrVuN7T_-XySqJnoTKARXDYMHY'),
  source_('EN-B2-A', 'EN', 2, 'A', '1Zn4i6vxbzuZ51k-yX2tb2FL-G48iJfX6Rvm-CUOzIL8'),
  source_('EN-B2-B', 'EN', 2, 'B', '1z03zlcHOWJ1TD6PSCXm9IFwxOPK9j8CLV5jIbv9E3ZE')
];

function source_(code, language, block, order, spreadsheetId) {
  return {
    code: code,
    language: language,
    block: block,
    order: order,
    spreadsheetId: spreadsheetId,
    url: 'https://docs.google.com/spreadsheets/d/' + spreadsheetId + '/edit'
  };
}

/** Creates/reuses the master workbook, syncs it, and installs auto-refresh. */
function installMasterResponses() {
  var properties = PropertiesService.getScriptProperties();
  var masterId = properties.getProperty(MASTER_PROPERTY);
  var master = null;

  if (masterId) {
    // An access error must not silently create a second master workbook.
    master = SpreadsheetApp.openById(masterId);
  }

  if (!master) {
    master = SpreadsheetApp.create('SANAS W2 — ВСЕ ОТВЕТЫ');
    properties.setProperty(MASTER_PROPERTY, master.getId());
  }

  syncMasterResponses();
  installSyncTrigger_();

  Logger.log('MASTER URL: ' + master.getUrl());
  return master.getUrl();
}

/** Refreshes the three master tabs from all 12 response spreadsheets. */
function syncMasterResponses() {
  var lock = LockService.getScriptLock();
  if (!lock.tryLock(1000)) throw new Error('Another master sync is running.');
  try {
    return syncMasterResponsesLocked_();
  } finally {
    lock.releaseLock();
  }
}

function syncMasterResponsesLocked_() {
  var masterId = PropertiesService.getScriptProperties().getProperty(MASTER_PROPERTY);
  if (!masterId) {
    throw new Error('Run installMasterResponses() first.');
  }

  var master = SpreadsheetApp.openById(masterId);
  var answers = [[
    'respondent_id', 'language', 'block', 'order', 'timestamp',
    'source', 'source_row', 'question_column', 'question', 'answer'
  ]];
  var respondents = [[
    'respondent_id', 'language', 'block', 'order', 'timestamp',
    'source', 'source_row', 'answered_questions', 'source_sheet'
  ]];
  var sources = [[
    'source', 'language', 'block', 'order', 'responses',
    'status', 'source_sheet', 'last_attempt'
  ]];
  var failedSources = [];
  var syncedAt = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss');

  RESPONSE_SOURCES.forEach(function (source) {
    try {
      var sourceBook = SpreadsheetApp.openById(source.spreadsheetId);
      var sourceSheet = getResponseSheet_(sourceBook);
      var data = sourceSheet.getDataRange().getDisplayValues();
      var headers = data.length ? data[0] : [];
      var responseCount = 0;

      for (var rowIndex = 1; rowIndex < data.length; rowIndex++) {
        var row = data[rowIndex];
        if (!rowHasData_(row)) continue;

        responseCount++;
        var respondentId = source.code + '-' + (rowIndex + 1);
        var timestamp = row[0] || '';
        var answeredCount = 0;

        for (var columnIndex = 1; columnIndex < headers.length; columnIndex++) {
          var answer = row[columnIndex] || '';
          if (answer === '') continue;
          answeredCount++;
          answers.push([
            respondentId,
            source.language,
            source.block,
            source.order,
            timestamp,
            source.code,
            rowIndex + 1,
            columnIndex + 1,
            headers[columnIndex] || ('Column ' + (columnIndex + 1)),
            answer
          ]);
        }

        respondents.push([
          respondentId,
          source.language,
          source.block,
          source.order,
          timestamp,
          source.code,
          rowIndex + 1,
          answeredCount,
          source.url
        ]);
      }

      sources.push([
        source.code, source.language, source.block, source.order,
        responseCount, 'OK', source.url, syncedAt
      ]);
    } catch (error) {
      failedSources.push(source.code);
      sources.push([
        source.code, source.language, source.block, source.order,
        '', 'ERROR: ' + error.message, source.url, syncedAt
      ]);
    }
  });

  if (failedSources.length) {
    sources.slice(1).forEach(function (row) {
      if (row[5] === 'OK') row[5] = 'READ OK; master unchanged';
    });
    writeTable_(getOrCreateSheet_(master, 'Источники'), sources, [100, 85, 55, 55, 85, 180, 360, 145]);
    SpreadsheetApp.flush();
    throw new Error('Master answers unchanged; failed sources: ' + failedSources.join(', '));
  }

  writeTable_(getOrCreateSheet_(master, 'Респонденты'), respondents, [100, 85, 55, 55, 145, 90, 75, 135, 360]);
  writeTable_(getOrCreateSheet_(master, 'Все ответы'), answers, [100, 85, 55, 55, 145, 90, 75, 105, 420, 420]);
  writeTable_(getOrCreateSheet_(master, 'Источники'), sources, [100, 85, 55, 55, 85, 180, 360, 145]);

  SpreadsheetApp.flush();
  Logger.log('Synced respondents: ' + (respondents.length - 1));
  Logger.log('MASTER URL: ' + master.getUrl());
}

/** Prints the master URL again if the log was closed. */
function showMasterUrl() {
  var masterId = PropertiesService.getScriptProperties().getProperty(MASTER_PROPERTY);
  if (!masterId) throw new Error('Run installMasterResponses() first.');
  var url = SpreadsheetApp.openById(masterId).getUrl();
  Logger.log('MASTER URL: ' + url);
  return url;
}

function rowHasData_(row) {
  for (var i = 0; i < row.length; i++) {
    if (row[i] !== '') return true;
  }
  return false;
}

function getOrCreateSheet_(book, name) {
  return book.getSheetByName(name) || book.insertSheet(name);
}

function getResponseSheet_(book) {
  var sheets = book.getSheets().filter(function (sheet) {
    return Boolean(sheet.getFormUrl());
  });
  if (sheets.length !== 1) {
    throw new Error('Expected exactly one form-linked response tab; found ' + sheets.length);
  }
  return sheets[0];
}

function writeTable_(sheet, values, widths) {
  // Grow the grid before touching the previous snapshot (long answers exceed
  // the default 1,000 rows after only a few dozen complete responses).
  var rowCount = values.length;
  var columnCount = values[0].length;
  if (sheet.getMaxRows() < rowCount) {
    sheet.insertRowsAfter(sheet.getMaxRows(), rowCount - sheet.getMaxRows());
  }
  if (sheet.getMaxColumns() < columnCount) {
    sheet.insertColumnsAfter(sheet.getMaxColumns(), columnCount - sheet.getMaxColumns());
  }
  if (sheet.getFilter()) sheet.getFilter().remove();
  sheet.getBandings().forEach(function (banding) { banding.remove(); });
  sheet.clear();
  sheet.clearConditionalFormatRules();

  sheet.getRange(1, 1, rowCount, columnCount).setValues(values);
  sheet.setFrozenRows(1);

  var header = sheet.getRange(1, 1, 1, columnCount);
  header
    .setBackground('#1F4E78')
    .setFontColor('#FFFFFF')
    .setFontWeight('bold')
    .setVerticalAlignment('middle');
  sheet.setRowHeight(1, 32);

  if (rowCount > 1) {
    var body = sheet.getRange(2, 1, rowCount - 1, columnCount);
    body.setVerticalAlignment('top');
    body.applyRowBanding(SpreadsheetApp.BandingTheme.LIGHT_GREY);
    sheet.getRange(1, 1, rowCount, columnCount).createFilter();
  }

  for (var i = 0; i < widths.length; i++) {
    sheet.setColumnWidth(i + 1, widths[i]);
  }

  if (sheet.getName() === 'Все ответы' && rowCount > 1) {
    sheet.getRange(2, 9, rowCount - 1, 2).setWrap(true);
  }
}

function installSyncTrigger_() {
  ScriptApp.getProjectTriggers().forEach(function (trigger) {
    if (trigger.getHandlerFunction() === 'syncMasterResponses') {
      ScriptApp.deleteTrigger(trigger);
    }
  });

  ScriptApp.newTrigger('syncMasterResponses')
    .timeBased()
    .everyMinutes(15)
    .create();
}
