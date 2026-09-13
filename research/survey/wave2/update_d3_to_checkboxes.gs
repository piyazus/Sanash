/**
 * One-time migration for the 12 existing SANAS W2 forms.
 *
 * Paste this file into the Apps Script project owned by the same Google account
 * as the forms, then run updateD3ToCheckboxes() once from the editor.
 * Run it before collecting responses because replacing an item creates a new
 * response column in the linked spreadsheet.
 */

var SANAS_W2_FORMS = [
  { id: '1z2_Z-8Se1Bd9XJGCDvrLYQFCj9C3HMRyIPWF_na4DGo', language: 'RU' },
  { id: '1xh1PwBszTQMB_elqIXbYfAiFVHdIxyFJ5vVkGYp83Gc', language: 'RU' },
  { id: '1voGP4S3mx90A2av04nNTo6xb-ifi4RXRJTAfcpl0eds', language: 'RU' },
  { id: '1wUYoO9W3T7FQ_-zvfTCtX6PbV0V842P-KszPyAEhFvM', language: 'RU' },
  { id: '1h2_ghtydBPN8kZSk6b_50QY3k037LF1LYwlJiSxTG2I', language: 'KK' },
  { id: '1eeRLCE7QFU8TYiwWwmyrzOEh7IR6APHMm9H2uf5H7sY', language: 'KK' },
  { id: '16kKYKlSyBLpvw-AcHb024YvPCezTEeK5pSbsjeHbM9Q', language: 'KK' },
  { id: '1Z9asksfXi_Lecpgv3cW1dckGPe5LTGnr9YX7XkOiU5w', language: 'KK' },
  { id: '1MkR0RIUvbVrny4oRUPQZ6EWaYF9136BzYOSY_qRbK-Q', language: 'EN' },
  { id: '19lcK4WOs5pbmJbHUCFhH6cIOXMAwZWzT34wMRdk2cnY', language: 'EN' },
  { id: '171TNIJQcYWKcONTLmczXhr20b1BAmE7E-cSwQcEECx4', language: 'EN' },
  { id: '1hgRK72t6Eui9hdiMzlc0vZlAqvFqo9S5EbGDn3wHmCM', language: 'EN' }
];

var D3_TITLES = {
  RU: {
    oldTitle: 'В какое время суток вы чаще всего ездите на автобусе?',
    newTitle: 'В какое время суток вы обычно ездите на автобусе? Выберите все подходящие варианты.'
  },
  KK: {
    oldTitle: 'Күннің қай уақытында автобуспен жиі жүресіз?',
    newTitle: 'Күннің қай уақытында автобуспен әдетте жүресіз? Барлық сәйкес нұсқаларды таңдаңыз.'
  },
  EN: {
    oldTitle: 'When do you most often travel by bus?',
    newTitle: 'When do you usually travel by bus? Select all that apply.'
  }
};

function updateD3ToCheckboxes() {
  var report = [];

  SANAS_W2_FORMS.forEach(function (target) {
    var form = FormApp.openById(target.id);
    var titles = D3_TITLES[target.language];
    var matchingItems = findD3Items_(form, titles);
    var radio = null;
    var checkboxes = [];

    matchingItems.forEach(function (item) {
      if (item.getType() === FormApp.ItemType.MULTIPLE_CHOICE && !radio) {
        radio = item.asMultipleChoiceItem();
      } else if (item.getType() === FormApp.ItemType.CHECKBOX) {
        checkboxes.push(item.asCheckboxItem());
      }
    });

    if (!radio && checkboxes.length === 0) {
      throw new Error('D3 not found in form: ' + form.getTitle());
    }

    if (!radio) {
      checkboxes[0].setTitle(titles.newTitle).setRequired(true);
      deleteExtraCheckboxes_(form, checkboxes.slice(1));
      report.push('Already checkbox: ' + form.getTitle());
      return;
    }

    var values = radio.getChoices().map(function (choice) {
      return choice.getValue();
    });
    var oldIndex = radio.getIndex();
    var checkbox = checkboxes.length > 0 ? checkboxes[0] : form.addCheckboxItem();
    checkbox
      .setTitle(titles.newTitle)
      .setChoiceValues(values)
      .setRequired(radio.isRequired());

    var helpText = radio.getHelpText();
    if (helpText) checkbox.setHelpText(helpText);

    // Delete the old radio item first. Then move the checkbox using the
    // numeric overload, which works in both old and new Apps Script runtimes.
    form.deleteItem(radio.getIndex());
    form.moveItem(checkbox.getIndex(), oldIndex);
    deleteExtraCheckboxes_(form, checkboxes.slice(1));
    report.push('Updated: ' + form.getTitle());
  });

  Logger.log(report.join('\n'));
}

function findD3Items_(form, titles) {
  var items = form.getItems();
  var matches = [];
  for (var i = 0; i < items.length; i++) {
    var title = items[i].getTitle();
    if (title === titles.oldTitle || title === titles.newTitle) {
      matches.push(items[i]);
    }
  }
  return matches;
}

function deleteExtraCheckboxes_(form, extras) {
  // Delete from the highest index down so earlier indexes remain valid.
  extras.sort(function (a, b) { return b.getIndex() - a.getIndex(); });
  extras.forEach(function (item) {
    form.deleteItem(item.getIndex());
  });
}
