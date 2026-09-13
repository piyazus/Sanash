/**
 * Sanas wave 2 survey builder for Google Forms.
 *
 * Creates the twelve forms of the wave 2 instrument, revision 2: three
 * languages (ru, kk, en) times two choice blocks times two situation orders.
 * Each form gets its own response spreadsheet. Question text comes from
 * instrument_ru_v2.md, instrument_kk_v2.md and instrument_en_v2.md; the choice
 * tasks come from design.json.
 *
 * How to run:
 *   1. script.google.com, new project, paste this file in.
 *   2. Run buildAll(). Authorise Forms, Sheets and Drive when asked.
 *   3. Read the links from the execution log (View, Logs).
 *
 * If buildAll() hits the six-minute execution limit, run buildRu(), buildKk()
 * and buildEn() instead: each builds the four forms of one language.
 * buildOne('ru', 1, 'A') builds a single form if you need to redo one.
 *
 * The script never enables email collection, sign-in or response limits.
 * Nothing is published: the links stay private until someone shares them.
 */

// ---------------------------------------------------------------------------
// Choice tasks. Mirrors design.json. Attention check sits at position 5 and a
// verbatim repeat of task 3 at position 10; neither enters estimation.
// ---------------------------------------------------------------------------

var DESIGN = {
  1: [
    { a: 'packed',   b: 'seated',   wait: 2,  peak: 0 },
    { a: 'packed',   b: 'standing', wait: 2,  peak: 1 },
    { a: 'packed',   b: 'standing', wait: 5,  peak: 0 },
    { a: 'packed',   b: 'seated',   wait: 5,  peak: 1 },
    { a: 'standing', b: 'seated',   wait: 8,  peak: 0 },
    { a: 'standing', b: 'seated',   wait: 8,  peak: 1 },
    { a: 'packed',   b: 'seated',   wait: 12, peak: 0 },
    { a: 'packed',   b: 'standing', wait: 12, peak: 1 }
  ],
  2: [
    { a: 'standing', b: 'seated',   wait: 2,  peak: 0 },
    { a: 'standing', b: 'seated',   wait: 2,  peak: 1 },
    { a: 'packed',   b: 'seated',   wait: 5,  peak: 0 },
    { a: 'packed',   b: 'standing', wait: 5,  peak: 1 },
    { a: 'packed',   b: 'standing', wait: 8,  peak: 0 },
    { a: 'packed',   b: 'seated',   wait: 8,  peak: 1 },
    { a: 'standing', b: 'seated',   wait: 12, peak: 0 },
    { a: 'standing', b: 'seated',   wait: 12, peak: 1 }
  ]
};

var ATTENTION_CHECK = { a: 'seated', b: 'packed', wait: 8, peak: 0 };

/**
 * Two fielding orders per block. Google Forms cannot shuffle sections, so order
 * is randomised by fielding two versions of each block instead: order B is order
 * A reversed, which cancels a linear order or fatigue effect once responses from
 * the two versions are pooled.
 *
 * Each entry is an index into the block's eight design tasks. Position 5 is
 * always the attention check and position 10 always repeats whatever sits at
 * position 3 in that same order, so the test-retest pair stays seven situations
 * apart in both versions.
 */
var ORDERS = {
  A: [0, 1, 2, 3, null, 4, 5, 6, 7],
  B: [7, 6, 5, 4, null, 3, 2, 1, 0]
};

/**
 * Returns the ten situations in fielding order.
 * @param {number} block 1 or 2
 * @param {string} order 'A' or 'B'
 */
function situationsFor_(block, order) {
  var d = DESIGN[block];
  var idx = ORDERS[order];
  if (!idx) throw new Error('Unknown order: ' + order);
  var out = [];
  for (var i = 0; i < idx.length; i++) {
    out.push(idx[i] === null ? ATTENTION_CHECK : d[idx[i]]);
  }
  out.push(out[2]); // position 10 repeats position 3
  return out;
}

// ---------------------------------------------------------------------------
// Text. One object per language, all wording taken verbatim from the
// instrument files. Do not paraphrase here: the three versions are fielded as
// written.
// ---------------------------------------------------------------------------

var TEXT = {};

TEXT.ru = {
  formTitle: 'Опрос: как вы выбираете автобус в Алматы',
  sheetName: 'SANAS W2 RU',
  intro:
    'Этот опрос о том, как вы выбираете автобус в Алматы. Один автобус уже стоит ' +
    'на остановке, другой придёт через несколько минут. Заполнение занимает около ' +
    'шести минут.\n\n' +
    'Правильных ответов нет. Нам важен ваш реальный выбор.\n\n' +
    'Ответы анонимны. Мы не спрашиваем имя, номер телефона и электронную почту. ' +
    'Результаты будут использованы в научном исследовании общественного транспорта ' +
    'в обобщённом виде.\n\n' +
    'Вы можете прекратить в любой момент, закрыв страницу.',
  endMessage: 'Спасибо. Ваши ответы записаны.',
  screenOutMessage: 'Спасибо за интерес. Этот опрос рассчитан на других участников.',

  consent: {
    title: 'Вам исполнилось 18 лет, и вы согласны участвовать?',
    yes: 'Да, согласен участвовать',
    no: 'Нет'
  },

  s1: {
    title: 'Сколько вам лет?',
    options: ['18-24', '25-34', '35-44', '45-54', '55-64', '65 и старше']
  },
  s2: {
    title: 'Как часто вы ездите на городском автобусе в Алматы?',
    keep: ['Каждый день или почти каждый день', '3-4 раза в неделю', '1-2 раза в неделю'],
    drop: ['Реже одного раза в неделю', 'Не езжу на городском автобусе']
  },
  s3: {
    title:
      'Как часто за последний месяц вам попадался автобус настолько полный, ' +
      'что вы сомневались, стоит ли в него садиться?',
    options: [
      'В каждой поездке или почти в каждой',
      'В большинстве поездок',
      'Примерно в половине поездок',
      'Редко',
      'Ни разу'
    ]
  },

  tripHeader: 'Ваша обычная поездка',
  tripHelp:
    'Дальше вопросы про одну поездку: ту, которую вы совершаете на автобусе чаще ' +
    'всего. Держите её в голове, отвечая на следующие вопросы.',
  t1: {
    title: 'Какая это поездка?',
    options: [
      'На работу',
      'На учёбу',
      'По личным делам: документы, магазины, врач',
      'К родным или друзьям, отдых',
      'Другое'
    ]
  },
  t2: {
    title: 'Насколько важно приехать к точному времени в этой поездке?',
    options: [
      'Обязательно приехать вовремя, опоздание имеет последствия',
      'Желательно вовремя, но небольшое опоздание допустимо',
      'Время прибытия не важно'
    ]
  },
  t3: {
    title: 'Сколько примерно занимает эта поездка на автобусе, без учёта ожидания?',
    options: ['Меньше 10 минут', '10-20 минут', '21-30 минут', '31-45 минут', 'Больше 45 минут']
  },
  t4: {
    title: 'Как часто ходит ваш автобус на этом маршруте в обычный день?',
    options: [
      'Каждые 5 минут или чаще',
      'Каждые 6-10 минут',
      'Каждые 11-15 минут',
      'Реже, чем раз в 15 минут',
      'Не знаю'
    ]
  },
  t5: {
    title: 'Сколько вы обычно ждёте автобус на остановке в этой поездке?',
    options: ['Меньше 3 минут', '3-5 минут', '6-10 минут', '11-15 минут', 'Больше 15 минут']
  },

  choiceHeader: 'Ситуации выбора',
  choiceHelp:
    'Дальше десять ситуаций. В каждой вы стоите на остановке своей обычной ' +
    'поездки. Оба автобуса идут туда, куда вам нужно, проезд стоит одинаково, ' +
    'время в пути одинаковое. Один автобус стоит на остановке сейчас. Другой ' +
    'придёт через указанное число минут, и мобильное приложение показывает, ' +
    'насколько он будет полным.\n\n' +
    'Отвечайте так, как поступили бы на самом деле, с вашим реальным запасом ' +
    'времени. Люди в опросах часто говорят, что подождут, а в жизни садятся в ' +
    'первый автобус. Нам нужен ваш настоящий выбор, а не правильный.\n\n' +
    'Что означают уровни заполненности:\n' +
    'Есть свободные места. Вы сядете. Поездка спокойная.\n' +
    'Только стоячие места. Сидячие места заняты, вы поедете стоя, но можете ' +
    'двигаться и нормально держаться.\n' +
    'Битком. Пассажиры стоят вплотную. Двигаться трудно, и есть риск, что вас ' +
    'просто не пустят внутрь.',

  crowding: {
    seated: 'есть свободные места',
    standing: 'только стоячие места',
    packed: 'битком'
  },
  timeLabel: { 0: 'Среда, 14:00.', 1: 'Среда, 08:00.' },
  waitLabel: { 2: '2 минуты', 5: '5 минут', 8: '8 минут', 12: '12 минут' },
  situationTitle: function (i) { return 'Ситуация ' + i + ' из 10'; },
  situationBody: function (time, a, wait, b) {
    return time + '\n' +
      'Автобус на остановке сейчас: ' + a + '.\n' +
      'Следующий автобус через ' + wait + ': ' + b + '.\n\n' +
      'Что вы выберете?';
  },
  boardOption: 'Сажусь в автобус на остановке',
  waitOption: function (wait) { return 'Жду следующий автобус ' + wait; },

  infoHeader: 'Информация о заполненности',
  a1: {
    title:
      'Сколько максимум вы готовы ждать, если точно знаете, что в следующем ' +
      'автобусе будут свободные места, а тот, что стоит на остановке, забит битком?',
    options: [
      'Не буду ждать, сяду в первый автобус',
      '1-2 минуты',
      '3-5 минут',
      '6-10 минут',
      '11-15 минут',
      'Больше 15 минут'
    ]
  },
  a2: {
    title: 'Пользуетесь ли вы сейчас мобильным приложением для поездок на автобусе в Алматы?',
    options: ['Да, почти в каждой поездке', 'Иногда', 'Нет']
  },
  a3: {
    title:
      'Если бы приложение показывало, насколько полон каждый приближающийся ' +
      'автобус, как часто вы бы смотрели туда перед посадкой?',
    options: [
      'Всегда, каждый раз когда жду автобус',
      'В большинстве случаев',
      'Иногда, в зависимости от ситуации',
      'Редко',
      'Не пользовался бы вовсе'
    ]
  },
  a4: {
    title: 'Насколько вы доверяли бы уровню заполненности, который показывает такое приложение?',
    options: [
      'Полностью доверял бы',
      'Доверял бы в большинстве случаев',
      'Доверял бы только если совпадает с тем, что вижу сам',
      'Не доверял бы'
    ]
  },
  a5: {
    title: 'Что бы вы сделали, если бы приложение показало, что оба ближайших автобуса битком?',
    options: [
      'Всё равно сел бы в первый',
      'Подождал бы третий автобус',
      'Поехал бы на такси',
      'Пошёл бы пешком',
      'Отложил бы поездку',
      'Другое'
    ]
  },

  demoHeader: 'О вас',
  d1: { title: 'Ваш пол', options: ['Мужской', 'Женский', 'Предпочитаю не указывать'] },
  d2: {
    title: 'Ваш основной род занятий',
    options: [
      'Студент или школьник',
      'Работаю полный рабочий день',
      'Работаю неполный рабочий день',
      'Свой бизнес или самозанятость',
      'Не работаю',
      'На пенсии',
      'Предпочитаю не указывать'
    ]
  },
  d3: {
    title: 'В какое время суток вы обычно ездите на автобусе? Выберите все подходящие варианты.',
    options: [
      'Утренний час пик, примерно 07:00-09:00',
      'Днём, примерно 09:00-16:00',
      'Вечерний час пик, примерно 17:00-19:00',
      'Вечером, после 19:00'
    ]
  },
  d4: {
    title: 'Из какого района Алматы вы чаще всего выезжаете?',
    options: [
      'Алатауский',
      'Алмалинский',
      'Ауэзовский',
      'Бостандыкский',
      'Жетысуский',
      'Медеуский',
      'Наурызбайский',
      'Турксибский',
      'За пределами Алматы'
    ]
  }
};

TEXT.kk = {
  formTitle: 'Сауалнама: Алматыда автобусты қалай таңдайсыз',
  sheetName: 'SANAS W2 KK',
  intro:
    'Бұл сауалнама Алматыда автобус таңдау туралы. Аялдамада бір автобус тұр, ал ' +
    'екіншісі бірнеше минуттан кейін келеді. Толтыру уақыты шамамен алты минут.\n\n' +
    'Дұрыс немесе бұрыс жауап жоқ. Бізге сіздің нақты таңдауыңыз қажет.\n\n' +
    'Жауаптар анонимді. Аты-жөніңізді, телефон нөміріңізді немесе электрондық ' +
    'поштаңызды сұрамаймыз. Нәтижелер қоғамдық көлік туралы ғылыми зерттеуде ' +
    'жинақталған түрде қолданылады.\n\n' +
    'Кез келген уақытта бетті жауып, тоқтата аласыз.',
  endMessage: 'Рақмет. Жауаптарыңыз жазылды.',
  screenOutMessage: 'Қызығушылығыңызға рақмет. Бұл сауалнама басқа қатысушыларға арналған.',

  consent: {
    title: 'Сізге 18 жас толды ма және қатысуға келісесіз бе?',
    yes: 'Иә, қатысуға келісемін',
    no: 'Жоқ'
  },

  s1: {
    title: 'Жасыңыз қанша?',
    options: ['18-24', '25-34', '35-44', '45-54', '55-64', '65 және одан жоғары']
  },
  s2: {
    title: 'Алматыда қалалық автобуспен қаншалықты жиі жүресіз?',
    keep: ['Күн сайын немесе дерлік күн сайын', 'Аптасына 3-4 рет', 'Аптасына 1-2 рет'],
    drop: ['Аптасына бір реттен сирек', 'Қалалық автобуспен жүрмеймін']
  },
  s3: {
    title:
      'Соңғы бір айда мінуге екі ойлы болатындай толы автобусқа қаншалықты жиі тап болдыңыз?',
    options: [
      'Әр сапарымда немесе дерлік әр сапарымда',
      'Сапарларымның көп бөлігінде',
      'Сапарларымның шамамен жартысында',
      'Сирек',
      'Мүлде кездестірмедім'
    ]
  },

  tripHeader: 'Сіздің әдеттегі сапарыңыз',
  tripHelp:
    'Келесі сұрақтар бір сапар туралы: автобуспен ең жиі жасайтын сапарыңыз ' +
    'туралы. Жауап бергенде соны есіңізде ұстаңыз.',
  t1: {
    title: 'Бұл қандай сапар?',
    options: [
      'Жұмысқа',
      'Оқуға',
      'Жеке істер: құжаттар, дүкен, дәрігер',
      'Туыс-таныстарға, демалыс',
      'Басқа'
    ]
  },
  t2: {
    title: 'Бұл сапарда дәл уақытында жету қаншалықты маңызды?',
    options: [
      'Міндетті түрде уақытында жетуім керек, кешігудің салдары бар',
      'Уақытында жеткен дұрыс, бірақ шамалы кешігуге болады',
      'Жету уақыты маңызды емес'
    ]
  },
  t3: {
    title: 'Бұл сапар автобуста шамамен қанша уақыт алады, күтуді есептемегенде?',
    options: ['10 минуттан аз', '10-20 минут', '21-30 минут', '31-45 минут', '45 минуттан артық']
  },
  t4: {
    title: 'Осы бағыттағы автобусыңыз кәдімгі күні қаншалықты жиі жүреді?',
    options: [
      'Әр 5 минут сайын немесе одан жиі',
      'Әр 6-10 минут сайын',
      'Әр 11-15 минут сайын',
      '15 минутта бір реттен сирек',
      'Білмеймін'
    ]
  },
  t5: {
    title: 'Бұл сапарда аялдамада әдетте қанша уақыт күтесіз?',
    options: ['3 минуттан аз', '3-5 минут', '6-10 минут', '11-15 минут', '15 минуттан артық']
  },

  choiceHeader: 'Таңдау жағдайлары',
  choiceHelp:
    'Алдыңызда он жағдай. Әрқайсысында сіз өзіңіздің әдеттегі сапарыңыздың ' +
    'аялдамасында тұрсыз. Екі автобус те сіз баратын бағытта жүреді, жол ақысы ' +
    'бірдей және жолда өтетін уақыт бірдей. Бір автобус қазір аялдамада тұр. ' +
    'Екіншісі көрсетілген минуттан кейін келеді, ал мобильді қосымша оның ' +
    'қаншалықты толы болатынын көрсетеді.\n\n' +
    'Шын мәнінде қалай істесеңіз, солай жауап беріңіз, қолыңыздағы нақты уақыт ' +
    'қорымен. Сауалнамада адамдар жиі күтемін дейді де, өмірде бірінші автобусқа ' +
    'мініп кетеді. Бізге дұрыс емес, шынайы таңдау керек.\n\n' +
    'Толықтық деңгейлері нені білдіреді:\n' +
    'Отыратын орын бар. Отырып кетесіз. Сапар жайлы.\n' +
    'Тек тұратын орын. Орындықтар бос емес, тұрып барасыз, бірақ қозғалуға және ' +
    'қалыпты ұстануға болады.\n' +
    'Өте толы. Жолаушылар тығыз тұр. Қозғалу қиын, тіпті сізді ішке кіргізбеуі де мүмкін.',

  crowding: {
    seated: 'отыратын орын бар',
    standing: 'тек тұратын орын',
    packed: 'өте толы'
  },
  timeLabel: { 0: 'Сәрсенбі, сағат 14:00.', 1: 'Сәрсенбі, сағат 08:00.' },
  waitLabel: { 2: '2 минуттан', 5: '5 минуттан', 8: '8 минуттан', 12: '12 минуттан' },
  situationTitle: function (i) { return i + '-жағдай, барлығы 10'; },
  situationBody: function (time, a, wait, b) {
    return time + '\n' +
      'Қазір аялдамада тұрған автобус: ' + a + '.\n' +
      'Келесі автобус ' + wait + ' кейін: ' + b + '.\n\n' +
      'Қайсысын таңдайсыз?';
  },
  boardOption: 'Аялдамадағы автобусқа мінемін',
  waitOption: function (wait) {
    return 'Келесі автобусты ' + wait.replace(' минуттан', ' минут') + ' күтемін';
  },

  infoHeader: 'Толықтық туралы ақпарат',
  a1: {
    title:
      'Аялдамадағы автобус өте толы, ал келесі автобуста отыратын орын бар екеніне ' +
      'сенімді болсаңыз, ең көп дегенде қанша күтер едіңіз?',
    options: [
      'Күтпеймін, бірінші автобусқа мінемін',
      '1-2 минут',
      '3-5 минут',
      '6-10 минут',
      '11-15 минут',
      '15 минуттан артық'
    ]
  },
  a2: {
    title: 'Қазір Алматыда автобус сапары үшін мобильді қосымша қолданасыз ба?',
    options: ['Иә, сапарларымның барлығына жуығында', 'Кейде', 'Жоқ']
  },
  a3: {
    title:
      'Егер қосымша әр келе жатқан автобустың қаншалықты толы екенін көрсетсе, ' +
      'мінер алдында оны қаншалықты жиі қарар едіңіз?',
    options: [
      'Әрдайым, автобус күткен сайын',
      'Көп жағдайда',
      'Кейде, жағдайға байланысты',
      'Сирек',
      'Мүлде пайдаланбас едім'
    ]
  },
  a4: {
    title: 'Мұндай қосымша көрсеткен толықтық деңгейіне қаншалықты сенер едіңіз?',
    options: [
      'Толық сенемін',
      'Көп жағдайда сенемін',
      'Тек өз көзіммен көргеніме сәйкес келсе сенемін',
      'Сенбеймін'
    ]
  },
  a5: {
    title: 'Қосымша алдағы екі автобустың да өте толы екенін көрсетсе, не істер едіңіз?',
    options: [
      'Сонда да бірінші автобусқа мінемін',
      'Үшінші автобусты күтемін',
      'Таксимен кетемін',
      'Жаяу барамын',
      'Сапарды кейінге қалдырамын',
      'Басқа'
    ]
  },

  demoHeader: 'Сіз туралы',
  d1: { title: 'Жынысыңыз', options: ['Ер адам', 'Әйел адам', 'Айтқым келмейді'] },
  d2: {
    title: 'Қазіргі негізгі қызметіңіз',
    options: [
      'Студент немесе оқушы',
      'Толық жұмыс күні бойынша жұмыс істеймін',
      'Толық емес жұмыс күні бойынша жұмыс істеймін',
      'Өз бизнесім бар немесе өзін-өзі жұмыспен қамтығанмын',
      'Жұмыссызбын',
      'Зейнеткермін',
      'Айтқым келмейді'
    ]
  },
  d3: {
    title: 'Күннің қай уақытында автобуспен әдетте жүресіз? Барлық сәйкес нұсқаларды таңдаңыз.',
    options: [
      'Таңғы қарбалас, шамамен 07:00-09:00',
      'Күндіз, шамамен 09:00-16:00',
      'Кешкі қарбалас, шамамен 17:00-19:00',
      'Кеш, 19:00-ден кейін'
    ]
  },
  d4: {
    title: 'Алматының қай ауданынан жиі жол жүресіз?',
    options: [
      'Алатау ауданы',
      'Алмалы ауданы',
      'Әуезов ауданы',
      'Бостандық ауданы',
      'Жетісу ауданы',
      'Медеу ауданы',
      'Наурызбай ауданы',
      'Түрксіб ауданы',
      'Алматыдан тыс'
    ]
  }
};

TEXT.en = {
  formTitle: 'Survey: how you choose between buses in Almaty',
  sheetName: 'SANAS W2 EN',
  intro:
    'This survey asks how you choose between buses in Almaty when one bus is ' +
    'already at the stop and another is coming shortly. It takes about six ' +
    'minutes.\n\n' +
    'There are no right answers. We want your real choice.\n\n' +
    'Your answers are anonymous. We do not collect your name, phone number or ' +
    'email address. The results will be used for academic research on public ' +
    'transport and may be published in aggregate form.\n\n' +
    'You can stop at any time by closing the page.',
  endMessage: 'Thank you. Your answers have been recorded.',
  screenOutMessage: 'Thank you for your interest. This survey is aimed at other participants.',

  consent: {
    title: 'Are you 18 or older, and do you agree to take part?',
    yes: 'Yes, I agree to take part',
    no: 'No'
  },

  s1: {
    title: 'How old are you?',
    options: ['18-24', '25-34', '35-44', '45-54', '55-64', '65 or older']
  },
  s2: {
    title: 'How often do you travel by city bus in Almaty?',
    keep: ['Daily or almost daily', '3-4 times a week', '1-2 times a week'],
    drop: ['Less than once a week', 'I do not travel by city bus']
  },
  s3: {
    title:
      'In the last month, how often have you encountered a bus so crowded that you hesitated to board?',
    options: [
      'Every trip or almost every trip',
      'On most trips',
      'On about half of my trips',
      'Rarely',
      'Never'
    ]
  },

  tripHeader: 'Your usual trip',
  tripHelp:
    'The next questions are about one trip: the bus trip you make most often. ' +
    'Keep it in mind while you answer.',
  t1: {
    title: 'What kind of trip is it?',
    options: [
      'To work',
      'To school or university',
      'Personal errands: documents, shopping, medical',
      'Visiting family or friends, leisure',
      'Other'
    ]
  },
  t2: {
    title: 'How important is it to arrive on time on this trip?',
    options: [
      'I must arrive on time, being late has consequences',
      'I prefer to be on time, but a small delay is acceptable',
      'Arrival time does not matter'
    ]
  },
  t3: {
    title: 'Roughly how long does this trip take on the bus, not counting the wait?',
    options: ['Under 10 minutes', '10-20 minutes', '21-30 minutes', '31-45 minutes', 'Over 45 minutes']
  },
  t4: {
    title: 'How often does your bus run on this route on a normal day?',
    options: [
      'Every 5 minutes or more often',
      'Every 6-10 minutes',
      'Every 11-15 minutes',
      'Less often than every 15 minutes',
      'I do not know'
    ]
  },
  t5: {
    title: 'How long do you usually wait at the stop on this trip?',
    options: ['Under 3 minutes', '3-5 minutes', '6-10 minutes', '11-15 minutes', 'Over 15 minutes']
  },

  choiceHeader: 'Choice tasks',
  choiceHelp:
    'Ten situations follow. In each one you are at the stop of your usual trip. ' +
    'Both buses go where you are going, the fare is the same and the in-vehicle ' +
    'time is the same. One bus is at the stop now. The other arrives in the ' +
    'stated number of minutes, and a mobile application tells you how full it ' +
    'will be.\n\n' +
    'Answer as you would actually behave, with the time you really have in hand. ' +
    'People in surveys often say they would wait and then board the first bus ' +
    'anyway. We want your real choice, not the right one.\n\n' +
    'What the crowding levels mean:\n' +
    'Seats available. You will sit down. A comfortable trip.\n' +
    'Standing room. Seats are taken, you will stand, but you can move and hold on normally.\n' +
    'Packed. Passengers are pressed together. Moving is hard, and you may not be ' +
    'let on board at all.',

  crowding: {
    seated: 'seats available',
    standing: 'standing room',
    packed: 'packed'
  },
  timeLabel: { 0: 'Wednesday, 14:00.', 1: 'Wednesday, 08:00.' },
  waitLabel: { 2: '2 minutes', 5: '5 minutes', 8: '8 minutes', 12: '12 minutes' },
  situationTitle: function (i) { return 'Situation ' + i + ' of 10'; },
  situationBody: function (time, a, wait, b) {
    return time + '\n' +
      'Bus at the stop now: ' + a + '.\n' +
      'Next bus in ' + wait + ': ' + b + '.\n\n' +
      'Which do you choose?';
  },
  boardOption: 'I board the bus at the stop',
  waitOption: function (wait) { return 'I wait ' + wait + ' for the next bus'; },

  infoHeader: 'Crowding information',
  a1: {
    title:
      'What is the longest you would wait if you knew for certain that the next ' +
      'bus has seats available and the bus at the stop is packed?',
    options: [
      'I would not wait, I would board the first bus',
      '1-2 minutes',
      '3-5 minutes',
      '6-10 minutes',
      '11-15 minutes',
      'Over 15 minutes'
    ]
  },
  a2: {
    title: 'Do you currently use a mobile application for bus trips in Almaty?',
    options: ['Yes, on almost every trip', 'Sometimes', 'No']
  },
  a3: {
    title:
      'If an application showed how full each approaching bus is, how often would you check it before boarding?',
    options: [
      'Always, every time I wait for a bus',
      'Most of the time',
      'Sometimes, depending on the situation',
      'Rarely',
      'I would not use it at all'
    ]
  },
  a4: {
    title: 'How much would you trust the crowding level shown by such an application?',
    options: [
      'I would trust it completely',
      'I would trust it most of the time',
      'I would trust it only if it matches what I can see myself',
      'I would not trust it'
    ]
  },
  a5: {
    title: 'What would you do if the application showed that both of the next two buses are packed?',
    options: [
      'Board the first one anyway',
      'Wait for a third bus',
      'Take a taxi',
      'Walk',
      'Postpone the trip',
      'Other'
    ]
  },

  demoHeader: 'About you',
  d1: { title: 'Your gender', options: ['Male', 'Female', 'Prefer not to say'] },
  d2: {
    title: 'Your main occupation',
    options: [
      'Student or pupil',
      'Employed full time',
      'Employed part time',
      'Self-employed or business owner',
      'Not working',
      'Retired',
      'Prefer not to say'
    ]
  },
  d3: {
    title: 'When do you usually travel by bus? Select all that apply.',
    options: [
      'Morning peak, roughly 07:00-09:00',
      'Midday, roughly 09:00-16:00',
      'Evening peak, roughly 17:00-19:00',
      'Evening, after 19:00'
    ]
  },
  d4: {
    title: 'Which district of Almaty do you most often travel from?',
    options: [
      'Alatau',
      'Almaly',
      'Auezov',
      'Bostandyk',
      'Zhetysu',
      'Medeu',
      'Nauryzbai',
      'Turksib',
      'Outside Almaty'
    ]
  }
};

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/**
 * Builds all twelve forms and logs their links: three languages, two blocks,
 * two situation orders. Apps Script caps a single execution at six minutes, so
 * if the run times out, call buildSome() for the rest.
 */
function buildAll() {
  var langs = ['ru', 'kk', 'en'];
  var orders = ['A', 'B'];
  var lines = [];
  for (var i = 0; i < langs.length; i++) {
    for (var block = 1; block <= 2; block++) {
      for (var k = 0; k < orders.length; k++) {
        var built = buildOne(langs[i], block, orders[k]);
        lines.push(
          built.name + '\n  live: ' + built.liveUrl + '\n  edit: ' + built.editUrl +
          '\n  responses: ' + built.sheetUrl
        );
      }
    }
  }
  Logger.log(lines.join('\n\n'));
}

/** Wrappers so a language can be built straight from the Run menu. */
function buildRu() { buildSome('ru'); }
function buildKk() { buildSome('kk'); }
function buildEn() { buildSome('en'); }

/**
 * Builds the four forms of one language. Use this if buildAll() runs out of
 * execution time: buildSome('ru'), then buildSome('kk'), then buildSome('en').
 * @param {string} lang 'ru', 'kk' or 'en'
 */
function buildSome(lang) {
  var lines = [];
  for (var block = 1; block <= 2; block++) {
    var orders = ['A', 'B'];
    for (var k = 0; k < orders.length; k++) {
      var built = buildOne(lang, block, orders[k]);
      lines.push(
        built.name + '\n  live: ' + built.liveUrl + '\n  edit: ' + built.editUrl +
        '\n  responses: ' + built.sheetUrl
      );
    }
  }
  Logger.log(lines.join('\n\n'));
}

/**
 * Builds one form.
 * @param {string} lang 'ru', 'kk' or 'en'
 * @param {number} block 1 or 2
 * @param {string} order 'A' or 'B', defaults to 'A'
 */
function buildOne(lang, block, order) {
  var t = TEXT[lang];
  order = order || 'A';
  if (!t) throw new Error('Unknown language: ' + lang);
  if (block !== 1 && block !== 2) throw new Error('Block must be 1 or 2');
  if (!ORDERS[order]) throw new Error('Order must be A or B');

  var name = t.sheetName + ' Block ' + block + ' Order ' + order;
  var form = FormApp.create(name);
  form.setTitle(t.formTitle);
  form.setDescription(t.intro);
  form.setProgressBar(true);
  form.setAllowResponseEdits(false);
  form.setShowLinkToRespondAgain(false);
  form.setLimitOneResponsePerUser(false);
  form.setConfirmationMessage(t.endMessage);
  setNoEmailCollection_(form);

  addConsentPage_(form, t);
  addScreeningPages_(form, t);
  addTripPage_(form, t);
  addChoicePages_(form, t, block, order);
  addInfoPage_(form, t);
  addDemographicsPage_(form, t);

  var ss = SpreadsheetApp.create(name + ' responses');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, ss.getId());

  return {
    name: name,
    liveUrl: form.getPublishedUrl(),
    editUrl: form.getEditUrl(),
    sheetUrl: ss.getUrl()
  };
}

// ---------------------------------------------------------------------------
// Page builders
// ---------------------------------------------------------------------------

/**
 * Consent, alone on the opening page so that "No" can end the form. Page
 * navigation only works from the last item of a page.
 */
function addConsentPage_(form, t) {
  var item = form.addMultipleChoiceItem();
  item.setTitle(t.consent.title).setRequired(true);
  item.setChoices([
    item.createChoice(t.consent.yes, FormApp.PageNavigationType.CONTINUE),
    item.createChoice(t.consent.no, FormApp.PageNavigationType.SUBMIT)
  ]);
}

function addScreeningPages_(form, t) {
  form.addPageBreakItem();
  addRadio_(form, t.s1.title, t.s1.options);

  // S2 alone on its page: the last two options screen the respondent out.
  form.addPageBreakItem();
  var s2 = form.addMultipleChoiceItem();
  s2.setTitle(t.s2.title).setRequired(true);
  var choices = [];
  for (var i = 0; i < t.s2.keep.length; i++) {
    choices.push(s2.createChoice(t.s2.keep[i], FormApp.PageNavigationType.CONTINUE));
  }
  for (var j = 0; j < t.s2.drop.length; j++) {
    choices.push(s2.createChoice(t.s2.drop[j], FormApp.PageNavigationType.SUBMIT));
  }
  s2.setChoices(choices);

  form.addPageBreakItem();
  addRadio_(form, t.s3.title, t.s3.options);
}

function addTripPage_(form, t) {
  form.addPageBreakItem().setTitle(t.tripHeader).setHelpText(t.tripHelp);
  addRadio_(form, t.t1.title, t.t1.options);
  addRadio_(form, t.t2.title, t.t2.options);
  addRadio_(form, t.t3.title, t.t3.options);
  addRadio_(form, t.t4.title, t.t4.options);
  addRadio_(form, t.t5.title, t.t5.options);
}

/** One page of instructions, then one page per situation. */
function addChoicePages_(form, t, block, order) {
  form.addPageBreakItem().setTitle(t.choiceHeader).setHelpText(t.choiceHelp);

  var situations = situationsFor_(block, order);
  for (var i = 0; i < situations.length; i++) {
    var s = situations[i];
    var wait = t.waitLabel[s.wait];
    form.addPageBreakItem().setTitle(t.situationTitle(i + 1));
    var item = form.addMultipleChoiceItem();
    item.setTitle(
      t.situationBody(t.timeLabel[s.peak], t.crowding[s.a], wait, t.crowding[s.b])
    );
    item.setChoices([
      item.createChoice(t.boardOption),
      item.createChoice(t.waitOption(wait))
    ]);
    item.setRequired(true);
  }
}

function addInfoPage_(form, t) {
  form.addPageBreakItem().setTitle(t.infoHeader);
  addRadio_(form, t.a1.title, t.a1.options);
  addRadio_(form, t.a2.title, t.a2.options);
  addRadio_(form, t.a3.title, t.a3.options);
  addRadio_(form, t.a4.title, t.a4.options);
  addRadio_(form, t.a5.title, t.a5.options);
}

function addDemographicsPage_(form, t) {
  form.addPageBreakItem().setTitle(t.demoHeader);
  addRadio_(form, t.d1.title, t.d1.options);
  addRadio_(form, t.d2.title, t.d2.options);
  addCheckbox_(form, t.d3.title, t.d3.options);
  addRadio_(form, t.d4.title, t.d4.options);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function addRadio_(form, title, options) {
  var item = form.addMultipleChoiceItem();
  item.setTitle(title).setChoiceValues(options).setRequired(true);
  return item;
}

function addCheckbox_(form, title, options) {
  var item = form.addCheckboxItem();
  item.setTitle(title).setChoiceValues(options).setRequired(true);
  return item;
}

/**
 * Email collection off. setEmailCollectionType is the current call and
 * setCollectEmail the older one; which exists depends on the account, so try
 * both and fail loudly only if neither works.
 */
function setNoEmailCollection_(form) {
  try {
    form.setEmailCollectionType(FormApp.EmailCollectionType.DO_NOT_COLLECT);
    return;
  } catch (e) {
    // fall through to the older call
  }
  form.setCollectEmail(false);
}
