"""Generate the wave 2 questionnaire in Kazakh, Russian and English.

All three versions are written from one specification and from `design.json`, so
the choice tasks cannot drift apart between languages. Wave 1 shipped instrument
files that described a questionnaire nobody had administered; generating the text
from the design removes that failure mode.

Run after `generate_design.py`. Writes `instrument_kk.md`, `instrument_ru.md`
and `instrument_en.md`.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN = HERE / "design.json"

CROWDING = {
    "kk": {
        "seated": "отыратын орын бар",
        "standing": "тек тұратын орын",
        "packed": "өте толы",
    },
    "ru": {
        "seated": "есть свободные места",
        "standing": "только стоячие места",
        "packed": "битком",
    },
    "en": {
        "seated": "seats available",
        "standing": "standing room",
        "packed": "packed",
    },
}

TEXT = {
    "kk": {
        "title": "2-толқын сауалнамасы — қазақша нұсқа",
        "subtitle": (
            "Жоба: Sanas, Алматы автобустарының толықтығы туралы нақты уақыттағы "
            "ақпарат. Нұсқа 1, 2026-09-01. Дизайн негіздемесі: `DESIGN_NOTE.md`."
        ),
        "intro_head": "Респондентке көрсетілетін кіріспе мәтін",
        "intro": (
            "Бұл сауалнама Алматыда автобус таңдау туралы. Аялдамада бір автобус "
            "тұр, ал екіншісі жақын арада келеді. Сізден қайсысын таңдайтыныңызды "
            "сұраймыз. Толтыру уақыты шамамен алты минут.\n\n"
            "Дұрыс немесе бұрыс жауап жоқ. Бізге сіздің нақты таңдауыңыз қажет.\n\n"
            "Жауаптар анонимді. Аты-жөніңізді, телефон нөміріңізді немесе "
            "электрондық поштаңызды сұрамаймыз. Нәтижелер қоғамдық көлік "
            "туралы ғылыми зерттеуде жинақталған түрде қолданылады.\n\n"
            "Кез келген уақытта бетті жауып, тоқтата аласыз."
        ),
        "screen_head": "1-бөлім: Іріктеу сұрақтары",
        "s1": "С1. Жасыңыз қанша?",
        "s1_opts": [
            "18-ден төмен",
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65 және одан жоғары",
        ],
        "s1_stop": (
            "_«18-ден төмен» таңдаған респондент үшін сауалнама осы жерде аяқталады._"
        ),
        "s2": "С2. Алматыда қалалық автобуспен қаншалықты жиі жүресіз?",
        "s2_opts": [
            "Күн сайын немесе дерлік күн сайын",
            "Аптасына 3-4 рет",
            "Аптасына 1-2 рет",
            "Аптасына бір реттен сирек",
            "Қалалық автобуспен жүрмеймін",
        ],
        "s2_stop": "_Соңғы екі жауапты таңдаған респондент үшін сауалнама аяқталады._",
        "s3": (
            "С3. Соңғы бір айда мінуге екі ойлы болатындай толы автобусқа "
            "қаншалықты жиі тап болдыңыз?"
        ),
        "s3_opts": [
            "Әр сапарымда немесе дерлік әр сапарымда",
            "Сапарларымның көп бөлігінде",
            "Сапарларымның шамамен жартысында",
            "Сирек",
            "Мүлде кездестірмедім",
        ],
        "tasks_head": "2-бөлім: Таңдау сұрақтары",
        "instructions": (
            "**Нұсқаулық.** Әр жағдайда сіз аялдамада тұрсыз. Екі автобус те сіз "
            "баратын бағытта жүреді, жол ақысы бірдей. Бір автобус қазір "
            "аялдамада тұр. Екіншісі бірнеше минуттан кейін келеді, ал мобильді "
            "қосымша оның қаншалықты толы болатынын көрсетеді. Шын мәнінде "
            "қайсысын таңдар едіңіз, соны белгілеңіз."
        ),
        "levels_head": "Толықтық деңгейлері бірінші сұраққа дейін бір рет түсіндіріледі:",
        "levels": [
            "**Отыратын орын бар** — отыруға орын табасыз.",
            "**Тек тұратын орын** — орындықтар бос емес, тұрып барасыз, бірақ "
            "қозғалуға болады.",
            "**Өте толы** — жолаушылар тығыз, қозғалу қиын.",
        ],
        "example_head": "Әр сұрақ осы түрде беріледі:",
        "example": (
            "> **{n}-сұрақ, барлығы 10.** Сәрсенбі, сағат {time}.\n"
            "> Аялдамада тұрған автобус: **{a}**.\n"
            "> Келесі автобус **{w} минуттан** кейін келеді: **{b}**.\n"
            ">\n"
            "> Қайсысын таңдайсыз?\n"
            "> - Аялдамадағы автобусқа мінемін\n"
            "> - Келесі автобусты {w} минут күтемін"
        ),
        "block_head": "{b}-блок сұрақтары",
        "cols": ["Сұрақ", "Аялдамадағы автобус", "Келесі автобус", "Келеді", "Уақыты"],
        "min": "мин",
        "peak": "сәрсенбі 08:00",
        "offpeak": "сәрсенбі 14:00",
        "checks_head": "Есептеуге кірмейтін екі қосымша сұрақ",
        "checks": (
            "| Орны | Мақсаты | Мазмұны |\n|---|---|---|\n"
            "| 5 | Зейін тексерісі | Аялдамадағы автобуста отыратын орын бар, "
            "келесі автобус 8 минуттан кейін келеді және өте толы. Күтуді таңдау "
            "зейінсіздікті білдіреді. |\n"
            "| 10 | Қайталау тексерісі | 3-сұрақ сөзбе-сөз қайталанады. |\n\n"
            "Сұрақтардың реті блок ішінде кездейсоқ араластырылады, тек осы екеуі "
            "өз орнында қалады."
        ),
        "att_head": "3-бөлім: Нақты уақыттағы ақпаратқа қатысты көзқарас",
        "a1": (
            "А1. Келесі автобуста отыратын орын болатынына сенімді болсаңыз, аз "
            "толы автобусты ең көп дегенде қанша күтер едіңіз?"
        ),
        "a1_opts": [
            "Күтпеймін, қай автобус келсе, соған мінемін",
            "1-2 минут",
            "3-5 минут",
            "6-10 минут",
            "10 минуттан артық",
        ],
        "a2": (
            "А2. Қазір Алматыда автобус сапарын жоспарлау үшін мобильді қосымша "
            "қолданасыз ба?"
        ),
        "a2_opts": ["Иә, сапарларымның көбінде", "Кейде", "Жоқ"],
        "a3": (
            "А3. Егер қосымша әр келе жатқан автобустың қаншалықты толы екенін "
            "көрсетсе, мінер алдында оны қаншалықты жиі қарар едіңіз?"
        ),
        "a3_opts": [
            "Әрдайым, автобус күткен сайын",
            "Көп жағдайда",
            "Кейде, жағдайға байланысты",
            "Сирек",
            "Мүлде пайдаланбас едім",
        ],
        "a4": (
            "А4. Мұндай қосымша көрсеткен толықтық деңгейіне қаншалықты сенер едіңіз?"
        ),
        "a4_opts": [
            "Толық сенемін",
            "Көп жағдайда сенемін",
            "Тек өз көзіммен көргеніме сәйкес келсе сенемін",
            "Сенбеймін",
        ],
        "dem_head": "4-бөлім: Демографиялық деректер",
        "d1": "Д1. Жынысыңыз",
        "d1_opts": ["Ер адам", "Әйел адам", "Айтқым келмейді"],
        "d2": "Д2. Қазіргі негізгі қызметіңіз",
        "d2_opts": [
            "Студент немесе оқушы",
            "Толық жұмыс күні бойынша жұмыс істеймін",
            "Толық емес жұмыс күні бойынша жұмыс істеймін",
            "Өз бизнесім бар немесе өзін-өзі жұмыспен қамтығанмын",
            "Жұмыссызбын",
            "Зейнеткермін",
            "Айтқым келмейді",
        ],
        "d3": "Д3. Автобуспен жүрудің негізгі мақсаты _(бірнешеуін таңдауға болады)_",
        "d3_opts": [
            "Жұмысқа бару",
            "Оқуға бару",
            "Жеке істер",
            "Демалу немесе кездесулер",
            "Басқа",
        ],
        "d4": "Д4. Күннің қай уақытында автобуспен жиі жүресіз?",
        "d4_opts": [
            "Таңғы қарбалас, шамамен 07:00-09:00",
            "Күндіз, шамамен 09:00-16:00",
            "Кешкі қарбалас, шамамен 17:00-19:00",
            "Кеш, 19:00-ден кейін",
        ],
        "d5": "Д5. Әдеттегі автобус сапарыңыз қанша уақыт алады?",
        "d5_opts": [
            "10 минуттан аз",
            "10-20 минут",
            "21-40 минут",
            "40 минуттан артық",
        ],
        "d6": (
            "Д6. Алматының қай ауданынан жиі жол жүресіз?\n"
            "_(аудандар тізімі және «Алматыдан тыс» нұсқасы)_"
        ),
    },
    "ru": {
        "title": "Опрос, волна 2 — русская версия",
        "subtitle": (
            "Проект: Sanas, информация о заполненности автобусов Алматы в реальном "
            "времени. Версия 1, 2026-09-01. Обоснование дизайна: `DESIGN_NOTE.md`."
        ),
        "intro_head": "Вступительный текст для респондента",
        "intro": (
            "Этот опрос о том, как вы выбираете автобус в Алматы. Один автобус "
            "уже стоит на остановке, другой придёт через несколько минут. "
            "Заполнение занимает около шести минут.\n\n"
            "Правильных ответов нет. Нам важен ваш реальный выбор.\n\n"
            "Ответы анонимны. Мы не спрашиваем имя, номер телефона и электронную "
            "почту. Результаты будут использованы в научном исследовании "
            "общественного транспорта в обобщённом виде.\n\n"
            "Вы можете прекратить в любой момент, закрыв страницу."
        ),
        "screen_head": "Раздел 1: Отборочные вопросы",
        "s1": "О1. Сколько вам лет?",
        "s1_opts": [
            "Младше 18",
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65 и старше",
        ],
        "s1_stop": "_Для выбравших «Младше 18» опрос заканчивается здесь._",
        "s2": "О2. Как часто вы ездите на городском автобусе в Алматы?",
        "s2_opts": [
            "Каждый день или почти каждый день",
            "3-4 раза в неделю",
            "1-2 раза в неделю",
            "Реже одного раза в неделю",
            "Не езжу на городском автобусе",
        ],
        "s2_stop": "_Для выбравших любой из последних двух вариантов опрос заканчивается._",
        "s3": (
            "О3. Как часто за последний месяц вам попадался автобус настолько "
            "полный, что вы сомневались, стоит ли в него садиться?"
        ),
        "s3_opts": [
            "В каждой поездке или почти в каждой",
            "В большинстве поездок",
            "Примерно в половине поездок",
            "Редко",
            "Ни разу",
        ],
        "tasks_head": "Раздел 2: Вопросы выбора",
        "instructions": (
            "**Инструкция.** В каждой ситуации вы стоите на остановке. Оба "
            "автобуса идут туда, куда вам нужно, проезд стоит одинаково. Один "
            "автобус стоит на остановке сейчас. Другой придёт через несколько "
            "минут, и мобильное приложение показывает, насколько он будет полным. "
            "Выберите автобус, в который вы сели бы на самом деле."
        ),
        "levels_head": "Уровни заполненности объясняются один раз, до первого вопроса:",
        "levels": [
            "**Есть свободные места** — вы сможете сесть.",
            "**Только стоячие места** — сидячие места заняты, вы поедете стоя, но "
            "можете двигаться.",
            "**Битком** — пассажиры стоят вплотную, двигаться трудно.",
        ],
        "example_head": "Каждый вопрос выглядит так:",
        "example": (
            "> **Вопрос {n} из 10.** Среда, {time}.\n"
            "> Автобус, который стоит на остановке: **{a}**.\n"
            "> Следующий автобус придёт через **{w} минут**: **{b}**.\n"
            ">\n"
            "> Что вы выберете?\n"
            "> - Сажусь в автобус на остановке\n"
            "> - Жду следующий автобус {w} минут"
        ),
        "block_head": "Вопросы блока {b}",
        "cols": [
            "Вопрос",
            "Автобус на остановке",
            "Следующий автобус",
            "Придёт через",
            "Время",
        ],
        "min": "мин",
        "peak": "среда 08:00",
        "offpeak": "среда 14:00",
        "checks_head": "Два дополнительных вопроса, не входящих в оценку модели",
        "checks": (
            "| Место | Назначение | Содержание |\n|---|---|---|\n"
            "| 5 | Проверка внимания | В автобусе на остановке есть свободные "
            "места, следующий придёт через 8 минут и будет битком. Выбор «ждать» "
            "означает невнимательность. |\n"
            "| 10 | Проверка на повтор | Вопрос 3 повторяется дословно. |\n\n"
            "Порядок вопросов внутри блока случайный, кроме этих двух, которые "
            "остаются на своих местах."
        ),
        "att_head": "Раздел 3: Отношение к информации в реальном времени",
        "a1": (
            "В1. Сколько максимум вы готовы ждать менее заполненный автобус, если "
            "точно знаете, что в следующем будут свободные места?"
        ),
        "a1_opts": [
            "Не буду ждать, сяду в тот автобус, который придёт первым",
            "1-2 минуты",
            "3-5 минут",
            "6-10 минут",
            "Больше 10 минут",
        ],
        "a2": (
            "В2. Пользуетесь ли вы сейчас мобильным приложением для планирования "
            "поездок на автобусе в Алматы?"
        ),
        "a2_opts": ["Да, в большинстве поездок", "Иногда", "Нет"],
        "a3": (
            "В3. Если бы приложение показывало, насколько полон каждый "
            "приближающийся автобус, как часто вы бы смотрели туда перед посадкой?"
        ),
        "a3_opts": [
            "Всегда, каждый раз когда жду автобус",
            "В большинстве случаев",
            "Иногда, в зависимости от ситуации",
            "Редко",
            "Не пользовался бы вовсе",
        ],
        "a4": (
            "В4. Насколько вы доверяли бы уровню заполненности, который "
            "показывает такое приложение?"
        ),
        "a4_opts": [
            "Полностью доверял бы",
            "Доверял бы в большинстве случаев",
            "Доверял бы только если совпадает с тем, что вижу сам",
            "Не доверял бы",
        ],
        "dem_head": "Раздел 4: Демографические данные",
        "d1": "Д1. Ваш пол",
        "d1_opts": ["Мужской", "Женский", "Предпочитаю не указывать"],
        "d2": "Д2. Ваш основной род занятий",
        "d2_opts": [
            "Студент или школьник",
            "Работаю полный рабочий день",
            "Работаю неполный рабочий день",
            "Свой бизнес или самозанятость",
            "Не работаю",
            "На пенсии",
            "Предпочитаю не указывать",
        ],
        "d3": "Д3. Основная цель ваших поездок на автобусе _(можно выбрать несколько)_",
        "d3_opts": [
            "Поездка на работу",
            "Поездка на учёбу",
            "Личные дела",
            "Отдых или встречи",
            "Другое",
        ],
        "d4": "Д4. В какое время суток вы чаще всего ездите на автобусе?",
        "d4_opts": [
            "Утренний час пик, примерно 07:00-09:00",
            "Днём, примерно 09:00-16:00",
            "Вечерний час пик, примерно 17:00-19:00",
            "Вечером, после 19:00",
        ],
        "d5": "Д5. Сколько обычно занимает ваша поездка на автобусе?",
        "d5_opts": [
            "Меньше 10 минут",
            "10-20 минут",
            "21-40 минут",
            "Больше 40 минут",
        ],
        "d6": (
            "Д6. Из какого района Алматы вы чаще всего выезжаете?\n"
            "_(список районов и вариант «за пределами Алматы»)_"
        ),
    },
    "en": {
        "title": "Wave 2 questionnaire — English version",
        "subtitle": (
            "Project: Sanas, real-time bus occupancy information for Almaty. "
            "Version 1, 2026-09-01. Design rationale: `DESIGN_NOTE.md`."
        ),
        "intro_head": "Introduction shown to the respondent",
        "intro": (
            "This survey asks how you choose between buses in Almaty when one bus "
            "is already at the stop and another is coming shortly. It takes about "
            "six minutes.\n\n"
            "There are no right answers. We want your real choice.\n\n"
            "Your answers are anonymous. We do not collect your name, phone number "
            "or email address. The results will be used for academic research on "
            "public transport and may be published in aggregate form.\n\n"
            "You can stop at any time by closing the page."
        ),
        "screen_head": "Section 1: Screening",
        "s1": "S1. How old are you?",
        "s1_opts": [
            "Under 18",
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65 or older",
        ],
        "s1_stop": '_Respondents selecting "Under 18" end here._',
        "s2": "S2. How often do you travel by city bus in Almaty?",
        "s2_opts": [
            "Daily or almost daily",
            "3-4 times a week",
            "1-2 times a week",
            "Less than once a week",
            "I do not travel by city bus",
        ],
        "s2_stop": "_Respondents selecting either of the last two options end here._",
        "s3": (
            "S3. In the last month, how often have you encountered a bus so "
            "crowded that you hesitated to board?"
        ),
        "s3_opts": [
            "Every trip or almost every trip",
            "On most trips",
            "On about half of my trips",
            "Rarely",
            "Never",
        ],
        "tasks_head": "Section 2: Choice tasks",
        "instructions": (
            "**Instructions.** In each situation you are standing at a bus stop. "
            "Both buses go where you are going and the fare is the same. One bus "
            "is at the stop now. The other arrives in a few minutes, and a mobile "
            "application tells you how full it will be. Choose the bus you would "
            "actually take."
        ),
        "levels_head": "The crowding levels are explained once, before the first task:",
        "levels": [
            "**Seats available** — you will be able to sit down.",
            "**Standing room** — most seats are taken, you will stand, but you can "
            "move.",
            "**Packed** — passengers are pressed together and it is hard to move.",
        ],
        "example_head": "Each task is presented in this form:",
        "example": (
            "> **Task {n} of 10.** It is a Wednesday at {time}.\n"
            "> The bus at the stop now is **{a}**.\n"
            "> The next bus arrives in **{w} minutes** and will have **{b}**.\n"
            ">\n"
            "> Which do you take?\n"
            "> - Board the bus at the stop now\n"
            "> - Wait {w} minutes for the next bus"
        ),
        "block_head": "Block {b} tasks",
        "cols": ["Task", "Bus at the stop", "Next bus", "Arrives in", "Time"],
        "min": "min",
        "peak": "Wednesday 08:00",
        "offpeak": "Wednesday 14:00",
        "checks_head": "Two additional tasks, excluded from estimation",
        "checks": (
            "| Position | Purpose | Content |\n|---|---|---|\n"
            "| 5 | Dominance check | The bus at the stop has seats available; the "
            "next bus arrives in 8 minutes and is packed. Choosing to wait "
            "indicates inattention. |\n"
            "| 10 | Test-retest | Task 3 repeated verbatim. |\n\n"
            "Task order is randomised within the block, except that these two hold "
            "their positions."
        ),
        "att_head": "Section 3: Attitudes toward real-time information",
        "a1": (
            "A1. What is the longest you would wait for a less crowded bus, if you "
            "knew for certain that the next bus would have seats?"
        ),
        "a1_opts": [
            "I would not wait, I take whichever bus arrives first",
            "1-2 minutes",
            "3-5 minutes",
            "6-10 minutes",
            "More than 10 minutes",
        ],
        "a2": (
            "A2. Do you currently use a mobile application to plan bus trips in Almaty?"
        ),
        "a2_opts": ["Yes, most times I travel", "Sometimes", "No"],
        "a3": (
            "A3. If an application showed how full each approaching bus is, how "
            "often would you check it before boarding?"
        ),
        "a3_opts": [
            "Every time I wait for a bus",
            "Most of the time",
            "Sometimes, depending on the situation",
            "Rarely",
            "I would never use it",
        ],
        "a4": (
            "A4. How much would you trust the crowding level shown by such an "
            "application?"
        ),
        "a4_opts": [
            "I would trust it completely",
            "I would trust it most of the time",
            "I would trust it only if it matched what I could see",
            "I would not trust it",
        ],
        "dem_head": "Section 4: Demographics",
        "d1": "D1. Gender",
        "d1_opts": ["Male", "Female", "Prefer not to say"],
        "d2": "D2. What is your current main occupation?",
        "d2_opts": [
            "Student",
            "Employed full-time",
            "Employed part-time",
            "Self-employed or own business",
            "Unemployed",
            "Retired",
            "Prefer not to say",
        ],
        "d3": "D3. What is the main purpose of your bus trips? _(select all that apply)_",
        "d3_opts": [
            "Travel to work",
            "Travel to study",
            "Personal errands",
            "Leisure or social visits",
            "Other",
        ],
        "d4": "D4. At what time of day do you most often travel by bus?",
        "d4_opts": [
            "Morning peak, roughly 07:00-09:00",
            "Midday, roughly 09:00-16:00",
            "Evening peak, roughly 17:00-19:00",
            "Evening, after 19:00",
        ],
        "d5": "D5. How long is your typical bus trip?",
        "d5_opts": [
            "Under 10 minutes",
            "10-20 minutes",
            "21-40 minutes",
            "More than 40 minutes",
        ],
        "d6": (
            "D6. Which district of Almaty do you travel from most often?\n"
            '_(list of districts, plus "outside Almaty")_'
        ),
    },
}

NOTES = {
    "kk": (
        "## Іске асыру бойынша ескертпелер\n\n"
        "- С1 және С2 жауапты жай жазып қана қоймай, сауалнаманы тоқтатуы керек.\n"
        "- Жас туралы сұрақ 18-ден басталады. 1-толқында 14-24 аралығы ұсынылған "
        "еді.\n"
        "- Әр жауаппен бірге блок нөмірі мен сұрақтардың реті жазылуы керек.\n"
        "- Google Forms уақыт белгісі сақталады.\n"
        "- Электрондық пошта немесе кез келген басқа сәйкестендіргіш жиналмайды."
    ),
    "ru": (
        "## Замечания по реализации\n\n"
        "- О1 и О2 должны завершать форму, а не просто записывать ответ.\n"
        "- Вопрос о возрасте начинается с 18. В волне 1 предлагался диапазон "
        "14-24.\n"
        "- С каждым ответом нужно записывать номер блока и порядок вопросов.\n"
        "- Отметка времени Google Forms сохраняется.\n"
        "- Электронная почта и любые другие идентификаторы не собираются."
    ),
    "en": (
        "## Notes for implementation\n\n"
        "- S1 and S2 must terminate the form, not merely record an answer. Wave 1 "
        "mixed ineligible respondents into the analysed sample because screening "
        "was absent.\n"
        "- The age question starts at 18. Wave 1 offered a 14-24 band, so the "
        "collected sample may include minors.\n"
        "- Record the assigned block number and the task order with each response.\n"
        "- Keep the Google Forms timestamp.\n"
        "- Do not collect email addresses or any other identifier."
    ),
}


def question(label, options):
    return "\n".join([f"**{label}**", ""] + [f"- {o}" for o in options] + [""])


def task_table(lang, design, block):
    t = TEXT[lang]
    head = "| " + " | ".join(t["cols"]) + " |"
    rule = "|" + "---|" * len(t["cols"])
    rows = []
    for n, task in enumerate(design[block], 1):
        when = t["peak"] if task["peak"] else t["offpeak"]
        rows.append(
            f"| {n} | {CROWDING[lang][task['bus_a']]} | "
            f"{CROWDING[lang][task['bus_b']]} | {task['wait']} {t['min']} | {when} |"
        )
    return "\n".join([head, rule] + rows)


def build(lang, design):
    t = TEXT[lang]
    first = design["1"][0]
    out = [
        f"# {t['title']}",
        "",
        t["subtitle"],
        "",
        "---",
        "",
        f"## {t['intro_head']}",
        "",
        t["intro"],
        "",
        "---",
        "",
        f"## {t['screen_head']}",
        "",
        question(t["s1"], t["s1_opts"]),
        t["s1_stop"],
        "",
        question(t["s2"], t["s2_opts"]),
        t["s2_stop"],
        "",
        question(t["s3"], t["s3_opts"]),
        "---",
        "",
        f"## {t['tasks_head']}",
        "",
        t["instructions"],
        "",
        t["levels_head"],
        "",
        *[f"- {line}" for line in t["levels"]],
        "",
        t["example_head"],
        "",
        t["example"].format(
            n=1,
            time=t["peak"].split()[-1] if first["peak"] else t["offpeak"].split()[-1],
            a=CROWDING[lang][first["bus_a"]],
            b=CROWDING[lang][first["bus_b"]],
            w=first["wait"],
        ),
        "",
    ]
    for block in ("1", "2"):
        out += [
            f"### {t['block_head'].format(b=block)}",
            "",
            task_table(lang, design, block),
            "",
        ]
    out += [
        f"### {t['checks_head']}",
        "",
        t["checks"],
        "",
        "---",
        "",
        f"## {t['att_head']}",
        "",
        question(t["a1"], t["a1_opts"]),
        question(t["a2"], t["a2_opts"]),
        question(t["a3"], t["a3_opts"]),
        question(t["a4"], t["a4_opts"]),
        "---",
        "",
        f"## {t['dem_head']}",
        "",
        question(t["d1"], t["d1_opts"]),
        question(t["d2"], t["d2_opts"]),
        question(t["d3"], t["d3_opts"]),
        question(t["d4"], t["d4_opts"]),
        question(t["d5"], t["d5_opts"]),
        f"**{t['d6']}**",
        "",
        "---",
        "",
        NOTES[lang],
        "",
    ]
    return "\n".join(out)


def main():
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    for lang in ("kk", "ru", "en"):
        path = HERE / f"instrument_{lang}.md"
        path.write_text(build(lang, design), encoding="utf-8")
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
