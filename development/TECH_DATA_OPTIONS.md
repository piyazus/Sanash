# Sanas: Technical and Dataset Options

Статус: evidence-backed decision brief, финальный стек не выбран  
Дата проверки: 2026-08-26  
Current research contract: `research/RTCI_RESEARCH_CHARTER.md`

## 1. Bottom line

- Публичного датасета, который одновременно содержит обычный автобусный салон,
  потолочную RGB/fisheye-камеру, реальные dense/crush loads, count ground truth
  и чистую commercial licence, не найдено.
- Лучший transfer dataset по геометрии и движению: **PMOF**, CC BY 4.0,
  потолочная fisheye-камера в движущемся пассажирском транспорте. Но только
  1-4 человека.
- Ближайшая опубликованная dense-bus база: **KAOHSIUNG-BUS**, включая сцены
  свыше 25 пассажиров. Она proprietary и публично не скачивается.
- Следовательно, собственные Almaty data обязательны. Публичные наборы могут
  ускорить pretraining/smoke tests, но не подтвердить Sanas accuracy.
- Начинать нужно с простого detector/count baseline. CSRNet + PFCASA остаётся
  кандидатом для dense occlusion, а не зафиксированным stack.
- Реалистичная цель месяца: один route/bus shadow pilot и измеренные gates, не
  city-wide RTCI launch.

## 2. Dataset landscape

### 2.1 Наиболее релевантные

| Dataset | Статус | View / range / labels | Licence и доступ | Роль |
|---|---|---|---|---|
| PMOF | **Проверено, лучший transfer** | 19,696 вручную размеченных 1920x1920 frames, 31 recording, moving passenger vehicle, ceiling fisheye, 1-4 пассажира; rotated boxes, track IDs, actions | CC BY 4.0; доступ по запросу авторам | Viewpoint, motion, fisheye detector pretraining; не dense validation |
| Gorelik/BeIntelli | **Проверено, слабая crowd coverage** | 4 inward-facing RGB-D cameras + LiDAR, 9,136 synchronized published samples; local audit max 4 occupants; pseudo-labels | Zenodo API record 20559664: data CC BY 4.0; toolkit repo не имеет ясной software licence | Pipeline/domain transfer; не current full local dataset и не crush-load evidence |
| KAOHSIUNG-BUS | **Проверено в paper, недоступно** | Две ceiling cameras в реальных автобусах, day/night/glare; head/density tasks; published test cases >25 passengers | Paper называет базу proprietary; public repository/licence не найдены | Лучший scientific precedent; запросить data-use agreement |

Primary sources:

- PMOF: https://swermuth.github.io/pmof/ and https://arxiv.org/abs/2606.13910
- Gorelik: https://doi.org/10.5281/zenodo.20559664 and
  https://arxiv.org/abs/2606.11739
- KAOHSIUNG-BUS paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7218726/

### 2.2 Fisheye и dense transfer candidates

| Dataset | Проверенный смысл | Ограничение |
|---|---|---|
| WEPDTOF | 10,544 overhead fisheye frames, до 35 people, rotated boxes/tracks | Official page: non-commercial, form-gated |
| CEPDOF | 25,504 overhead fisheye classroom frames, до 13 people, low-light/IR sequences | Non-commercial, form-gated |
| HABBOF | 5,837 overhead fisheye frames, до 4 people | Non-commercial, низкая релевантность |
| LOAF | 42,942 frames, 45 scenes, dense fisheye, max at least 65, rotated boxes + physical locations | CC BY-NC-SA 4.0 / restrictive website terms; academic comparison only |
| THEODORE | 100,000 synthetic omnidirectional top-view images, boxes + masks | CC BY 4.0; synthetic indoor scenes, не bus validation |
| RPEE-HEADS | Локально: dense elevated views и low light, raw head boxes | CC BY-SA 4.0 и сильный domain gap |
| DISCO | Локально: dense/low-light crowd density maps | CC BY 4.0, outdoor plaza, не fisheye/bus |

Sources:

- WEPDTOF: https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/
- CEPDOF: https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof/
- HABBOF: https://vip.bu.edu/projects/vsns/cossy/datasets/habbof/
- LOAF: https://arxiv.org/abs/2307.08252
- THEODORE: https://www.tu-chemnitz.de/etit/dst/forschung/comp_vision/datasets/theodore/index.php.en
- RPEE-HEADS: https://doi.org/10.34735/ped.2024.2
- DISCO: https://zenodo.org/records/3828468

### 2.3 Не использовать как current answer

- PCDS: bus-door RGB-D APC, CC BY-NC-SA 3.0, Baidu-gated, другая задача.
- Berlin-APC: низкоразмерные sensor arrays для boarding/alighting, не
  RGB cabin occupancy.
- TADD bus cabin: anomaly footage без нужных occupancy labels; bus download
  не подтверждён доступным.
- BUS-HAR: staged actions, максимум два человека.
- Roboflow aggregations: не использовать без трассировки origin, consent и
  licence каждого изображения.

## 3. Собственный Almaty dataset

### 3.1 Сначала определить target

Рекомендуемый кандидат для whole-bus RTCI:

- сохранять raw `N_passengers`;
- operational score рассчитывать как `N_passengers / certified_capacity`;
- seat occupancy хранить отдельно;
- passenger-facing levels определять через вместимость и проверенное
  восприятие, а не через равные интервалы model output.

Если FOV не покрывает весь салон, нельзя обучать «total cabin count» по
невидимым людям. Нужно изменить геометрию, добавить камеру или честно сузить
target до visible-zone density.

### 3.2 Collection protocol

- Начать с одного route и одного bus type.
- Для каждой записи хранить vehicle, route/trip/stop, UTC + monotonic time,
  camera/lens, height/angle/calibration, day/night/weather и door state.
- Целенаправленно собирать empty, sparse, dense, night, glare, motion blur,
  seated/standing, children, winter clothing, bags, reflections и obstruction.
- Ground truth: onboard census после остановок + offline annotations.
- На выбранных frames размечать total count, head/person location, visibility,
  occlusion, seated/standing, cabin zone, ignore region и image-quality flags.
- Не размечать все соседние frames. Использовать occupancy transitions и
  stratified independent frames.
- Начальный planning range: 2k-5k independently useful labelled frames;
  double-label 10-20% и adjudicate disagreements.
- Split только по complete trips/days/buses. Adjacent frames не могут попасть
  в разные train/test sets.

### 3.3 Evaluation contract

- count MAE/RMSE и signed bias;
- exact / within-1 / within-2 count accuracy;
- five-level macro-F1, confusion matrix, weighted kappa;
- severe error rate `abs(predicted_level - true_level) >= 2`;
- confidence calibration;
- p95 end-to-end latency, RAM, power и valid-availability rate;
- отдельные срезы day/night, sparse/dense, seated/standing, glare/blur и bus.

Acceptance thresholds остаются TBD и фиксируются до просмотра final test.

## 4. Model comparison, не stack decision

### B0: minimum robust baseline

Single-class person/head detector с COCO initialization, calibrated cabin ROI
и 2-3 s temporal median/EMA. Сначала выдаёт raw count + confidence; пять уровней
являются отдельным calibrated mapping.

Implementation candidate: RTMDet-tiny / MMDetection, потому что OpenMMLab и
MMDeploy имеют Apache-2.0 stack и TensorRT/Jetson deployment path:

- https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet
- https://github.com/open-mmlab/mmdeploy

Наличие export path не является измеренной Sanas latency.

### C1: rotation-aware detector

RTMDet-R/MMRotate с rotated person/head boxes. Он согласуется с radial geometry
и PMOF annotations, но exact operators/export на Orin Nano нужно проверить
smoke test до выбора:

- https://github.com/open-mmlab/mmrotate

### C2: density estimator

CSRNet + PFCASA/PFCA сравнивается на том же Almaty split. Paper проверяет
ShanghaiTech, не bus/Jetson. Official implementation/licence и TensorRT path
для выбранного кода пока не подтверждены:

- https://arxiv.org/abs/2605.18349

Hybrid detector+density оправдан только если dense-band error улучшается
настолько, чтобы окупить compute, annotation и debugging complexity.

## 5. Prototype engineering architecture

```text
RGB camera(s)
  -> acquisition + timestamps
  -> image quality / privacy masks
  -> inference
  -> temporal filter + calibration
  -> score, level, confidence, quality/status
  -> local event log
  -> Avtobys adapter

Acquisition
  -> consented evidence sampler
  -> encrypted local NVMe
  -> controlled research export
```

Capture, inference, evidence recording и network publisher должны быть
отдельными supervised processes. Camera stale/covered/black/severely blurred
выдаёт `unknown`, а не последний известный уровень.

### Camera/FOV gate

- 160° у IMX219-160 является diagonal fisheye FoV; нельзя применять обычную
  rectilinear формулу покрытия.
- Измерить салон и passenger grid на seated/standing head heights.
- Снять одинаковые scenes для нескольких mounting positions и для one-vs-two
  camera setup.
- Калибровать каждый физический модуль через OpenCV fisheye model и сохранять
  `K`, `D`, images и reprojection error:
  https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
- Proposed gate: не менее 95% заранее заданных passenger-grid points пригодны
  для разметки, без critical zone, невидимой всеми камерами.

### Compute/recording risks

- Jetson Orin Nano Developer Kit является pre-production platform.
- Carrier-board operating ambient указан как 0-35 °C. Это не покрывает
  автоматически реальный автобус Алматы.
- У Orin Nano нет NVENC; continuous H.264 software encode конкурирует с
  inference за CPU/power. Предпочтительны sampled stills/event windows; если
  continuous video обязателен, отдельно тестировать hardware-compressed camera
  path.
- USB-C на dev kit предназначен для данных. DC input: 9-20 V через barrel.

Official sources:

- https://developer.nvidia.com/embedded/faq
- https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf
- https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/Multimedia/SoftwareEncodeInOrinNano.html

### Data message candidate

```text
schema_version, message_id, bus_id, route_id, trip_id,
frame_time_utc, valid_until_utc, sequence,
occupancy_score_0_1, occupancy_level_1_5, confidence,
quality_flags, status(valid|degraded|unknown),
model_version, calibration_version
```

Transport остаётся open. Обязательны TTL, deduplication, store-and-forward
telemetry и запрет показывать stale messages как current.

## 6. One-month go/no-go plan

### Week 1: specification + bench

- Freeze target/levels/metrics/schema.
- Bring up camera; measure frames, latency, power, thermal and recording load.
- Test power loss, disk full, network loss and camera restart.
- 24-hour bench run.

Proposed Gate 1: zero unexpected reboot, no thermal throttling, recovery works,
storage/power sizing measured. Failure means no bus installation.

### Week 2: stationary bus + FOV

- Permission, exact bus measurements and fisheye calibration.
- One-camera position matrix, day/backlight/evening volunteers.
- Compare dual RGB only if one camera fails coverage.

Gate 2: coverage and reproducible calibration pass. Otherwise change geometry
or declare the bus type unsupported.

### Week 3: moving shadow pilot

- One route/bus, RTCI hidden from passengers.
- Locked software/config.
- Manual ground truth on stratified trips.
- Vibration, network, sun, night, thermal and battery monitoring.

### Week 4: readiness, not automatic intervention

- Analyze sensor accuracy/availability and survey.
- Calculate field-experiment power from observed eligible decisions.
- Freeze/preregister protocol.
- Enable RTCI only if technical, ethics/privacy and causal-design gates pass.

The intervention must use the same locked firmware/model as shadow validation;
visibility switches server-side, not through a new device build.
