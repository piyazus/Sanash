---
type: source
kind: batch
citation: Батч литературы по CV и сенсорам, отобран refgraph 2026-08-26
doi:
local: research/refs/cv-hardware-corpus/ (PDF на диске, в git не отслеживаются)
verified: extracted-text
tracks: [cv-device]
ingested: 2026-08-26
---

# Корпус CV и сенсоров, батч 2026-08-26

44 PDF. Вердикты и причины отбора: `research/refs/REFGRAPH_REPORT.md`.
Раздел `Verification pass 2026-08-26` там же содержит повторную сверку: у всех
44 файлов заголовок, авторы и предмет совпали с причиной вердикта.

**Что проверено:** заголовок, авторы, площадка, предмет. Извлечение pypdf,
первые страницы. **Что не проверено:** методы, выборки, метрики внутри статей.
Для цитирования в тексте статьи нужна полнотекстовая проверка.

**Трек:** это литература устройства (CV и железо), а не RTCI-поведение.
Пересечение с RTCI-треком в батче было: три работы уже лежат отдельно в
`research/refs/rtci-supporting/`.

## Транспорт, подсчёт и заполненность салона

Ближайшие к задаче Sanas работы.

| Файл | Работа |
|---|---|
| `zhao2021.pdf` | Zhao, Lei, Li, Zhao, Han, Hou. Detection of crowdedness in bus compartments based on ResNet algorithm and video images. Multimedia Tools and Applications, 2021 |
| `sensors-20-02178 (1).pdf` | Hsu, Chen, Perng. Estimation of the number of passengers in a bus using deep learning. Sensors, 2020 |
| `2606.11739v1.pdf` | Gorelik, Karrow, Sivrikaya et al. Multi-view in-cabin monitoring system for public transport vehicles. arXiv:2606.11739 |
| `2605.18349v1.pdf` | Rostamza, Del Re, Varughese, Olaverri-Monreal. Optimising CSRNet with parameter-free attention mechanisms for crowd counting in public transport. arXiv:2605.18349 |
| `Training_a_Regression-Based_Model...Ranked_Image_Pairs_and_Triplets.pdf` | Lee, Lee, Kim et al. Training a regression-based model for crowd counting in transit cars using ranked image pairs and triplets. IEEE Access, 2024 |
| `2403.20173v1.pdf` | Guo, Zhang, Zhao. MCNet: a crowd density estimation network based on integrating multi-scale attention module. arXiv:2403.20173 |
| `jimaging-06-00062.pdf` | Mazzeo, Contino, Spagnolo, Distante, Stella, Nitti, Reno. MH-MetroNet: a multi-head CNN for passenger-crowd attendance estimation. Journal of Imaging, 2020 |
| `DICTA.2018.8615794.pdf` | Goh, Chua, Lim, Atmosukarto. Image analytics for train crowd estimation. DICTA 2018 |
| `2508.03749v1.pdf` | Fiorista, Abdelhalim, Pincus, Thistle, Zhao et al. Closed-circuit television data as an emergent data source in urban rail platform crowding estimation. arXiv:2508.03749 |
| `j.knosys.2017.02.016.pdf` | Liu, Yin, Jia, Xie. Passenger flow estimation based on convolutional neural network in public transportation system. Knowledge-Based Systems, 2017 |
| `CLAPC_A_Hybrid_CNN-LSTM...Public_Transport.pdf` | Seo et al. CLAPC: hybrid CNN-LSTM architecture for automated passenger counting from video streams in public transport. IEEE Open Journal of ITS, 2026 |
| `meghana2020.pdf` | Meghana, Sarode, Tambade, Marathe, Charniya. Automated crowd management in bus transport service. 2020 |
| `1-s2.0-S0166361524001234-main.pdf` | McCarthy, Ghaderi, Marti, Jayaraman, Dia. Video-based automatic people counting for public transport: on-bus versus off-bus deployment. Computers in Industry, 2024 |
| `1804.04339v2.pdf` | Sun, Akhtar, Song, Zhang, Li, Mian. Benchmark data and method for real-time people counting in cluttered scenes using depth sensors. Датасет PCDS, более 4500 видео с дверей автобусов. arXiv:1804.04339 |
| `1.3455989.pdf` | Yahiaoui, Khoudour et al. Real-time passenger counting in buses using dense stereovision. Journal of Electronic Imaging, 2010 |
| `soli.2018.8476774.pdf` | Tan, Atmosukarto, Lim. Video analytics for indoor crowd estimation. 2018 |

## Модели crowd counting общего назначения

Фон для выбора архитектуры, не транспортные данные.

| Файл | Работа |
|---|---|
| `1802.10062v4.pdf` | Li, Zhang, Chen. CSRNet: dilated CNNs for understanding the highly congested scenes. arXiv:1802.10062 |
| `2111.08851v5.pdf` | Shi, Cao, Raschka. Deep neural networks for rank-consistent ordinal regression based on conditional probabilities (CORN). arXiv:2111.08851 |
| `2304.07193v2.pdf` | Oquab, Darcet, Moutakanni et al. DINOv2: learning robust visual features without supervision. TMLR, 2024 |
| `2210.10392v4.pdf` | Zhang, Choi, Hong. Spatio-channel attention blocks for cross-modal crowd counting. arXiv:2210.10392 |
| `j.neucom.2019.08.018.pdf` | Gao, Wang, Yuan. SCAR: spatial/channel-wise attention regression networks for crowd counting. Neurocomputing 363, 2019 |
| `chen2020.pdf` | Chen, Su, Wang. Crowd counting with crowd attention convolutional neural network. Neurocomputing 382, 2020 |
| `wang2022.pdf` | Wang, Breckon. Crowd counting via segmentation guided attention networks and curriculum loss. IEEE Transactions on ITS, 2022 |
| `ding2021.pdf` | Ding, He, Lin, Wang et al. Crowd density estimation using fusion of multi-layer features. IEEE Transactions on ITS |
| `DTCC_Multi-level_dilated_convolution_with_transformer...pdf` | Miao, Zhang, Peng, Peng, Yin. DTCC: multi-level dilated convolution with transformer for weakly-supervised crowd counting. Computational Visual Media, 2023 |
| `jimaging-06-00028-v2.pdf` | Sooksatra, Kondo, Bunnun, Yoshitaka. Redesigned skip-network for crowd counting with dilated convolution and backward connection. Journal of Imaging, 2020 |
| `sensors-24-01816.pdf` | Zhao, Ma, Jia, Wang, Hei. A dilated CNN for cross-layers of contextual information for congested crowd counting. Sensors 24:1816, 2024 |
| `Single_Convolutional_Neural_Network_With_Three_Layers...pdf` | Alashban, Alsadan et al. Single CNN with three layers model for crowd density estimation. IEEE Access, 2022 |

## Не-RGB сенсоры

| Файл | Работа |
|---|---|
| `24_iris_IRMA_Matrix_en.pdf` | iris. IRMA MATRIX, ToF-сенсор подсчёта, короткий datasheet |
| `dasip48288.2019.9049169.pdf` | Stec, Herrmann, Stabernack. Using time-of-flight sensors for people counting applications. DASIP 2019 |
| `sensors-26-01639.pdf` | On the characterisation of the time-of-flight VL53L5CX sensor by STMicroelectronics. Sensors, 2026 |
| `ETRI Journal - 2025 - Jeong - Privacy-preserving labeling-free occupancy counting...pdf` | Jeong, Park. Privacy-preserving labeling-free occupancy counting sensor based on ToF camera and clustering. ETRI Journal, 2025 |
| `A_Lightweight_Real-Time_Human_Detection...Low-Resolution_Thermal_Sensors.pdf` | Lightweight real-time human detection and tracking for privacy-preserving occupancy monitoring using low-resolution thermal sensors. IEEE Access, 2026 |

## Edge и встраиваемое исполнение

| Файл | Работа |
|---|---|
| `computers-14-00476.pdf` | Diaz-Santos, Caballero-Gil, Caballero-Gil. Real-time passenger flow analysis in tram stations using YOLO-based computer vision and edge AI on Jetson Nano. Computers, 2025 |
| `sensors-25-01695.pdf` | Sanchez Laguna, Davalos Guzman, Aguilar Lobo. Edge computing based on CNN for passenger counting: case study in Guadalajara, Mexico. Sensors, 2025 |
| `TSP_CMC_35974.pdf` | Gu, Wu, Wang, Chen, Yan. A deep learning-based crowd counting method and system implementation on neural processing unit platform. CMC, 2023 |

`2403.20173v1.pdf` (MCNet) также содержит измерения энергопотребления и
скорости на embedded-устройстве, но основной его вклад модельный.

## Валидация и стандарты

| Файл | Работа |
|---|---|
| `457-v2.1-ses.pdf` | VDV Recommendation 457, version 2.1, 2018. Automatic passenger counting systems, 150 страниц |
| `2104.09697v3.pdf` | Ellenberger, Siebert. Introducing the partitioned equivalence test: AI in APC validation. arXiv:2104.09697 |
| `Artificial intelligence in automatic passenger counting...pdf` | Та же работа Ellenberger и Siebert, опубликованная версия, Transportmetrica A |
| `sensors-23-07719.pdf` | Pronello, Garzon Ruiz. Evaluating the performance of video-based automated passenger counting systems in real-world conditions. Sensors 23:7719, 2023 |

## Обзоры

| Файл | Работа |
|---|---|
| `2009.12619v6.pdf` | Darsena, Gelli, Iudice, Verde. Sensing technologies for crowd management, adaptation and information dissemination in public transportation systems: a review. IEEE Sensors Journal |
| `sustainability-15-01332.pdf` | Kuchar, Pirnik, Janota, Malobicky, Kubik, Sismisova. Passenger occupancy estimation in vehicles: a review of current methods and research challenges. Sustainability 15:1332, 2023 |
| `eng-05-00172.pdf` | Radovan, Mrsic, Dambic, Mihaljevic. A review of passenger counting in public transport concepts with solution proposal based on image processing and machine learning. Eng, 2024 |
| `BDCC-05-00050.pdf` | Gouiaa, Akhloufi, Shahbazi. Advances in CNN based crowd counting and density estimation. Big Data and Cognitive Computing |

## Что из этого затрагивает открытые решения

- [[occupancy-sensing-methods]]
- [[edge-inference-constraints]]
- [[apc-validation-and-standards]]
