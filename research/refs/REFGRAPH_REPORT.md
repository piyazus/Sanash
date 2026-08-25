# Refgraph Report — batch 2026-08-26

56 PDFs dropped in project root, staged to `tmp/refgraph_batch_2026-08-26/`
(not committed — tmp/ is gitignored), extracted via
`.claude/skills/refgraph/scripts/extract_pdf_metadata.py`, classified against
`GROUND_TRUTH.md` (ceiling-RGB CV device track: CSRNet+PFCASA candidate,
Jetson/IMX219 hardware candidates, APC sensor comparators) and
`research/RTCI_RESEARCH_CHARTER.md` (RTCI causal behavioral track).

This batch is CV/hardware literature for the device track, not RTCI
behavioral literature — different from the 12 seed papers in
`rtci_paper_inventory.csv`. Three items overlap both tracks (Drabicki 2023,
Pi 2018, Zhang-Kennedy 2023) and are already filed under
`research/refs/rtci-supporting/`.

## Verdicts

| File | Verdict | Reason | Status |
|---|---|---|---|
| 1-s2.0-S0166361524001234-main.pdf | keep | McCarthy et al., video APC on-bus vs off-bus field trials — directly relevant to camera placement decision | verified from extracted text |
| 1-s2.0-S2210539523000196-main.pdf | **duplicate** | Same paper as `research/refs/rtci-supporting/drabicki2023_rtbm.pdf` (Drabicki 2023, RTCI WTW) | verified from extracted text |
| 1.3455989.pdf | keep | Real-time bus passenger counting via stereovision (99%/97% accuracy) — APC sensor comparator | verified from extracted text |
| 1706.05286v1 (2).pdf | **duplicate** | Byte-identical to 1706.05286v1.pdf | verified (identical size, identical text) |
| 1706.05286v1.pdf | reject | CO2-based indoor occupancy counting (building HVAC context) — different sensing modality, not camera/CV, not transit | verified from extracted text |
| 1802.10062v4.pdf | keep | CSRNet paper — the exact model candidate named in GROUND_TRUTH.md | verified from extracted text |
| 1804.04339v2.pdf | keep | RGB-D people counting at bus doors, PCDS dataset — closest sensor-modality comparator | verified from extracted text |
| 2009.12619v6.pdf | keep | The exact review paper (arXiv:2009.12619) the ResearchRabbit session couldn't find — now recovered manually | verified from extracted text |
| 2104.09697v3.pdf | keep (pair with dup below) | Ellenberger/Siebert partitioned equivalence test for APC validation, arXiv preprint version | verified from extracted text |
| 2111.08851v5.pdf | keep | CORN ordinal regression paper — named as historical candidate in GROUND_TRUTH.md line 123 | verified from extracted text |
| 2210.10392v4.pdf | keep | Cross-modal (RGB-T/RGB-D) crowd counting attention blocks — relevant to day/night camera choice open question | verified from extracted text |
| 2304.07193v2.pdf | borderline / keep | DINOv2 — named as historical candidate (DINOv2+CORN) in GROUND_TRUTH.md line 123, not current stack | verified from extracted text |
| 2403.20173v1 (1).pdf | **duplicate** | Byte-identical to 2403.20173v1.pdf | verified (identical size, identical text) |
| 2403.20173v1.pdf | keep | MCNet, metro crowd density estimation, embedded deployment — relevant to Jetson-class hardware constraint | verified from extracted text |
| 24_iris_IRMA_Matrix_en.pdf | keep | Commercial ToF APC sensor spec sheet — hardware comparator, alternative to camera-based approach | verified from extracted text |
| 2508.03749v1.pdf | keep | CCTV-based rail platform crowding (WMATA), compares YOLOv11/RT-DETRv2/APGCC/Crowd-ViT/DeepLabV3 — recent, directly relevant model comparison | verified from extracted text |
| 2512.10357v1.pdf | reject | mmWave radar static people counting, indoor dense scenario — different sensing modality, not camera-based, not transit-specific | verified from extracted text |
| 2605.18349v1.pdf | keep | PFCASA paper — cited directly in GROUND_TRUTH.md line 292 as the candidate head on CSRNet | verified from extracted text |
| 2606.11739v1.pdf | keep | Gorelik et al., multi-view in-cabin monitoring — matches the Gorelik/BeIntelli dataset already referenced in GROUND_TRUTH.md line 164 | verified from extracted text |
| 3544548.3581241.pdf | **duplicate** | Same paper as `research/refs/rtci-supporting/zhangkennedy2023_chi.pdf` (Zhang-Kennedy 2023, CHI) | verified from extracted text |
| 457-v2.1-ses.pdf | keep | VDV 457 — European standard for APC data exchange format, relevant to any future data-interchange decision with Avtobys | verified from extracted text |
| A_Lightweight_Real-Time_Human_Detection...Thermal_Sensors.pdf | keep | Low-resolution thermal occupancy sensing — relevant to low-light/night alternative discussed as open question | verified from extracted text |
| Artificial intelligence...partitioned equivalence test (1).pdf | **duplicate** | Byte-identical to the non-suffixed file below | verified (identical size, identical text) |
| Artificial intelligence...partitioned equivalence test.pdf | keep | Journal-published version of the same Ellenberger/Siebert method as 2104.09697v3.pdf — keep one, treat other as its published counterpart, not separately citable as new | verified from extracted text |
| BDCC-05-00050.pdf | keep | Survey: CNN-based crowd counting and density estimation methods — general model-family background | verified from extracted text |
| chen2020.pdf | keep | Crowd attention CNN for crowd counting — general model background | verified from extracted text |
| CLAPC_...Hybrid_CNN-LSTM...Passenger_Counting...pdf | keep | Hybrid CNN-LSTM specifically for video-based APC in public transport — directly on-topic | verified from extracted text |
| computers-14-00476.pdf | keep | YOLO-based passenger flow on Jetson Nano (edge AI) — directly relevant to Jetson hardware feasibility question | verified from extracted text |
| CVPR.2018.00120.pdf | **duplicate** | Same paper as 1802.10062v4.pdf (CSRNet, published CVPR version vs arXiv preprint) — keep one copy | verified from extracted text |
| dasip48288.2019.9049169.pdf | keep | ToF sensors for people counting — hardware comparator relevant to VL53L7CX-class sensor decisions | verified from extracted text |
| DICTA.2018.8615794.pdf | keep | Image analytics for train crowd estimation — transit-specific comparator | verified from extracted text |
| ding2021.pdf | keep | Crowd density estimation via multi-layer feature fusion — general model background | verified from extracted text |
| DTCC_Multi-level_dilated_convolution...pdf | keep | Weakly-supervised crowd counting with transformer — relevant given own-data labeling cost is an open question | verified from extracted text |
| eng-05-00172.pdf | keep | Review: passenger counting concepts, image processing + ML — directly on-topic survey | verified from extracted text |
| ETRI Journal...Jeong...ToF camera and clustering.pdf | keep | Privacy-preserving labeling-free occupancy counting via ToF + clustering — relevant to privacy requirement in GROUND_TRUTH RTCI gates | verified from extracted text |
| IRMA-MATRIX_R2_ProductDataSheet_4-1_en.pdf | **duplicate (near)** | Same commercial product as 24_iris_IRMA_Matrix_en.pdf, longer datasheet version — keep the more complete one, treat short version as redundant | verified from extracted text |
| j.knosys.2017.02.016.pdf | keep | Passenger flow estimation via CNN in public transport — directly on-topic | verified from extracted text |
| j.neucom.2019.02.071.pdf | **reject — filename/content mismatch** | Filename suggests Neurocomputing/passenger counting; actual content is an astrophysics paper on cosmic microwave background dipole asymmetry and axion monodromy cosmic strings (Physics of the Dark Universe, 2019). Completely unrelated to Sanas. | verified from extracted text, confirmed with direct re-extraction |
| j.neucom.2019.08.018.pdf | keep | SCAR: spatial/channel attention regression for crowd counting — general model background | verified from extracted text |
| jimaging-06-00028-v2.pdf | keep | Redesigned skip-network, dilated convolution, crowd counting — general model background | verified from extracted text |
| jimaging-06-00062.pdf | keep | MH-MetroNet, multi-head CNN for passenger-crowd attendance — transit-specific | verified from extracted text |
| meghana2020.pdf | keep | Automated crowd management in bus transport service — directly on-topic | verified from extracted text |
| pi2018.pdf | **duplicate** | Same paper as `research/refs/rtci-supporting/pi2018_perception.pdf` (Pi et al. 2018, bus fullness perception) | verified from extracted text |
| sensors-20-02178 (1).pdf | keep | Estimation of passengers in a bus using deep learning — directly on-topic, no non-suffixed twin found in this batch (orphaned download suffix, not a true duplicate) | verified from extracted text |
| sensors-23-07719.pdf | keep | Evaluating video-based APC systems in real-world conditions, comparative study — directly relevant to validation methodology | verified from extracted text |
| sensors-24-01816.pdf | keep | Dilated CNN cross-layer context for congested crowd counting — general model background | verified from extracted text |
| sensors-25-01695.pdf | keep | Edge-computing CNN passenger counting, case study Guadalajara — directly relevant edge-deployment comparator | verified from extracted text |
| sensors-26-01639.pdf | keep | Characterization of VL53L5CX ToF sensor — same sensor family as the cancelled VL53L7CX door-APC design; relevant background even though that architecture is cancelled, useful for sensor-comparison context | verified from extracted text |
| Single_Convolutional_Neural_Network...pdf | keep | Single CNN three-layer model for crowd density — general model background | verified from extracted text |
| soli.2018.8476774.pdf | keep | Video analytics for indoor crowd estimation — general background | verified from extracted text |
| sustainability-15-01332.pdf | keep | Review: passenger occupancy estimation methods and research challenges — directly on-topic survey | verified from extracted text |
| TITS.2020.2983475.pdf | **reject — filename/content mismatch** | Filename pattern suggests IEEE Trans. Intelligent Transportation Systems; actual content is a software-engineering education paper on method chains and code comments in readability experiments. Completely unrelated to Sanas. | verified from extracted text, confirmed with direct re-extraction |
| Training_a_Regression-Based_Model...Ranked_Image_Pairs...pdf | keep | Regression-based crowd counting in transit cars using ranked image pairs/triplets — directly on-topic | verified from extracted text |
| TSP_CMC_35974.pdf | keep | Deep-learning crowd counting on NPU (neural processing unit) platform — relevant to edge hardware constraint | verified from extracted text |
| wang2022.pdf | keep | Crowd counting via segmentation-guided attention + curriculum loss — general model background | verified from extracted text |
| zhao2021.pdf | keep | Crowd counting method (needs full-text check, title extracted as garbled filename artifact) — provisional keep pending confirmation | metadata only, title unreliable |

## Summary

- **Total processed:** 56
- **Keep:** 42
- **Reject (off-topic sensing modality):** 2 (CO2 occupancy, mmWave radar)
- **Reject (filename/content mismatch, wrong paper entirely):** 2 (cosmology paper, software-readability paper)
- **True duplicates (byte-identical or same published work):** 8 pairs/instances — `1706.05286v1 (2).pdf`, `2403.20173v1 (1).pdf`, `Artificial intelligence...(1).pdf`, `1-s2.0-S2210539523000196-main.pdf` (dup of rtci-supporting), `pi2018.pdf` (dup of rtci-supporting), `3544548.3581241.pdf` (dup of rtci-supporting), `CVPR.2018.00120.pdf` (dup of 1802.10062v4.pdf, arXiv vs published CSRNet), `IRMA-MATRIX_R2_ProductDataSheet_4-1_en.pdf` (near-dup of 24_iris_IRMA_Matrix_en.pdf)

## Not deleted yet

Nothing has been deleted. This report is the list for review before any
`rm`. All 56 files currently sit in `tmp/refgraph_batch_2026-08-26/`
(gitignored, not committed).
