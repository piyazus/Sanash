# Sensing-modality trade study for bus passenger counting

Written 2026-08-10 (Donatello). Companion to `findings.md` (measured results)
and `../paper/related_work_notes.md` (coordinator's literature sweep).

Every number carries a verification tag:

- **[full-text]** — confirmed from the paper/page body read in this session
- **[abstract]** — inferred from an abstract or secondary summary, not yet
  verified against full text (Danyshpan must verify before citing)
- **[vendor]** — manufacturer claim, no independent measurement
- **not stated** — searched and not found; do not invent

Scite is out of monthly quota until 2026-09-01 (per related_work_notes.md),
so full-text verification beyond what WebFetch could read is deferred.

---

## 1. The acceptance bar: VDV 457

What it actually is: a recommendation ("VDV-Schrift 457", v2.1, 4/2018) of
the Verband Deutscher Verkehrsunternehmen (Association of German Transport
Companies) for automatic passenger counting systems. The governing document
is https://www.vdv.de/457-v2.1-ses.pdfx (located this session; PDF body not
parsed locally — no poppler — so the requirements below come from secondary
literature, tagged accordingly).

What it requires, as reported in the validation-methodology literature:

- Maximum systematic error (bias) of **1%**, i.e. "99% accuracy" refers to
  bias, not per-event hit rate. [abstract — reported consistently across
  Siebert & Ellenberger, arXiv:2104.09697, and its journal version
  (T. Transportmetrica A, DOI 10.1080/23249935.2023.2267702, HTTP 403 this
  session); also cited as "the +-1% requirement of VDV 457" by CLAPC
  (Seo et al. 2026, IEEE OJ-ITS) per related_work_notes.md]
- Since the 2018 revision, acceptance uses an **equivalence test** (replacing
  a t-test criterion) with user error bounded at 5%. [abstract, same sources]
- Practical consequence measured by that literature: demonstrating +-1% bias
  needs large stratified samples (thousands of door events). **Our 25-event
  validation set cannot demonstrate or refute VDV 457 compliance in
  principle** — it can only measure per-event detection with wide intervals.
- A separate criterion sometimes quoted ("97% of stops within +-2 passengers")
  is from North American agency specs, not VDV 457 — do not conflate.
  [Mass Transit, "Defining APC Accuracy Standards in North America"]

## 2. Modality table

Cost classes: $ (<$50/door in parts), $$ ($50-500), $$$ (>$500 commercial
unit + integration). Commercial APC unit prices are not published by any of
the vendors below (searched); $$$ is inferred from the market segment, not a
quote.

| Modality | Measured/claimed accuracy (source) | Cost class | Privacy | Low-light | Install | Failure modes |
|---|---|---|---|---|---|---|
| Commercial 3D ToF APC (iris IRMA MATRIX) | "99+ %" count accuracy [vendor: iris-sensing.com product page]; 500-pixel ToF matrix | $$$ | depth only, no RGB — strong | active IR — robust | per-door header mount, PoE | vendor-vs-field gap (see Pronello row); objects vs people (prams, bikes) |
| Commercial stereo APC (Hella Aglaia APS series) | ">98%" [vendor: APS-R-PoE, via datalinkinternational.com]; 110 deg scan angle claimed for one variant | $$$ | onboard processing, no video retention claimed [vendor] | passive stereo — degrades in darkness unless IR-assisted (vendor does not state) | per-door | same class as above |
| Commercial APC (DILAX) | "up to 99%" incl. strollers/bicycles/luggage; "latest generation up to 98%" [vendor: dilax.com] | $$$ | claims GDPR-conform [vendor] | not stated | per-door | same class |
| Commercial APC (Masats) | **not stated by vendor** (searched, nothing published) | $$$ | not stated | not stated | integrated in door mechanisms | unknown |
| Field reality check, video APC | Vendor claimed 98%; measured **53.17% boarding / 55.29% alighting** over 20 days of real bus service (Asti); authors' RPi+YOLOv5+DeepSORT: 72.27%/74.59% [full-text: Pronello & Garzon Ruiz 2023, Sensors 23(18):7719, PMC10537391] | $$ (RPi build) | RGB video — weak | poor | per-door camera | the headline row of this table: a 98% vendor claim measured at 53-55% in service |
| Research depth APC, trains | NAPC: LSTM on low-res 3D LiDAR over train doors, ~96% correct door-phase counts [abstract: Seidel et al. 2021, IEEE OJ-ITS]; CLAPC: CNN-LSTM on ToF above train doors, 99.79%/99.97% boarding/alighting, states meeting VDV 457 +-1% [abstract: Seo et al. 2026, IEEE OJ-ITS] | $$-$$$ | depth only — strong | active sensing — robust | top-down over door | learned models need labelled door data; train doorways, not bus vestibules |
| Research stereo APC, bus doors | 99%/97% on two datasets [abstract: Yahiaoui et al. 2010, J. Electronic Imaging] | $$ | depth | passive stereo | top-down over door | 2010-era hardware |
| Single-chip multizone ToF (ST VL53L5CX/L7CX/L8CX) | No published transit-door accuracy for this exact chip class (searched). Closest: ToF camera above doorway >90% single-entry/exit [abstract: Jeong et al. 2025, ETRI J.]; office low-res ToF zone counting ~0.4% error rate [abstract: Lu et al. 2021, Energy & Buildings]. Specs: 8x8 zones, L5CX 65 deg diag / L7CX 90 deg diag (60x60 deg square) FoV, range to 350-400 cm, **15 Hz max at 8x8, 60 Hz at 4x4** [full-text: ST datasheets/product pages] | $ (bare chip low single-digit USD at volume — distributor unit price NOT confirmed this session; Pololu L8CX carrier board $19.95 [vendor listing]) | absolute — 64 range values, no image | active IR — robust; IR-sunlight interference at door is the open question | tiny module, I2C/SPI, one per door lintel | resolution floor (our ablation tests exactly this); sunlight at open door; multi-person merging |
| mmWave radar 60 GHz (TI IWR6843 class) | Indoor people counting 83.5-98.9% across papers [abstract: MDPI Sensors 26(4):1289 (98.86%, 0-3 people); mmCounter arXiv:2512.10357: 87% F1 familiar / 60% F1 unseen environments]; TI ships in-cabin occupancy reference designs, **no bus-door APC accuracy found** | $$ | absolute — point cloud/doppler | immune to light | flexible mounting, needs RF-transparent cover | static (non-moving) people are hard; multipath in a metal cabin; per-environment tuning |
| CO2 concentration | Occupied/vacant 95.8%, count ~80.6% in rooms [abstract: CD-HOC, arXiv:1706.05286]; response lag ~7 min measured to occupancy change [abstract: AIVC dynamic-NN paper] | $ | absolute | immune | one sensor per cabin | **lag kills it**: stop-to-stop dynamics are 20-60 s, ventilation/doors dominate the signal; aggregate only, no boarding/alighting split |
| WiFi/BLE probe counting | With MAC de-randomization ~97% in controlled settings but needing **>= 22 min observation windows** [abstract: unsupervised-clustering APC, PMC10256033 area]; naive probe counting defeated by randomization [abstract: multiple, e.g. IEEE TMC 2022 de-randomization literature] | $ | poor in perception (tracks devices), even if hashed | immune | one scanner per vehicle | MAC randomization; device-ownership bias (children, feature phones, off devices); window length incompatible with per-stop counts |
| Thermal arrays (MLX90640 32x24, AMG8833 8x8) | Doorway line-crossing counters ~93-95%+ in controlled indoor settings [abstract: Sensors 21(12):4062; Energies 14(15):4542] | $ | near-absolute (32x24 thermal silhouettes) | immune to visible light | door lintel | ambient temperature approaching body temperature (summer bus interior 30-37 C) collapses contrast; sun-heated surfaces; low frame rates at low noise |
| IR beam-break pairs | "up to 80%" foot-traffic accuracy; 60-75% with direct sunlight; side-by-side crossings merge [industry sources: flowcounters.com, doorcountersystems.com — treat as informed vendor commentary, no peer-reviewed figure found] | $ | absolute | IR-based, sunlight-sensitive | door frame, trivial | simultaneous crossings; wide doors need multiple beams; still the historical baseline in transit |

## 3. Recommendation

Door depth-APC as Phase 1, with the multizone-ToF variant as the cost/privacy
end point, and the Phase-2 RGB ordinal head as drift correction. Grounds,
tied to measurements in `findings.md` and `experiments/log.md`:

1. **Cabin-wide depth occupancy measurably fails** on the only available
   in-cabin data: 10.0% floor coverage in-spec, Pearson r 0.184
   [0.133, 0.231], non-monotonic above two occupants, and even a 4-camera
   union leaves most occupied floor unseen (28.8% covered at spec range).
   Counting at the chokepoint (door) instead of sensing the whole volume is
   what the commercial APC industry converged on, and our own negative
   result independently pushes the same way.
2. **Depth at the door is the strongest published modality.** Train-door
   depth counters report the only numbers in VDV 457 territory (CLAPC
   99.79/99.97 [abstract]); commercial 3D APC vendors all claim 98-99%
   [vendor]. The Pronello field study is the caution: claims and field
   performance can differ by 40+ points, so nothing we ship should quote a
   number that was not measured in service.
3. **The multizone ToF variant is worth the ablation.** It is the only
   modality in the table that is simultaneously ~$ cost, privacy-absolute
   (64 range values cannot re-identify a face), and active-IR dark-robust.
   Published building/doorway results support >90% [abstract] but nobody has
   published a bus-door study at the 8x8/4x4 resolution floor — that gap is
   our paper's claim #1 (see related_work_notes.md). The VL53L7CX's 15 Hz
   cap at 8x8 also justifies validating the pipeline at reduced frame rates.
4. **What depth/ToF cannot do** — recover from missed events between
   terminus resets — is exactly what the dormant Phase-2 RGB ordinal head
   (DINOv2/ConvNeXt + CORN) is kept for: periodic absolute crowding
   estimates to re-anchor the cumulative count. CO2, WiFi/BLE and thermal
   fail as primaries on lag, randomization and summer-temperature contrast
   respectively, and beam-break is dominated by every depth option.
5. **Acceptance framing**: nothing validated on 25 staged episodes speaks to
   VDV 457. The standard's own methodology literature puts required samples
   in the thousands of events; a compliance claim needs a revenue-service
   pilot with manual reference counts.

## Sources

- VDV 457 v2.1 PDF: https://www.vdv.de/457-v2.1-ses.pdfx
- Equivalence-test methodology: https://arxiv.org/abs/2104.09697 ;
  https://www.tandfonline.com/doi/full/10.1080/23249935.2023.2267702
- iris IRMA MATRIX: https://www.iris-sensing.com/us/products/irma-matrix/
- Hella APS: https://www.datalinkinternational.com/Applications/APC/ ;
  https://www.emamidesign.de/en/news/detail/people-counter-aps-90.html
- DILAX: https://www.dilax.com/en/products/automatic-passenger-counting
- Pronello & Garzon Ruiz 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10537391/
- NAPC / CLAPC / Jeong / Lu / Stec: consensus.app URLs in
  ../paper/related_work_notes.md (all [abstract])
- VL53L5CX datasheet: https://www.st.com/resource/en/datasheet/vl53l5cx.pdf
- VL53L7CX product page: https://www.st.com/en/imaging-and-photonics-solutions/vl53l7cx.html
- VL53L8CX carrier: https://www.pololu.com/product/3419
- mmWave: https://www.mdpi.com/1424-8220/26/4/1289 ;
  https://arxiv.org/abs/2512.10357 ; https://www.ti.com/product/IWR6843
- CO2: https://arxiv.org/pdf/1706.05286 ;
  https://www.aivc.org/sites/default/files/101.1367385309.full_.pdf
- WiFi probes: https://pmc.ncbi.nlm.nih.gov/articles/PMC10256033/ ;
  https://dl.acm.org/doi/10.1109/TMC.2022.3205924
- Thermal: https://doi.org/10.3390/s21124062 ;
  https://www.mdpi.com/1996-1073/14/15/4542
- Beam-break: https://www.flowcounters.com/blogs-detail/ai-stereo-vision-footfall-counter-vs-infrared-counter-what-actually-works-where ;
  https://doorcountersystems.com/guides/how-infrared-counters-work
- North American per-stop criterion (NOT VDV 457):
  https://www.masstransitmag.com/technology/article/12128610/defining-apc-accuracy-standards-in-north-america
