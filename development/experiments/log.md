# Experiment Log (append-only, do not edit past entries)

<!--
Entry template:

## YYYY-MM-DD HH:MM — <short description>
- commit: <git hash>
- kaggle kernel: <slug>
- config: <model/backbone, epochs, batch size, dataset subset>
- result: <key metric(s)>
- notes: <what changed, what to try next>
-->

## 2026-08-09 — MODEL + DATA DECISION (nothing trained yet)

**STATUS: NOTHING HAS BEEN RUN. This entry records a decision and its
reasoning, not a result. No kernel has been pushed, no GPU quota spent, no
epoch trained. There are no metrics in this entry because there are none.**

- commit: 8db774b760ecd37227b37f29a2495167cc661bda
- kaggle kernels: written but NOT pushed —
  `diypyzsdiyas/sanas-extract-subset` (CPU),
  `diypyzsdiyas/sanas-depth-occupancy` (CPU),
  `diypyzsdiyas/sanas-train-corn-smoke` (GPU, awaiting approval)
- result: none

### Decision

Branch 1 (RGB, trained): frozen **DINOv2 ViT-S/14 + CORN ordinal head**, with
**ConvNeXt-Tiny + the same head** as the edge-deployable comparison, trained on
identical splits.
Branch 2 (depth, not trained): **geometric BEV occupancy from the RealSense
depth stream**, background-subtracted against empty-cabin frames.
Target for the smoke test: **raw count 0-4**. No 0-1 normalisation and no
5-level mapping until real crowding data exists; the switch is a config change
(`TargetConfig`), not a rewrite.

### Candidates considered

| candidate | verified evidence | outcome |
|---|---|---|
| CSRNet-style density regression | 16.26M params; SHA MAE 68.2/MSE 115.0, SHB MAE 10.6/MSE 16.0 (Table 5, arXiv 1802.10062) | Deferred. Right shape for a crowded cabin, but this dataset has no crowding to regress. Reference impl `leeyeehoo/CSRNet-pytorch` has **no licence file at all**, so it would need reimplementation from the paper. |
| P2PNet point-based | SHA MAE 52.74/MSE 85.06, SHB MAE 6.25/MSE 9.9 (Table 2, arXiv 2107.12746). Paper states no FPS or param count. | **REJECTED FOR THE PRODUCT PATH ON LICENCE.** `TencentYoutuResearch/CrowdCounting-P2PNet` LICENSE: *"Use in source and binary forms shall only be for the purpose of academic research."* Sanas ships inside Avtobys (Innoforce), which is not academic research. Best accuracy of any candidate and still unusable. Note this contradicts the "Стек: P2PNet" line in the global CLAUDE.md — the product stack has to change. Fine for the paper track only. |
| DINOv2 ViT-S/14 + CORN | 21M params, ImageNet linear 81.1% (repo README). Licence verbatim: *"DINOv2 code and model weights are released under the Apache License 2.0."* (the FAIR Noncommercial clauses in that repo cover the X-ray/cell variants, not DINOv2.) | **CHOSEN** for run 1. Frozen backbone means only the head trains, so the smoke test is minutes of GPU. |
| ConvNeXt-Tiny + CORN | `facebookresearch/ConvNeXt` MIT; ConvNeXt-V2 LICENSE file is also MIT text; timm Apache-2.0. | **CHOSEN** as the edge comparison. Plain convolutions export to ONNX/TensorRT far more predictably than ViT attention. |
| Ultralytics YOLO | AGPL-3.0 (GitHub API). | Not pursued. AGPL is a poor fit for a shipped product. RT-DETR and mmdetection are Apache-2.0 if detection-and-count is ever wanted. |

CORN loss is implemented from the formulation (arXiv 2111.08851), not vendored,
so the repo does not inherit a third-party licence for a few lines of
arithmetic. Reference impl `coral-pytorch` is MIT if a swap is ever wanted.

### Dataset limitations found (these drive the decision)

Verified by reading the ZIP central directory over HTTP range requests and
decoding samples. The 73.5 GB archive was never downloaded.

1. **Maximum 4 occupants, ever.** Per-frame distribution from
   `person_states.json` (9,975 frames): `{1: 4989, 2: 4418, 3: 546, 4: 22}`,
   26 unique person IDs across the whole recording, max 6 masks in any single
   camera view. `bboxes_3d` has 898 empty-cabin frames. The action vocabulary
   (`mugging`, `vandalism`, `littering`, `smoke`, `drink (alcoholic)`) shows
   this is a staged behavioural/safety recording with actors, not a crowding
   dataset. **Occupancy levels 3-5 of the intended 5-level scale have zero
   training examples.**
2. **Single 32-minute daytime session**, 2026-05-21 09:10:44-09:43:04 UTC,
   11 sub-sequences (10 gaps > 5 s). **Low-light robustness cannot be
   evaluated on this dataset at all.**
3. **Not nuScenes on disk.** Custom ROS-derived layout; nuScenes is an export
   target via the toolkit's `export_nuscenes.py`. Loaders must not assume it.
4. **Ground truth is pseudo-labels.** `segmentations/*.json` carry a `score`
   field and the toolkit ships `scripts/autolabeling/`. Not human annotation.
5. **Depth frames are PNG named `*.jpg`** (magic `89504e47`). Decode by
   content. Handled in `sniff_image_ext`.
6. **Depth intrinsics differ from colour**: depth is 848x480 fx~422
   (640x480 for back_right), colour is 1280x720 fx~920. Using colour
   intrinsics on depth would be wrong by >2x.
7. **Paper/archive count mismatch**: arXiv 2606.11739 says 9,136 synchronised
   samples; the archive has 9,975 `states/` and 10,034 `bboxes_3d`. Unresolved.
8. **Toolkit repo `EvgenyGorelik/multiview_incabin_dataset` has no licence
   file.** Data is CC-BY-4.0 and calibration values are read from the
   archive's own YAML/JSON, but none of their code is copied. Deprojection is
   written from the sensor model.

### Branch 2 rationale (depth instead of lidar)

Checked the paper: lidar is used for ICP calibration, qualitative mesh
validation and as a BEVFusion input — it is **not** their label source, labels
come from cameras. So "repeat their method" does not apply. The deployable
version of the idea is stereo depth, not an OS0-128 in every bus: it is metric,
already in this dataset, and IR stereo works in darkness, which covers the
low-light gap above.

Extrinsics come from the archive itself (`frame_ids.json` maps camera ->
optical frame, `target_transforms.json` gives the URDF chain to `base_link`),
so no unlicensed toolkit code is needed. Verified: front_left_depth resolves to
camera origin `[-1.176, 4.220, 1.953]` in `base_link`, and deprojected points
span a plausible cabin volume. Cabin floor is z~0, confirmed against a
`bboxes_3d` sample whose box spans z -0.055 to 1.782.

**Range caveat, measured on 20 sampled front_left_depth frames (8,140,800
pixels).** The D435i depth module is specified for roughly 0.3-3.0 m and a bus
cabin is longer than that:

| bucket | share of all pixels |
|---|---|
| zero / no return | 13.7% |
| 0.3-3.0 m (spec window) | **41.9%** |
| 3.0-6.0 m | 24.6% |
| 6.0-12.0 m | 10.4% |
| > 12.0 m | 9.4% |
| saturated at 65535 | 1.7% |

Per-frame p95 depth averages 23.2 m, which is physically impossible inside a
bus, so the far field is noise. **Under 42% of the frame is inside the reliable
window.** Provisional read: depth **supplements** RGB in the near field, it
does not replace it for full-cabin occupancy. The kernel prints this check and
says so explicitly when the in-range fraction is under 50%. To be confirmed on
the full subset.

### Planned config (when approved)

- subset: front_left_color stride 10 = 1,289 frames + 1,204 matched depth
  frames + all count annotations = **0.676 GB selected, ~0.741 GB transferred**
  in 2,456 range requests (measured by `--dry-run`). No 73.5 GB copy anywhere.
- split: **by recording sub-sequence, not at random** — frames are ~0.067 s
  apart, so a random split would put near-duplicates on both sides.
- branch 1: 1 epoch, batch 32, AdamW lr 1e-3 on the head only, 224x224.
- branch 2: no training; BEV grid 0.1 m cells, z 0.6-2.0 m, background from
  zero-occupant frames, scored against the same raw counts.
- backend: Kaggle. CPU kernels first (zero GPU quota), one P100 session for
  branch 1. Documented free tier is ~30 GPU-hrs/week
  (kaggle.com/docs/efficient-gpu-usage); **remaining quota unverified — the CLI
  exposes no quota endpoint, Diyas must read it at kaggle.com/settings.**

### Local plumbing checks (NOT experiments, no GPU, no quota)

Ran locally only to prove the code executes. These are not results and must not
be cited as any kind of performance number:
- index build over range requests: 298,694 members parsed, matches the
  independently-parsed central directory.
- extraction: 24 colour + matched depth frames + 32,911 annotation files;
  all 20 sampled depth files correctly renamed `.jpg` -> `.png` by magic-byte
  sniff; CRC32 verified per member.
- branch 2 executed on **23 frames** — far too few to mean anything.
- CORN math unit-checked: loss ~0 at the optimum, 0.693 (ln 2) at chance,
  cumulative probabilities rank-consistent, conditional subsets confirmed via
  gradient masking, sequence split disjoint.

### Notes / next

- Re-open the model choice when real camera data arrives; substitute-data
  results will not transfer, especially for levels 3-5 which are unobserved.
- A parallel search for a dataset with real crowding is running separately.
- Open question for Diyas: `count_view` (persons in this camera view) vs
  `count_cabin` (total occupants from 3D boxes). Default is `count_view`
  because it is what a single camera can actually observe; `count_cabin` is the
  product target but is not recoverable from one view under occlusion. Both
  columns are written to `labels.jsonl`.

## 2026-08-10 — sanas-extract-subset RAN; kernel chaining found broken; kernels made self-contained

- commit: 8db774b760ecd37227b37f29a2495167cc661bda (working tree has uncommitted changes)
- kaggle kernel: `diypyzsdiyas/sanas-extract-subset` — **RAN, succeeded**
- config: front_left_color stride 10, matched front_left depth, full label tail
- result (extraction only, no model): 35,404 files, 0.686 GB on disk, 1,289 frames,
  1,204 depth extensions corrected by magic-byte sniff.
  count_view  {0: 218, 1: 632, 2: 389, 3: 50}
  count_cabin {0: 120, 1: 656, 2: 453, 3: 60}
- **No model has been trained. No GPU has been spent.**

### Finding: `kernel_sources` does not pass data between kernels

A diagnostic run showed `/kaggle/input` containing only:
`notebooks/diypyzsdiyas/sanas-extract-subset` with 5 files and **zero
subdirectories**. `kernel_sources` mounts the source kernel's CODE, not its
output, so there is no `subset/` to read. `kaggle kernels output` does return
`subset/...`, so the data exists, it is just not reachable from a chained
kernel. The chained design in the previous entry was wrong.

Fix, approved by Diyas: every kernel is self-contained and fetches its own
subset by HTTP range request from the HuggingFace mirror. This applies to
`train_corn` too, which had the same broken assumption and would have failed
identically on its first run.

### Anti-drift mechanism

Four self-contained kernels sharing hand-copied logic is how the later
1-camera vs 4-camera comparison would quietly start measuring our own
inconsistency instead of measuring cameras. So `src/sanas/` is now the single
source of truth and `scripts/build_kernels.py` inlines modules into each
kernel between markers. `--check` is the drift detector.

Verified: the vendored block in `depth_occupancy.py` and
`depth_occupancy_multizone.py` is byte-identical (62,590 chars); the two
kernels differ only in `main(default_mode=...)`. Selection is shared via
`src/sanas/selection.py` and carries a regression assertion against the frame
counts the executed run produced (1,289 / 1,204 / 32,911 / 35,404) — all
kernels print CLEAN.

### Preliminary depth finding: the rig cannot see the whole cabin floor

Measured locally on a 30-frame sample (NOT the sanctioned run). Coverage is
the fraction of the region where occupants actually appear — derived from
every 3D box in `bboxes_3d` over the full recording, dilated 0.3 m, 4,110
cells — that receives any depth return:

| max range | 1 cam (front_left) | union of 4 cameras |
|---|---|---|
| 3.0 m (D435i spec) | 6.7% | **28.8%** |
| 4.5 m | 12.7% | 44.7% |
| 6.0 m | 16.1% | 54.9% |
| 10.0 m | 21.5% | 63.3% |

**Four cameras never cover the floor, at any range gate.** Even trusting depth
well beyond spec, a third of the occupied region has no return. The multizone
kernel therefore STOPS at step 1 by default rather than reporting a merged-BEV
number over floor it cannot see, since that would understate occupancy in a
way that looks like a model error rather than a sensing gap.
Caveat: this sample had only ONE empty-cabin frame to build coverage from.
The cameras are static so the footprint is close to fixed, but the sanctioned
run on 1,289 frames should be believed over this.

### Statistics warning for when results arrive

The subset tops out at 3 occupants and only 50 frames contain 3. Correlations
will be fragile. Both depth kernels now report bootstrap 95% intervals beside
every coefficient and print explicit FRAGILE lines listing sparse levels, and
the head-to-head states plainly when the 1-camera and 4-camera intervals
overlap. Do not quote a bare coefficient from this data.

### Not run

`sanas-depth-occupancy` (approved, coordinator will push),
`sanas-depth-occupancy-multizone` (NOT approved), `sanas-train-corn-smoke`
(NOT approved, needs GPU quota check).

## 2026-08-10 — PIVOT to door-mounted 3D APC; multizone CANCELLED; feasibility gate PASSED with caveats

- commit: 8db774b760ecd37227b37f29a2495167cc661bda (working tree dirty)
- kaggle kernel: none pushed. **No GPU spent. Nothing trained.**
- result: feasibility analysis only, run locally on already-extracted annotations
  and a 30-frame depth sample.

### Decision

Architecture moves from cabin-wide multi-camera occupancy to a **door-mounted 3D
Automatic Passenger Counter**: depth deprojection over a door aperture,
boarding/alighting by virtual line crossing, cumulative count with terminus reset.
Same principle as iris GmbH IRMA MATRIX, INIT and Dilax.

**Reason: the MVP defers incident detection (safety) to a later phase.** With no
safety requirement, cabin-wide multi-sensor coverage is over-built for what the
product must do, and the simpler industry-standard design is justified by MVP
priority. Explicitly NOT adding a Kalman filter or sensor fusion at this stage;
the cumulative counter is N_t = N_{t-1} + boardings - alightings.

`notebooks/depth_occupancy_multizone/` is marked **CANCELLED**, inactive, not
deleted. See `notebooks/depth_occupancy_multizone/CANCELLED.md`. It was never
pushed and never ran.

Unchanged and still current: `src/sanas/ziprange.py`, the deprojection and BEV
logic, the extraction path, `count_view`/`count_cabin`, and the CORN head with
DINOv2/ConvNeXt as future RGB backbone candidates.

### Feasibility gate (point 1): does any camera overlook a door?

Doors were located from the data, not assumed: `person_states.json` gives the exact
state strings, and each person's `pose_id` matches a `bbox_center` in `bboxes_3d`
at the same timestamp, so boarding/alighting positions are recoverable in
`base_link`.

**Finding A — no camera is top-down.** Optical axis angle from straight down:

| camera | mount height | axis off vertical | i.e. below horizontal |
|---|---|---|---|
| front_left_depth | 1.95 m | 71 deg | 19 deg |
| front_right_depth | 2.03 m | 74 deg | 16 deg |
| center_left_depth | 1.99 m | 67 deg | 23 deg |
| back_right_depth | 2.04 m | 71 deg | 19 deg |

A real APC looks straight down at the aperture; that is what makes line crossing
robust to occlusion. **This rig has no such view and cannot reproduce true APC
geometry.** Any result here is an oblique across-the-aisle approximation.

**Finding B — exactly one camera reaches the door zone in spec.** Points landing in
a 1.0 x 1.0 m door-zone footprint on the right side (x 0.5-1.5, y 0.05-1.05):

| camera | points/frame within 0.3-3.0 m | actual range |
|---|---|---|
| center_left_depth | **1,746** | 2.05-2.77 m |
| front_left_depth | 0 | 3.62-4.55 m |
| front_right_depth | 0 | 4.00-4.66 m |
| back_right_depth | 0 | 4.91-5.43 m |

So a single-camera door test is possible; a multi-camera one is not.

**Finding C — the labelled crossing positions are outside spec, but the
trajectories are not.** All 24 threshold positions (first frame of an entry
episode, last frame of an exit episode) sit 3.24-5.96 m from center_left, because
at that instant the person is still on the kerb, roughly 1 m outside the right
wall. Zero are usable. However, all **25 of 25** episodes have trajectories that
pass through center_left's usable zone, closest approach 1.22-2.24 m.

**Consequence for the design: the virtual line must be placed INSIDE the cabin, in
the vestibule a couple of metres from center_left, not across the door aperture
itself.** Everyone who boards or alights crosses it, within the reliable window.
This is a real deviation from the APC pattern and needs Diyas's sign-off before
counting code is written.

### Ground truth (point 5) — exact strings, as found

`person_states.json` state vocabulary, verbatim and with frame counts:
`walk` 4905, `sit` 2921, `stand` 2809, `hold on` 1305, `wheelchair` 1257,
`sit down` 828, `stand up` 705, **`exit vehicle` 436**, **`enter vehicle` 377**,
`` (empty) 8.

The paper's phrasing "entering or alighting" corresponds to `enter vehicle` and
`exit vehicle`. These are per-frame states, not events, so they were grouped into
episodes (contiguous runs, 2 s gap): **12 boardings + 13 alightings = 25 episodes**
across the whole 32-minute recording, median episode 29 frames (~2.9 s).

Ready-made ground truth with no manual annotation, but **25 events is the entire
validation set**, involving roughly 9 distinct persons, one of whom (`pose_id 1`)
contributes 8 episodes. Any accuracy figure from this will be extremely coarse;
a single miscount moves it by 4 percentage points.

### Not built yet

Points 2-4 (door-zone coverage kernel, line-crossing tracker, cumulative counter)
are held pending sign-off on the inside-the-cabin line placement from Finding C.
Building them against the aperture, as briefed, would produce nothing measurable.

## 2026-08-10 — two-phase plan recorded; Phase 2 assets marked dormant; weak-label constraints captured (NOTHING BUILT)

- commit: 8db774b760ecd37227b37f29a2495167cc661bda (working tree dirty)
- kaggle kernel: none pushed. No GPU. No counting code written.
- status: **Finding C sign-off still outstanding.** No line-crossing work started.

### Phasing

- **Phase 1** — door-mounted 3D APC: depth deprojection over a door zone, virtual
  line crossing, cumulative count. Depth geometry only, no learned model.
- **Phase 2** — RGB whole-frame cabin classification, DINOv2 or ConvNeXt with the
  CORN head, once camera permission for a real bus lands.

Marked **PHASE 2 ASSET - DORMANT, NOT DEAD** in their module docstrings:
`src/sanas/models.py`, `src/sanas/corn.py`. Same treatment as the cancelled
multizone branch. The banners sit in module docstrings, which
`scripts/build_kernels.py` strips when inlining, so generated kernels are byte
identical and `--check` stays clean.

Also Phase 2, not yet marked pending confirmation: `src/sanas/config.py`,
`src/sanas/data.py`, `notebooks/train_corn/` (that kernel *is* the Phase 2 model).

### Verification of the view-vs-cabin gap

Recomputed locally from the complete label tail (no download; all 10,034
`bboxes_3d` and all 12,890 front_left segmentation JSONs are already on disk).
Reproduces the executed Kaggle run exactly:

    count_view  {0: 218, 1: 632, 2: 389, 3: 50}
    count_cabin {0: 120, 1: 656, 2: 453, 3: 60}

Independent confirmation of `sanas-extract-subset`. Two corrections to how that
gap was characterised:

1. `218 - 120 = 98` is not the count of frames hiding someone. Measured directly,
   **view==0 and cabin>0 is 102 frames (7.9%)**, not 98 (7.6%). The subtraction
   under-reports because 4 frames have view>0 while cabin==0 - pseudo-label
   disagreement, so the zero-sets are not nested.
2. The more relevant number is larger. Counting any disagreement, **view < cabin
   in 181 frames (14.0%)** and view > cabin in 7 (0.5%). The 7.9% only counts
   frames where the camera sees nobody at all; the camera undercounts in 14% of
   frames. A weak-label scheme binding cabin-level counts to one view inherits
   that 14%, not 7.6%.

### Weak-supervision design constraints — RECORDED, DELIBERATELY NOT IMPLEMENTED

Idea (Phase 2, not now): door line-crossing gives a running occupancy N_t, and RGB
frames at the same timestamps inherit it as a noisy label before any manual
annotation. Requirements to keep the option open:

1. **Label schema must state which quantity it holds.** Already satisfied in
   spirit: `labels.jsonl` carries `count_view` and `count_cabin` as separate named
   columns, never a single implicit "count". A door-derived label is a third,
   cabin-level quantity with different provenance and would need its own named
   column plus a provenance marker - not a reuse of `count_cabin`.
2. **Cumulative counters drift, so labels degrade along a run.** Nothing in the
   current schema records time since last reset. It would need a per-segment
   anchor so a future run can down-weight late-segment labels.
3. **Reset anchoring is the only absolute ground truth** and must be recorded
   explicitly, not derived later.

Current `labels.jsonl` row: `timestamp`, `camera`, `image`, `sequence`,
`count_view`, `min_score`, `count_cabin`, `cabin_dt_ns`. Frame binding is by
19-digit nanosecond timestamp throughout, so timestamp-joining a future
door-count series is already possible. `sequence` is a recording-gap segment id
(5 s gaps), NOT a terminus reset - do not conflate them.

**Blocker for exercising any of this on the substitute dataset:** it is a single
32-minute session with no route, terminus or GPS data. There are no resets to
anchor to, so constraints 2 and 3 cannot be tested here at all - only designed
for. Flagging so nobody later mistakes `sequence` for a reset anchor.

No fields added, no counting code, no weak-label table. Design constraints
recorded only.

## 2026-08-10 — Phase 1 source switches to PCDS; download BLOCKED on host; nothing fetched

- commit: 8db774b760ecd37227b37f29a2495167cc661bda (working tree dirty)
- kaggle kernel: none pushed. No GPU. No download. No counting code written.
- result: remote verification only.

### Source change

Phase 1 moves off the Gorelik vestibule-line workaround (cancelled; it was a
patch for sideways-looking cameras and would not have tested real APC geometry)
onto **PCDS**, github.com/shijieS/people-counting-dataset, arXiv:1804.04339.
Camera is ceiling-mounted at bus doors with a pitch angle, which is the geometry
Finding C proved Gorelik lacks. Finding C stands as the reason for the switch.

Gorelik data is retained: it remains the source of `enter vehicle` /
`exit vehicle` cross-check labels and is the Phase 2 asset for cabin RGB.

Full provenance, licence and caveats: `docs/datasets.md` section 3.

### Download links — checked, and this is the blocker

- **Google Drive: dead, HTTP 404.** The README's note is accurate, now confirmed.
- **Baidu Pan: alive but gated.** shorturlinfo returns shareid 6773605951,
  uk 2972546568, expired_type 0, so the share is live. Listing returns errno -9
  (code required); verify returns errno 105 (anti-automation). Needs a browser
  session and in practice a Baidu account.
- GitHub: 0 releases, 0 assets.

**Volume is unknown and unobtainable remotely. It is stated nowhere** — not the
README, not the paper, not the project page. Estimate from the authors' own demo
clip durations (depth 16-29 s, real YouTube metadata) and Kinect V1 raw depth
rate puts it somewhere in **150-800 GB**, an uncertainty of more than 5x. That
is an estimate, not a measurement, and **no download should start on it**.
Someone must open the Baidu link in a browser and read the real folder size.

### Sensor caveats, all three verified

1. **Sunlight: present in the data, and labelled.** N+/N- is literally the
   sunlight axis. Paper: "Kinect V1 camera is sensitive to illumination
   conditions. For strong illumination, there is often noise in the videos [...]
   recorded in either direct sunlight or diffused sunlight". So PCDS does NOT
   avoid the condition. But it is imbalanced: only **3,370 of 20,908 people
   (16.1%)** are in strong sunlight (N+C+ 2,086 and N+C- 1,284, against N-C+
   12,074 and N-C- 5,464). **Aggregate accuracy would be optimistic for an
   Almaty door; results must be stratified N+ vs N-.**
2. **Kinect V1 differs from the D435i we assumed.** Structured-light IR vs
   active IR stereo; 0.8-4.0 m default range (0.4-3.0 m near mode) vs ideal
   0.3-3 m; 640x480/320x240/80x60 vs the 848x480 measured in Gorelik; 57x43 deg
   FOV. The two fail differently in sunlight, so PCDS noise does not predict
   D435i behaviour in either direction. The paper states no resolution, frame
   rate or range at all. Separately, **the Gorelik cameras are never identified
   as D435i either** — that has been an inference from 848x480 throughout.
3. **2016 data.** From scene naming (`25_20160411_front`) and the README, not
   from the paper body. The README misreads its own example as "04, Nov. 2016"
   when the format gives 11 April 2016.

### Licence — hard blocker, recorded in docs/datasets.md

**CC BY-NC-SA 3.0.** NonCommercial excludes commercial Avtobys outright, and is
stricter than RPEE-HEADS (CC BY-SA 4.0, where only ShareAlike is unsettled).
Same class as the P2PNet academic-only clause: **the shipped model must be
retrained on licence-clean data, meaning our own recording.** PCDS validates the
line-crossing logic; it cannot supply production weights. Authors offer a
commercial contact if that is ever needed.

### Also this session

`src/sanas/config.py`, `src/sanas/data.py` and `notebooks/train_corn/` given the
same PHASE 2 DORMANT banner as `models.py` and `corn.py`, so the model and the
kernel that trains it stay consistent. All banners are in module docstrings,
which the generator strips, so `build_kernels.py --check` stays clean. Verified.

### Not done

No counting code. Phase 1 implementation is blocked on two things: the Finding C
decision is now moot for Gorelik but the PCDS equivalent is unresolved, and more
immediately **nobody can build against PCDS until the archive is actually
obtainable.** Next action is a human opening the Baidu share.

## 2026-08-10 — door-APC virtual-line counter RAN locally; full-res vs 8x8 vs 4x4; PCDS still blocked

- commit: 7673883d7094c8fd57ef7ef570e1759937d96ead (working tree dirty; new
  `src/sanas/door_apc.py`, `scripts/run_door_apc.py` in this change set)
- backend: **local CPU only.** No kernel pushed, no GPU, nothing trained.
  Geometric pipeline; an accelerator has no role here.
- config: center_left_depth only; line x=0.20 m in base_link, hysteresis
  ±0.15 m; zone x[-0.6,1.6] y[-0.2,1.3] z[0.2,2.0]; bg = per-pixel median
  over 48 empty-cabin frames; NN tracker gate 0.8 m; ALL DEFAULTS UNTUNED
  (marked in ApcConfig). Multizone sim: 60×60° crop, 8x8 and 4x4 zone
  pooling at p20, same tracker. ~7.7 Hz (every 2nd frame), 3 episodes at
  15 Hz as rate spot check — identical events at both rates.
- data: 2,215 depth frames, 0.427 GB selected / 0.429 GB transferred in
  1,778 range requests + ~7 MB annotations. Hard cap 0.5 GB held. CRC32
  verified; 2,215 `.jpg`→`.png` magic-byte corrections.
- result (n=25 episodes = 12 board + 13 alight; every event = 4 pp):

  | | full 848x480 | 8x8 sim | 4x4 sim |
  |---|---|---|---|
  | detected | 19/25 (CI 57-89%) | **24/25 (CI 80-99%)** | 21/25 (CI 65-94%) |
  | direction | 19/19 | 24/24 | 18/21 |
  | FPs / 63 s negatives | 0 | 0 | 0 |
  | N_t MAE vs door truth | 1.45 | **0.17** | 1.92 |
  | final error | -3 | +1 | +3 |

- notes:
  1. **8x8 beat full resolution under untuned defaults.** Diagnosed, not
     guessed: full-res loses the six line-hover episodes to cluster
     fragmentation + track birth stealing the association at the crossing
     instant; 8x8 pooling regularises the person to one centroid. Left
     unfixed deliberately — tuning against the only 25 events would corrupt
     the validation set.
  2. **4x4 is below the working floor** (3 direction errors, 5 duplicates).
  3. **Vestibule-line ≠ door-event counting, measured:** actors already
     aboard cross the line plane between episodes (wheelchair user ~t+262 s),
     so line-truth MAE (1.53 for 8x8) ≫ door-truth MAE (0.17). A cost of the
     Finding C inside-the-cabin placement that aperture APC does not pay.
  4. Counting-band ablation (coordinator request → node_design.md §3):
     detection flat down to 0.7 m band width, collapses at 0.5 m
     (16/25). ±0.25 m ceiling strip is below the measured working minimum.
  5. Shared failure: back-to-back double exit (pids 190/172, 0.1 s apart)
     merged by every variant — simultaneous crossings remain the hard case.
  6. Frame-rate deviation on the record: the ±5 s full-15 Hz brief costs
     746 MB > 0.5 GB cap; ran at 7.7 Hz with asymmetric evidence-based
     buffers (board [-2,+5] s, alight [-5,+2] s) + 15 Hz spot check.
  7. **Task-brief context correction:** this run executes the Gorelik
     vestibule-line validation that entry "Phase 1 source switches to PCDS"
     had cancelled. Coordinator-approved reversal: PCDS remains unobtainable
     (Baidu-gated, human browser session required, volume unknown) and its
     CC BY-NC-SA 3.0 licence is paper-track only regardless. Zero PCDS bytes
     downloaded. Finding C line placement approved via delegated sign-off.
  8. Episode grouping reproduces the log exactly (25 = 12+13, 377/436 state
     frames); median episode 31 frames vs 29 recorded earlier — grouping
     artefact, headline counts identical.
- artifacts: `paper/results/apc_validation.json` (incl. 1 Hz N_t series +
  per-episode table, feeds figures), `paper/results/apc_validation.md`,
  `docs/trade_study.md` (VDV 457 verified-to-secondary-sources; all numbers
  tagged full-text / abstract / vendor / not-stated).
- next: real ToF hardware (VL53L7CX) at a real door top-down; PCDS if a
  human opens the Baidu share; simultaneous-crossing handling; do NOT quote
  any number here as bus performance — substitute data, oblique geometry.

## 2026-08-10 — code-review fixes + Fig. 1 definition clarifications; metrics verified unchanged

- commit: 7673883d7094c8fd57ef7ef570e1759937d96ead (working tree dirty)
- backend: local CPU. No fetch, no GPU. Re-scored already-local frames only.
- changes (coordinator review, 4 items):
  1. `scripts/run_door_apc.py` process_all: hard SystemExit when any selected
     frame or background frame is missing on disk — `--evaluate` on a partial
     fetch can no longer silently write results.
  2. Spot-check indices now a checked contract: `SPOTCHECK_EPISODES` carries
     expected (pid, kind) and `check_spotcheck()` hard-verifies at startup
     (prints: [6]=pid 1 board, [14]=pid 119 alight, [21]=pid 158 board).
  3. `src/sanas/door_apc.py` gains `self_test()` — 6 deterministic scripted
     tracker checks (clean board, clean alight, hysteresis hover no-count,
     side-0 birth exiting either way no-count, round trip board+alight),
     same no-framework convention as the CORN math verification. PASSED.
  4. Misleading `block_of` fallback comment fixed (bounds covers negative
     chunks too; fallback is defensive/unreachable).
- Fig. 1 mismatches (CADy): all three definitional, nothing recomputed,
  no published number wrong:
  1. `series_1hz.estimate` endpoints are RAW counts; final_error fields are
     errors vs truth at the last processed frame (est 0/-4/+2 minus door
     truth -1 -> +1/-3/+3). Documented.
  2. `series_1hz.truth` is full-session LINE-position truth (92 transitions,
     not chunk-gated/zero-filled); the clean 25-step staircase is the new
     additive `series_1hz.door_truth` field (25 transitions, range -4..+1;
     negative because pre-aboard actors exit after an anchor of 0).
  3. MAEs are over the 2,167 processed frames, never the 1 Hz grid;
     on-grid MAE would score frozen estimates on unobserved intervals.
  A `definitions` block was added to apc_validation.json and mirrored in
  apc_validation.md.
- verification: post-fix `--evaluate` re-run on local data is BYTE-IDENTICAL
  to the published JSON before the definitions addition; after the addition,
  a field-level diff confirms every previously published field identical and
  only `definitions` + 3 `door_truth` arrays added. Ruff format/check clean.

## 2026-08-10 — COORDINATION RECORD: architecture decided, team run complete, paper + hardware + pitch delivered

- commit: 7673883 (working tree dirty; this entry records the coordinated
  session, not a training run)
- backend: local CPU + background agents only. No GPU spent, no kernel
  pushed, nothing published externally.

### Decision (coordinator, under design authority delegated by Diyas)

Sensing architecture for Sanas Phase 1: **door-mounted downward depth APC
per door**, counting line inside the vestibule (Finding C signed off under
delegated authority), with the **single-chip 64-zone ToF (VL53L7CX class)**
as the target production sensor, pending hardware bench validation.
Cabin-wide depth occupancy rejected on measured evidence (10% coverage,
r=0.184, non-monotonic). Salon RGB + CORN ordinal head remains Phase 2,
dormant, untrained. Fusion: cumulative count, terminus reset, ordinal
5-level output to Avtobys.

Basis: this session's ablation (8x8 sim 24/25 detected, direction 24/24,
0 FPs, door-truth MAE 0.17 on n=25 episodes — see the two entries above)
plus the trade study (docs/trade_study.md). All caveats stand: substitute
data, oblique geometry, 25 events, staged actors; no VDV 457 claim.

### Team outputs this session

- Donatello: src/sanas/door_apc.py + scripts/run_door_apc.py (reviewed,
  self-test green, results byte-identical after review fixes),
  paper/results/apc_validation.{json,md}, docs/trade_study.md.
- CADy: docs/hardware/ (system_architecture, wiring, node_design, bom +
  4 SVG figures, FoV corrected to confirmed 60x60), paper/figures/
  fig1-fig3 scripts + renders; fig4-fig7 exported from SVG via Edge
  headless by coordinator (no cairo toolchain on this machine).
- Danyshpan: paper/sanas_apc.tex (IEEEtran draft, compilable; LaTeX not
  installed locally — compile on Overleaf), paper/figures.md,
  paper/related_work_notes.md (coordinator), bibliography with explicit
  TODO markers for unverified metadata.
- Pitch: docs/pitch/ (avtobys_deck, one_pager, elevator_pitch RU+EN,
  akimat_note, demo_script) — all numbers sourced or marked
  [нужна реальная цифра].

### Coordinator corrections on the record

- VL53L7CX FoV: first hardware draft assumed 60 deg diagonal; confirmed
  60x60 square (90 deg diagonal). Head-height strip is ±0.35 m = exactly
  the measured 0.7 m band minimum, zero margin — mounting needs tilt or
  inboard offset. Propagated to node_design.md, figures.md, sanas_apc.tex,
  sensor_placement.svg.
- Fig. 1 truth semantics reconciled via the JSON definitions block;
  door_truth staircase added by Donatello, caption rewritten by Danyshpan.

### Open items for Diyas

1. Buy 2-3 VL53L7CX breakout boards + ESP32-S3 and bench-test the real
   sensor top-down (the single biggest unvalidated step; sim used D435i
   noise, oblique view).
2. PCDS validation needs a human Baidu session (volume unknown, licence
   NC — paper track only).
3. Scite quota resets 2026-09-01 — finish full-text citation verification
   before any submission.
4. GPU smoke run for Phase 2 CORN head: still NOT approved, still blocked.
5. Paper author list / affiliation TODO in sanas_apc.tex.

---

## 2026-08-13 — repo reorganised into three track folders

Not an experiment run. Structural move only: no code logic, no numbers and
no results changed. Recorded here because this log is the place that says
where things live.

Layout now (top level, one folder per role group):

| Was | Is |
|---|---|
| `paper/` | `research/paper/` |
| `docs/trade_study.md` | `research/trade_study.md` |
| `2606.11739v1.pdf` (root) | `research/refs/` (gitignored) |
| `src/`, `scripts/`, `notebooks/`, `experiments/` | `development/` |
| `docs/hardware/` | `development/hardware/` |
| `docs/findings.md`, `docs/datasets.md` | `development/` |
| `docs/pitch/*` | `business/*` |
| `data/`, `outputs/` | unchanged, stay at repo root (3.8 GB, gitignored) |

`docs/` no longer exists.

Path fixes that came with the move:
- `development/scripts/run_door_apc.py` — `../data` became `../../data`,
  results now written to `../../research/paper/results/`.
- `business/demo/make_demo_animation.py` — REPO is `parents[2]`, sources
  `development/src`, reads `research/paper/results/`, writes next to itself.
- Kernels regenerated by `development/scripts/build_kernels.py` so the
  inlined-module banners match the new module path; `--check` is clean.
  Kernel *behaviour* is byte-identical apart from those comment banners.
- Markdown/tex cross-references rewritten to the new repo-root-relative
  paths. Entries above this line keep their original paths on purpose —
  this log is append-only history, not a live index.

Verified after the move: `run_door_apc.py --plan` reproduces the same
selection (2,215 frames, 0.427 GB, episode cross-check CLEAN),
`build_kernels.py --check` up to date, `sanas` imports resolve, ruff
check/format status unchanged from the pre-move baseline (152 pre-existing
lint errors, 4 generated kernels unformatted — same as before).

## 2026-08-25 — documentation reconciled; STATE.md added; one stale number corrected

Not an experiment run. No code logic changed, no kernel touched, no GPU, no
fetch. Recorded here because this log is the place that says where things
live, same as the reorganisation entry above.

### Problem

Three documents each claimed authority and disagreed with each other and
with this log on the two questions that matter most — what the sensor is
and whether a model is involved:

| Document | Said | Actual (this log, 2026-08-10) |
|---|---|---|
| `SANASH_Master_Document.md` §3/§5/§7 (v1.0, 6 Aug) | Jetson Orin Nano Super + P2PNet + Hikvision RGB, ~$3,457 procurement | 64-zone ToF (VL53L7CX class) + ESP32-S3; P2PNet licence-excluded; no trained model in Phase 1 |
| `development/sanas_cv_track_conclusions.md` §8 (10 Aug) | "final decision": Jetson + 2× RealSense D435i | superseded the same day by the ToF decision |
| same, §5 | "4-camera hypothesis, test not run" | test ran; refuted — 28.8% coverage at spec range |

The master document's own 23 Aug note pointed readers at
`sanas_cv_track_conclusions.md` §8 as the current answer — i.e. one stale
document forwarding to another.

### Changes

1. **`STATE.md` added at repo root.** Current-state map: architecture in
   force, measured results with their caveats, what was rejected and on what
   evidence, a document-authority table, and the open items ranked by
   blocking power. Carries no numbers of its own — every figure cites the
   file it came from. States the conflict rule explicitly: this log wins.
2. **Staleness banners** added to `SANASH_Master_Document.md` and
   `development/sanas_cv_track_conclusions.md`, each naming the specific
   sections not to trust and the specific sections still worth reading.
   Neither document was deleted or rewritten — the still-valid parts are
   substantial (research question, novelty argument, 43 sourced references,
   the outreach template; the dataset analysis and the two-phase framing).
3. **`README.md`** gains a "Start here" block pointing at `STATE.md` and
   stating the conflict rule.
4. **`development/findings.md` §5 — stale number corrected.** It read
   "98 frames, 7.6%" for single-camera occlusion. The entry
   "two-phase plan recorded" above had already corrected this by direct
   measurement but findings.md was never updated. Now reads: `view == 0 and
   cabin > 0` is **102 frames (7.9%)**; the operative figure is any
   undercount, `view < cabin`, at **181 frames (14.0%)**; `view > cabin` is
   7 frames (0.5%). The subtraction under-reports because the zero-sets are
   not nested.
5. **Two PDFs moved**, `Methodology.pdf` and
   `Methodology for emp. papers.pdf`, from the repo root into
   `research/coursework/`. `research/coursework/README.md` already listed
   both in its contents table while they sat at root; the files are
   gitignored either way, so history is unaffected.

No source file under `development/src/` or `development/scripts/` was
touched, so no kernel regeneration was needed and no result can have moved.

## 2026-08-25 — ARCHITECTURE PIVOT: door APC + RGB-cabin design abandoned, ceiling device to replace both phases

**STATUS: deletion recorded, not a result. Nothing below was measured today
— this entry documents what was removed and why, per Diyas's explicit
instruction in-session. Reason for the pivot was not stated beyond "будем
создавать свой отдельный девайс для потолка" — not filled in further here
to avoid inventing a justification that wasn't given.**

- commit at time of deletion: a9712795bb516fa604f16f700c124d8832eec6dc

### Decision

Both prior phases are cancelled, not just Phase 1:
- Phase 1 (door-mounted 64-zone ToF APC counter, VL53L7CX class, geometric
  count, no model) — cancelled.
- Phase 2 (RGB cabin classification, DINOv2/ConvNeXt + CORN ordinal head,
  dormant) — cancelled, not merely left dormant.

New direction: a single ceiling-mounted device, replacing both. Sensor
modality, exact placement, and CV method are **open** — not specified yet,
to be decided fresh rather than carried over from the trade study or
backbone literature that supported the door/cabin design.

### What was deleted this session

- `development/src/sanas/*` — all 11 modules (`door_apc.py`, `corn.py`,
  `models.py`, `depth_occupancy.py`, `depth_kernel.py`, `config.py`,
  `data.py`, `labels.py`, `selection.py`, `fetch.py`, `ziprange.py`,
  `__init__.py`)
- `development/hardware/*` — all 4 docs + 4 SVGs (door-node design, wiring,
  BOM, system architecture)
- `development/notebooks/{depth_occupancy,depth_occupancy_multizone,
  extract_subset,train_corn}/` — all 4 kernel scaffolds
- `research/paper/sanas_apc.tex`, `related_work_notes.md`, `figures.md`,
  `figures/` (Fig. 1-7), `results/apc_validation.{json,md}` — the entire
  Phase-1 paper draft
- `research/trade_study.md` — sensor modality trade study (depth vs mmWave/
  CO2/Wi-Fi/thermal), tied to the door/cabin decision
- `research/refs/{tof-apc,density-estimation,backbone-methods,
  alternative-modalities}/` — 49 papers total + `graph.json`,
  `REFGRAPH_REPORT.md`, `graph_view.html` (all untracked, not in git
  history either way)
- `research/download_links.md`, `research/master_bibliography.md`,
  `research/paper/related_work_density_estimation.md`,
  `research/scripts/` (refgraph.py tooling for the deleted corpus)

**Kept, deliberately:** `research/refs/rtci-supporting/` and the RTCI
causal-experiment literature — separate paper track (does real-time
crowding info change boarding decisions), not part of the device design
being replaced. `business/*` (deck, one-pager, demo video/stills) — not
touched, flagged to Diyas as referencing the old door-counter demo, no
deletion instruction given for that track yet. `STATE.md`,
`development/findings.md`, `development/sanas_cv_track_conclusions.md` —
left as historical record of what was measured on the abandoned design;
not rewritten in this entry, will need a pass once the ceiling device's
own architecture exists to describe.

All deletions were via `git rm` (tracked files, recoverable from history)
or plain delete (untracked refs/lit, not recoverable — were gitignored per
`CLAUDE.md`'s "PDFs too large for history" rule). Nothing was committed;
working tree only.

## 2026-08-25 — NEW STACK DECISION: ceiling device, full-frame RGB, single phase (nothing trained yet)

**STATUS: decision only, dictated by Diyas in-session. No kernel run, no
GPU spent, no code written yet for this stack — that's next.**

### Hardware

- **Jetson Orin Nano Super Dev Kit** — onboard compute, TensorRT for
  ViT/attention architectures
- **Waveshare IMX219-160** camera (IR variant IMX219-160IR for night)
- NVMe SSD — storage
- USB-C PD power bank (65W+, 15V via trigger cable) — standalone, not
  wired into bus electrical system

### CV model, MVP

**CSRNet + PFCASA** (Rostamza et al., arXiv:2605.18349, JKU Linz) —
parameter-free attention, effective at low crowd density (<40 people),
matches bus-cabin conditions. This is the paper filed as `rostamza2026` in
the now-deleted `research/refs/density-estimation/` corpus (see the
architecture-pivot entry above) — full text was already read and
synthesized once; the PDF itself needs re-fetching if it's going to be
cited or re-verified.

### Architecture

Single phase only — what was previously called Phase 2. Full-frame RGB
classification of the whole cabin. One shared backbone, separate head.
Output: 5 ordinal density levels (empty -> crush load) + continuous 0-1
score.

Baseline / previous CV stack, not used in MVP: frozen **DINOv2 ViT-S/14 +
CORN ordinal head** + band/strip pooling + conformal prediction;
**ConvNeXt-Tiny + CORN** as the edge-deployable comparison. Both backbones'
source papers (`oquab2023`, `shi2023`) were in the deleted
`backbone-methods/` corpus — same re-fetch note as above applies if this
baseline gets built out.

### Datasets

1. Multi-View In-Cabin Monitoring System (Gorelik, Karrow, Sivrikaya,
   Albayrak / GT-ARC, TU Berlin + Baumann / MAN Truck & Bus SE) — 9,136
   synced RGB+depth samples, 4 cabin cameras + LiDAR, German urban bus,
   nuScenes format. (This is `gorelik2026`, already in `research/refs/`
   pre-pivot — file itself untouched by the deletion pass, wasn't inside
   the four deleted subfolders.)
2. RPEE-HEADS — **field_natural subset only**, `lab_capped` excluded.
   License caution carried over from prior review: CC BY-SA 4.0,
   ShareAlike applies to derived weights — not resolved for a shipped
   product, was previously scoped "paper-track only". Needs a decision
   now that this is the MVP path, not just the paper.
3. DISCO — for pretraining.

### Training backend

Kaggle CLI (`kaggle kernels push/status/output`) — stand-in until a
dedicated GPU server exists. Same backend as before the pivot.

### Software integration

Avtobys (Innoforce) displays the continuous 0-1 score as a green-to-red
gradient.

### Decision: PCDS dropped from the stack

Was scoped for door-zone validation, which is cancelled. Format (RGB-D
directly over a door) doesn't fit full-frame cabin classification. License
was already NC-restricted (paper-track only) and the archive (Baidu) was
already flagged as practically unfetchable (anti-automation, 150-800GB
size estimate, 5x spread). Both the use case and the access path are gone
— removed rather than kept as unused baggage.

### Open, not decided in this entry

- RPEE-HEADS ShareAlike-on-weights question (above)
- `business/*` (deck, one-pager, demo video) still shows the cancelled
  door counter — no instruction yet on whether to update or leave
- `STATE.md` still describes the cancelled two-phase door/cabin
  architecture — needs a full rewrite pass once this stack is confirmed
  stable, not touched in this entry

## 2026-08-25 — business/ pitch materials for the door counter deleted

Diyas instruction: "удаляй все что связано с дверным проходом" (delete
everything related to the doorway). Checked each file in `business/` for
door-counter content before deleting — all five `.md` files were entirely
built around the door sensor pitch (Phase 1 as primary solution, RGB
cabin camera only mentioned as a dotted-line future Phase 2), not mixed
content, so deleted whole rather than editing sections:

- `business/akimat_note.md`, `avtobys_deck.md`, `demo_script.md`,
  `elevator_pitch.md`, `one_pager.md`
- `business/demo/` entire folder — `make_demo_animation.py`,
  `sanas_demo.mp4`, `stills/still_{01_sensor_view,02_boarding,
  03_alighting,04_final_card}.png` — animation was literally the 8x8
  door-sensor readout + boarding/alighting counter, no part of it
  transfers to a ceiling device

**Not touched:** `business/outreach/*` — the researcher contact lists
mention "APC" only in describing *other academics'* published work
(Pronello's field APC comparison, Qian's crowdsourced-fullness-vs-APC
validation), not our own hardware. Different track, left alone.

`business/` now contains only `outreach/`. No pitch deck, one-pager, demo
script, or demo video exists for the ceiling device yet — new versions
need writing once the stack is stable enough to pitch (currently still
open: RPEE-HEADS license question, and this stack itself is one message
old, untested).

## 2026-08-26 — RESEARCH SCOPE DECISION: RTCI behaviour is the core study

**STATUS: research-scope decision only. No field experiment, survey response,
device run or causal result is recorded here.**

Diyas clarified that the central research question is the extent to which
real-time bus occupancy information changes Almaty commuters' boarding
decisions. The ceiling occupancy device is therefore a measurement/enabling
system for RTCI, not the scientific endpoint by itself.

Planned evidence sequence:

1. audit and scale the existing stated-preference survey;
2. build a reproducible RTCI literature review;
3. validate occupancy measurement in silent mode;
4. run a limited Avtobys field experiment comparing boarding now vs waiting.

The initial idea of one baseline week followed by one city-wide information
week is not fixed methodology. It is vulnerable to time confounding and will
be compared against app-level randomized A/B or cluster crossover on a selected
high-frequency route. Current working protocol:
`research/RTCI_RESEARCH_CHARTER.md`.

## 2026-08-26 — PUBLICATION TARGET: Transportation Research Part C

**STATUS: publication strategy decision, not a submission or acceptance.**

Diyas selected *Transportation Research Part C: Emerging Technologies* as the
target journal. This raises the required contribution above a survey or device
benchmark. The working paper must connect:

1. measured occupancy technology and information quality;
2. causal boarding/waiting behaviour in the field;
3. a behaviour-aware choice model;
4. implications for waiting, load distribution and service reliability.

Journal-fit rationale and relevant TR-C precedents were added to
`research/RTCI_RESEARCH_CHARTER.md`. No claim is made that the current project
already meets this threshold.
