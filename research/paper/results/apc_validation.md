# Door-APC virtual-line validation on center_left_depth

Run 2026-08-10, local CPU only (no GPU, no Kaggle, nothing trained).
Commit at run time: `7673883d7094c8fd57ef7ef570e1759937d96ead` (working tree
dirty; counter code `development/src/sanas/door_apc.py` and runner
`development/scripts/run_door_apc.py` are part of the uncommitted change set this entry
describes). Machine-readable results: `apc_validation.json` (same directory,
includes the full 1 Hz N_t series and per-episode table for figures).

**These numbers are a pipeline check on substitute data, not a claim about
real bus performance.** Oblique across-the-aisle viewpoint instead of APC
top-down, one door, staged actors, stationary bus, daytime only,
pseudo-label ground truth, and n = 25 events, 8 of which are one person.

## Setup

- Data: Gorelik et al. in-cabin dataset (Zenodo 10.5281/zenodo.20559664,
  CC-BY 4.0), `center_left_depth` stream only — the single camera whose
  door-zone view lies inside the D435i 0.3-3.0 m spec window (Finding B).
- Virtual counting line inside the vestibule at x = 0.20 m in `base_link`
  (Finding C placement), 1.54-2.39 m from the camera across the corridor
  plane, hysteresis band +-0.15 m. All counter parameters are UNTUNED
  defaults chosen from geometry inspection, marked in `ApcConfig`.
- Ground truth: 25 episodes (12 board / 13 alight) from `person_states.json`
  (`enter vehicle` 377 frames / `exit vehicle` 436 frames, grouped per
  person+state with 2 s gap tolerance — reproduces the log's counts
  exactly; median episode 31 frames vs the log's 29, a grouping artefact,
  headline counts identical). Cabin count over time from `bboxes_3d`.
- Frames: 2,215 selected (0.427 GB compressed, hard cap 0.5 GB), 2,167
  processed: 21 episode chunks (board [-2,+5] s, alight [-5,+2] s buffers,
  asymmetric per the measured crossing-time distribution), 6 negative
  chunks totalling 63 s spread across the session, 48 empty-cabin
  background frames from 4 stretches.
- **Frame-rate deviation, on the record**: the brief asked for +-5 s
  buffers at the full 15 Hz; measured cost 746 MB before negatives, over
  the 0.5 GB cap for every buffer trim tried. Run instead at every 2nd
  frame (~7.7 Hz) with three episodes fetched at 15 Hz as a spot check.
  Result: event streams at 15 Hz and 7.7 Hz are identical on all three
  (same counts, kinds). 7.7 Hz is also the frame-rate class of the real
  8x8 sensor being simulated (VL53L7CX-class: 15 Hz max at 8x8).
- Multizone simulation: 60x60 deg crop (matches VL53L7CX square FoV;
  vertical limited to the sensor's ~58 deg), pooled to 8x8 and 4x4 zone
  ranges (20th-percentile of valid pixels per block, untuned), identical
  zone/line/tracker logic downstream.

## Headline table

| metric | full res (848x480) | 8x8 ToF sim | 4x4 ToF sim |
|---|---|---|---|
| episodes detected | 19/25 (76%, CI [57, 89]) | **24/25 (96%, CI [80, 99])** | 21/25 (84%, CI [65, 94]) |
| direction correct (of detected) | 19/19 | 24/24 | 18/21 |
| duplicate events in windows | 1 | 0 | 5 |
| false positives on 63 s negatives | 0 (CI [0, 3.49]/min) | 0 (CI [0, 3.49]/min) | 0 (CI [0, 3.49]/min) |
| N_t MAE vs door-event truth | 1.45 [1.08, 1.91] | **0.17 [0.11, 0.26]** | 1.92 [1.47, 2.37] |
| N_t final error vs door truth | -3 | **+1** | +3 |
| N_t MAE vs line-position truth | 2.83 [2.07, 3.60] | 1.53 [0.94, 2.17] | 0.93 [0.69, 1.19] |

Wilson 95% CIs for proportions, Garwood/Poisson for the FP rate, block
bootstrap over chunks for MAE. Every detection is worth 4 percentage
points; treat all intervals as the honest summary.

## The three findings

**1. The 8x8 simulation beats full resolution with these untuned defaults,
and the mechanism is diagnosed, not guessed.** Full-res misses exactly the
six line-hover episodes (pids 135, 146, 158 x2, 172, and 135's alight): a
person stands ON the line for seconds (traced: centroid at x 0.17-0.18 for
>3 s in episode 146), and at the crossing moment depth-noise fragmentation
splits the silhouette into two clusters; the greedy tracker births a new
track on the cabin side and the crossing is attributed to nobody. Coarse
8x8 pooling spatially regularises the person to one centroid and counts
the same crossing cleanly. This is a genuine property of the untuned
full-res clustering (cell 0.15 m, min 60 points), not a code bug; coarser
full-res clustering would likely recover it, but tuning against these 25
events would corrupt the only validation set, so it is left measured and
unfixed. The paper framing: **spatial pooling to sensor-grid resolution
acted as regularisation for door counting; the $-class 64-zone simulation
did not degrade counting on this data — it improved it.**

**2. 4x4 is the resolution floor where direction breaks.** Detection holds
(21/25) but direction errors (3) and duplicates (5) appear: two zones'
worth of person straddling the line chatters. 8x8 is the working minimum
in this geometry; 4x4 is not.

**3. Line-crossing counts and door-event counts are different quantities on
this data.** The line-position truth (people on the cabin side of the
plane, from `bboxes_3d`) moves between episodes without any door event —
actors already aboard (e.g. the wheelchair user, ~t+262 s) cross the
vestibule plane internally. A deployed vestibule-line counter would count
them; a door-aperture APC would not. On this staged recording the gap
dominates the line-truth MAE (mz8: 1.53 vs 0.17 door-truth). Deployment
consequence: an inside-the-cabin line needs either a zone that internal
circulation cannot clip, or acceptance that N_t tracks "cabin side of the
line", not "aboard". This is a measured cost of the Finding C vestibule
placement that pure aperture geometry would not pay.

## Counting-band ablation (mounting-spec input, coordinator request)

Full-res path, zone truncated to |x - line| <= w:

| w (m) | band width | detected | direction | FPs |
|---|---|---|---|---|
| 1.00 | 2.0 m | 19/25 | 19/19 | 0 |
| 0.70 | 1.4 m | 20/25 | 20/20 | 0 |
| 0.50 | 1.0 m | 19/25 | 19/19 | 0 |
| 0.35 | 0.7 m | 19/25 | 19/19 | 0 |
| 0.25 | 0.5 m | **16/25** | 15/16 | 0 |

Detection is flat down to a 0.7 m band and collapses at 0.5 m: with
hysteresis +-0.15 m, a 0.5 m band leaves only 0.1 m of visible approach on
each side and tracks cannot establish a side before crossing. For
node_design.md section 3: **the effective counting band must be at least
~0.7 m wide in the direction of travel; the +-0.25 m head-height strip of
an un-tilted ceiling 60 deg sensor is below the working minimum measured
here.** Caveat: measured on an oblique view narrowed in base_link, not on
a real ceiling footprint; the hysteresis needs to scale down if the band
does.

## Definitions (added after CADy's Fig. 1 review, 2026-08-10)

The JSON now carries a machine-readable `definitions` block; prose version:

- **line truth** (`series_1hz.truth`): boxes with center x < x_line per
  annotated frame, whole session, 92 transitions, range 0-2. Not
  chunk-gated, not zero-filled; it moves whenever anyone crosses the plane,
  door event or not. 1 Hz samples take the nearest FOLLOWING annotated
  frame (up to the gap length across the 10 recording gaps).
- **door truth** (`series_1hz.door_truth`, new field): anchor + cumulative
  25-episode staircase (+1 board / -1 alight at episode midpoints). Exactly
  25 transitions, range -4..+1. It goes NEGATIVE because the anchor counts
  only line-side occupancy at the first processed frame; actors already
  aboard (never seen boarding) exit later. Fig. 1 should plot THIS as the
  door-event reference, and the caption should say why it dips below zero.
- **`series_1hz.estimate`**: the raw cumulative count, not an error. Frozen
  outside processed chunks. Its endpoint (0/-4/+2 for mz8/full/mz4) minus
  session-end door truth (anchor - 1 = -1) gives
  `final_error_vs_door_truth` (+1/-3/+3). No inconsistency.
- **MAEs**: over the 2,167 processed frames only, never the 1 Hz grid.
  On-grid MAE differs by construction: outside processed chunks the
  estimate is frozen over intervals the counter never observed, so an
  on-grid figure would mix counter error with unobservable gap drift.
  Published MAE = counter error on observed data.

## What this does not show

- No VDV 457 statement is possible from 25 events (the standard's own
  methodology literature needs thousands, stratified).
- No low-light, no motion, no crowding (max 2 people simultaneously at the
  line; the back-to-back double exit pids 190/172 0.1 s apart produced the
  one shared miss — every variant merged the second person; simultaneous
  crossings remain the known hard case, consistent with the beam-break and
  APC literature).
- ToF noise is not D435i stereo noise; the simulation inherits the D435i's
  failure modes, not the VL53L7CX's (sunlight behaviour differs in kind).
- One door, one bus, one 32-minute staged session.

## Provenance

- Downloaded this run: 20,021 annotation/calibration files (13.8 MB on
  disk, ~7 MB transferred) + 2,215 depth frames (428.7 MB on disk,
  429.5 MB transferred, 1,778 range requests, 1,796 s, CRC32-verified per
  member, all 2,215 misnamed `.jpg` corrected to `.png` by magic-byte
  sniff). Total ~437 MB transferred; hard cap 0.5 GB respected. No PCDS
  bytes (Baidu-gated, unobtainable programmatically; see log).
- center_left_depth's own intrinsics (848x480, fx 426.6, zero distortion)
  from the archive's `camera_info.yaml`; extrinsics from `frame_ids.json` +
  `target_transforms.json`. Camera origin [-1.335, 0.398, 1.988] m,
  axis 67 deg off vertical.
- Licence: data CC-BY 4.0; counter written from the sensor model, no
  toolkit code.
