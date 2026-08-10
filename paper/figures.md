# Figure specification — Sanas APC paper (`paper/sanas_apc.tex`)

Owner: Danyshpan (numbering, captions, interpretation). Builder: CADy
(plotting code and SVG-to-PDF export). Written 2026-08-10.

Rules:
1. Figure numbers below are load-bearing: they match `\label`/`\ref` and the
   in-text order of first reference in `sanas_apc.tex`. Do not renumber
   without editing the .tex.
2. Result plots (Figs. 1-3) are generated **only** from
   `paper/results/apc_validation.json`. No numbers from anywhere else, no
   smoothing, no interpolation of the estimate series.
3. Captions are fixed and live in the .tex; they are reproduced here verbatim
   so the built figure can be checked against what the caption promises.
   CADy does not edit captions. If a figure cannot show what the caption
   says, flag it to Danyshpan instead of adapting the caption.
4. Output files go to `paper/figures/` with the exact filenames below
   (the .tex includes them by these names). Vector PDF preferred; if the
   toolchain needs an intermediate SVG/PNG, keep the PDF as the deliverable.
5. IEEE conventions: serif or neutral sans text, single-column figures
   ~3.5 in wide, double-column ~7.16 in wide, fonts no smaller than ~7 pt
   after scaling. No chartjunk, no title inside the plot (the caption is the
   title).

---

## Fig. 1 — Cumulative count N_t vs door-event truth (time series)

- File: `paper/figures/fig1_nt_series.pdf`
- Placement: double column (`figure*`), referenced in Sec. VI (Results).
- Type: step-line time series. Semantics per the JSON `definitions` block
  (authoritative); the spec below already conforms to it.
- Series (five step lines total, all on the same axes):
  - x: `nt.full.series_1hz.t_rel_s` (identical grid in all three variants,
    1 Hz, 0 to ~1941 s).
  - DOOR-EVENT TRUTH: `nt.full.series_1hz.door_truth` (NEW field). The
    25-transition staircase, range -4..+1, session end -1. Draw as the bold
    reference staircase. It goes negative by construction: counts are
    anchored at 0 at the first processed frame, and actors already aboard
    (never observed boarding) alight later. Do not clip, offset, or
    re-anchor it.
  - LINE-POSITION TRUTH: `nt.full.series_1hz.truth`. Persons on the cabin
    side of the line plane; 92 transitions, range 0..2; moves whenever
    anyone crosses the plane, door event or not. Draw as a THIN DOTTED
    line, visually subordinate to door_truth. It is in the figure to make
    the Sec. VI-C divergence visible (e.g. the wheelchair crossing around
    t+262 s moves it while door_truth stays flat).
  - Estimates: `nt.full.series_1hz.estimate`, `nt.mz8.series_1hz.estimate`,
    `nt.mz4.series_1hz.estimate`. RAW cumulative counts anchored at 0 (not
    errors — do not subtract anything). Step lines; emphasise mz8, keep
    full and mz4 thinner/lighter. Estimates are frozen outside processed
    chunks; plot as-is, no masking or interpolation. Expected endpoints:
    mz8 0, full -4, mz4 +2 (sanity check against the JSON, which wins).
  - Episode markers: from `variants.full.per_episode[*]` take `start_ns`,
    `kind`. Convert to seconds on the same t_rel axis (t_rel = (start_ns -
    t0_ns)/1e9; t0 as established in the first build — if ambiguous, ask
    Danyshpan rather than guessing). Small up-triangles for `board`,
    down-triangles for `alight`, along the top edge.
- Axes: x "time since session start (s)", y "on-board count N_t (anchor 0)".
  y-ticks integer only (data range -4 to +2 plus headroom for markers).
- Annotations: right-edge endpoint labels are optional; if drawn, label the
  RAW endpoints (door truth -1; mz8 0; full -4; mz4 +2). Do NOT annotate
  the final errors (+1/-3/+3) inside the plot; they live in the caption and
  Table II.
- Legend: "door-event truth", "line-position truth", "full 848x480",
  "8x8 sim", "4x4 sim".
- Caption (verbatim, from .tex): "Cumulative on-board count N_t at 1 Hz for
  the full-resolution pipeline and the 8x8 and 4x4 zone simulations,
  against two references: door-event truth (bold staircase; the 25 labelled
  episodes applied at episode midpoints) and line-position truth (dotted;
  persons on the cabin side of the line plane, which moves whenever anyone
  crosses it, door event or not). All counts are anchored at zero at the
  first processed frame, so door-event truth dips to -4: actors already
  aboard, whose boarding was never observed, alight later. Markers on the
  upper edge give labelled boarding (up) and alighting (down) episodes;
  estimates freeze outside processed chunks. Session-end door-event truth
  is -1 against estimate endpoints of 0 (8x8), -4 (full), and +2 (4x4),
  giving the final errors +1, -3, +3 of Table II. MAE values are computed
  over the 2,167 processed frames, not this 1 Hz display grid."

## Fig. 2 — Resolution ablation summary

- File: `paper/figures/fig2_resolution_ablation.pdf`
- Placement: single column, referenced in Sec. VI.
- Type: two stacked panels (a)/(b), point estimates with error bars. No bars
  without intervals; the intervals are the point of this figure.
- Data (per variant full / mz8 / mz4):
  - Panel (a): detection rate with Wilson 95% CI.
    `variants.<v>.detection_rate`, `variants.<v>.detection_ci95`.
    Values: full 0.76 [0.5657, 0.8850]; mz8 0.96 [0.8046, 0.9929];
    mz4 0.84 [0.6535, 0.9360]. Annotate each point "19/25", "24/25",
    "21/25".
  - Panel (b): N_t MAE vs door-event truth with bootstrap 95% CI.
    `nt.<v>.mae_vs_door_truth`, `nt.<v>.mae_vs_door_truth_ci95`.
    Values: full 1.454 [1.082, 1.908]; mz8 0.174 [0.109, 0.261];
    mz4 1.92 [1.47, 2.37] (rounded; read the exact mz4 values from
    `nt.mz4.*` in the JSON, which is authoritative). Per the JSON
    `definitions` block these MAEs are computed over the 2,167 processed
    frames, not the 1 Hz display grid; do not recompute MAE from the
    series_1hz arrays.
- Axes: x categorical (full 848x480, 8x8, 4x4); (a) y "episodes detected
  (fraction)" range 0.5-1.0; (b) y "N_t MAE (persons)" starting at 0.
- Caption (verbatim): "Resolution ablation on n = 25 episodes. (a) Episode
  detection rate with Wilson 95% intervals. (b) Cumulative-count MAE against
  door-event truth with block-bootstrap 95% intervals. The 8x8 simulation
  outperforms full resolution under identical untuned parameters; every
  episode is worth 4 percentage points, so the intervals are the honest
  summary."

## Fig. 3 — Counting-band width sweep

- File: `paper/figures/fig3_band_sweep.pdf`
- Placement: single column, referenced in Sec. VI.
- Type: line + markers, detection fraction vs band width.
- Data: `band_ablation.<w>` for w in {1.0, 0.7, 0.5, 0.35, 0.25}.
  x = band width 2w in metres: 2.0, 1.4, 1.0, 0.7, 0.5.
  y = `true_positives`/25: 19, 20, 19, 19, 16 → 0.76, 0.80, 0.76, 0.76,
  0.64. Annotate each marker with "k/25". Wilson CIs are in
  `band_ablation.<w>.detection_ci95`; draw them as a light band or error
  bars (preferred: error bars).
- Annotations:
  - vertical reference line at x = 0.7 m labelled to the effect of
    "±0.35 m head-height strip, untilted ceiling node (60°x60° FoV)" (the
    node-design connection; VL53L7CX FoV confirmed 60°x60° square / 90°
    diagonal, see docs/hardware/node_design.md — the earlier ±0.25 m at
    0.5 m came from a superseded 60°-diagonal assumption; exact short
    label wording up to CADy's layout, meaning fixed).
  - x-axis descending from 2.0 to 0.5 (narrowing band reads left to right).
- Axes: x "counting-band width along travel direction (m)", y "episodes
  detected (fraction of 25)".
- Caption (verbatim): "Detection against counting-band width
  (full-resolution path, zone truncated to |x - x_line| <= w; band width
  2w). Detection is flat down to a 0.7 m band and collapses at 0.5 m, where
  the ±0.15 m hysteresis leaves only 0.1 m of visible approach per side.
  The reference line at 0.7 m marks the ±0.35 m head-height strip of an
  untilted ceiling-mounted node (60°x60° field of view), which sits exactly
  at the measured working minimum with no margin."

---

## Figs. 4-7 — CADy's hardware figures (existing SVGs, export to PDF)

Cross-reference of CADy's internal H-numbers to paper figure numbers:

| Paper figure | CADy internal | Source SVG | Target PDF | Placement |
|---|---|---|---|---|
| Fig. 4 | H1 | `docs/hardware/figures/system_architecture.svg` | `paper/figures/fig4_system_architecture.pdf` | single column |
| Fig. 5 | H2 | `docs/hardware/figures/sensor_placement.svg` | `paper/figures/fig5_sensor_placement.pdf` | double column |
| Fig. 6 | H3 | `docs/hardware/figures/wiring_schematic.svg` | `paper/figures/fig6_wiring_schematic.pdf` | double column |
| Fig. 7 | H4 | `docs/hardware/figures/dataflow.svg` | `paper/figures/fig7_dataflow.pdf` | double column |

All four are referenced in Sec. VII (System Design). Export requirements:
text converted to paths or fonts embedded; check legibility at IEEE column
width; ASSUMED/TBD markers already inside the SVGs must remain visible
(they are part of the honesty framing, do not clean them out).

Paper captions for Figs. 4-7 (fixed in the .tex; they supersede the longer
H1-H4 caption text in `docs/hardware/*.md` for the paper only — the hardware
docs keep their own):

- Fig. 4: "Per-bus system architecture. Two door nodes (VL53L7CX on I2C to
  an ESP32-S3) stream raw 8x8 frames over a terminated CAN 2.0 trunk to a
  gateway running the geometric counting engine, GPS terminus reset, and LTE
  uplink to the transit backend. The Phase-2 RGB drift corrector (dashed) is
  designed but not trained."
- Fig. 5: "Proposed sensor placement, side (A) and top (B) views of an
  assumed 12 m two-door low-floor bus. Each door carries a face-down ToF
  node above the lintel; the virtual counting line lies inside the vestibule
  within the 1.2 to 2.2 m band measured in Sec. V-A, and the counting-band
  result of Fig. 3 constrains the mount offset or tilt. All dimensions are
  parametric assumptions pending a real vehicle survey."
- Fig. 6: "Power and data wiring for a two-door installation. A fused 24 V
  trunk feeds local buck converters at the gateway and at each node; the CAN
  pair is carried in the same four-conductor harness and terminated at both
  physical ends. Only the two named ICs are fixed; every rating, gauge, and
  part number is an open item (Sec. VIII)."
- Fig. 7: "Data flow with derivable rates. Each node produces 64 x 2 B =
  128 B per frame; at the vendor-rated 15 Hz this is approximately
  15.4 kbit/s per node before CAN framing, about 12.5% of a 500 kbit/s bus
  for two nodes. Raw frames are retained at the gateway as licence-clean
  training data. The Phase-2 ordinal corrector enters dashed."

---

## Tables (for completeness; Danyshpan builds these directly in LaTeX)

- Table I — cabin-wide BEV occupancy score by ground-truth count
  (source: findings.md §5 kernel output). Already in the .tex.
- Table II — headline validation results across resolutions
  (source: apc_validation.json `variants` + `nt`). Already in the .tex.

## Explicitly not figures

- The negative-result correlation (r = 0.184) stays in text + Table I; its
  underlying per-frame data is not in apc_validation.json and a figure would
  require re-running the occupancy kernel. Decision: no figure.
- The 15 Hz vs 7.7 Hz rate spot check (3 episodes, identical events) stays
  in text; three points do not support a figure.
