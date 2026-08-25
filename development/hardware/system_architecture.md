# Sanas hardware — system architecture (door-mounted 3D APC)

Written 2026-08-10 by CADy. Status: concept documentation of the decided
architecture. Nothing here is procured, built, or validated on hardware.
Figure numbering is internal to the hardware docs (H1-H4); paper figure
numbers and captions belong to Danyshpan.

Provenance rule as in `development/findings.md`: every number is either measured
(cross-referenced to findings.md / development/experiments/log.md), or marked ASSUMED
(reasonable default, replace with real data) or TBD (must be sourced, see
`research/trade_study.md`). The assumptions register at the end of this file
collects all of them.

---

## 1. Measured basis (why this architecture)

Decided by the coordinator; the measured record behind the decision:

1. **Cabin-wide depth occupancy was rejected on evidence.** One in-cabin
   depth camera covers 10.0% of the region occupants actually use, correlates
   with ground-truth count at Pearson r = 0.184 (95% CI [0.133, 0.231]), and
   the score is non-monotonic past two occupants (findings.md §5). The
   preliminary four-camera union reaches only 28.8% at spec range
   (findings.md §5, log 2026-08-10).
2. **Door-mounted 3D APC is the industry-standard shape** (iris IRMA MATRIX /
   INIT / Dilax class) and the MVP defers incident detection, so cabin-wide
   coverage is over-built for what the product must do (log 2026-08-10, pivot
   entry).
3. **A true APC looks straight down** — none of the substitute-dataset
   cameras did (Finding A), which is why our design mounts the sensor face
   down above each door.
4. **The counting line goes inside the vestibule, not across the aperture.**
   At the labelled crossing instant people are still on the kerb, outside the
   depth window; all 25/25 boarding/alighting trajectories pass through the
   in-cabin zone at 1.22-2.24 m (Finding C, log 2026-08-10).
5. **The geometric APC path is CPU-class.** An accelerator (Jetson-class) is
   justified only by the Phase-2 CNN, and the argument there is TensorRT
   toolchain maturity, not throughput (findings.md §7).

## 2. Architecture overview

Per 12 m two-door bus (bus platform parameters ASSUMED, see register):

- **2 × door sensor node.** Downward-looking depth sensor above each door
  aperture. Two variants:
  - (a) 3D depth camera module, RealSense-class (module TBD, trade study);
  - (b) low-cost 64-zone ToF node: ST VL53L7CX (8×8 zones) on an ESP32-S3
    carrier — the baseline for the BOM and the paper's novelty claim.
    Donatello is validating variant (b) by simulation in parallel.
- **In-vehicle bus:** CAN 2.0 trunk (choice justified in §3) carrying data
  plus 24 V power in one 4-conductor harness.
- **1 × gateway** (CPU-class SBC, board TBD): frame ingest and raw logging,
  virtual-line crossing detection, cumulative count
  N_t = N_(t-1) + boardings - alightings, GPS terminus reset, LTE uplink.
  No Kalman filter or sensor fusion in the MVP (log 2026-08-10, pivot entry).
- **LTE uplink** to the Avtobys backend (Innoforce; API contract TBD), which
  serves crowding info to the rider app.
- **Phase 2 (dashed in all figures, dormant):** one wide-angle salon RGB
  camera feeding a DINOv2/ConvNeXt-Tiny + CORN ordinal head as a drift
  corrector for the cumulative count. Dormant until municipal camera
  permission; the model assets carry the same DORMANT banner in `development/src/sanas/`
  (log 2026-08-10).

## 3. In-vehicle bus: CAN, not RS485

Decision: **CAN 2.0** at 500 kbit/s (bitrate ASSUMED, to be fixed in the
trade study). Reasoning:

| Criterion | CAN 2.0 | RS485 |
|---|---|---|
| Vehicle environment | native automotive bus: differential, error counters, automatic retransmit, fault confinement | differential, robust, but no link-layer error handling — you write it |
| Protocol effort | link layer exists; nodes push frames asynchronously (multi-master arbitration) | half-duplex master-slave; custom framing, addressing, CRC and polling schedule to design, implement and debug |
| MCU support | ESP32-S3 has an on-chip TWAI (CAN 2.0) controller per Espressif family documentation — **UNVERIFIED here, confirm against the datasheet**; only an external transceiver is needed | UART + transceiver, also simple |
| Payload fit | 8 B/frame: a 128 B depth frame fragments into 16 frames — acceptable at these rates (arithmetic below) | arbitrary frame length, marginally simpler |
| Bandwidth needed | ≈31 kbit/s per node incl. framing (≈62 kbit/s for 2 nodes, ≈12.5% of a 500 kbit/s bus) — trivial either way | same, trivial |
| Cable length | rule of thumb ~40 m at 1 Mbit/s, longer at lower bitrates (ASSUMED, not verified) — a 12 m bus is comfortably inside | longer runs possible, irrelevant at 12 m |

The deciding factors are the automotive pedigree and not having to write and
debug a custom polling protocol; RS485's advantages (cable length, payload
size) don't matter at this scale. Unverified assumptions in this choice:
ESP32-S3 TWAI availability and maximum bitrate; 500 kbit/s bitrate;
bus-length rule of thumb; transceiver part (TBD, must be a 3.3 V automotive
part).

Bandwidth arithmetic (also shown in figure H4): 64 zones × 2 B = 128 B per
depth frame; at f = 15 Hz (vendor-rated max at 8×8,
confirmed — node_design.md §6) that is 1,920 B/s ≈ 15.4 kbit/s payload per node.
Classic CAN framing ≈ ×2 overhead (16 frames × ≈130 bits worst-case stuffed,
estimate) → ≈31 kbit/s per node, ≈62 kbit/s for two nodes.

## 4. Figures

### Figure H1 — `figures/system_architecture.svg`

Block diagram of the per-bus system. Two door sensor nodes (variant (b):
VL53L7CX on I²C to an ESP32-S3; variant (a) noted as a dashed alternative)
attach to a terminated CAN 2.0 trunk; the gateway carries the CAN interface,
the geometric APC engine (line crossing → boarding/alighting events →
cumulative count N_t), the GPS-geofence terminus reset, and the LTE uplink to
the Avtobys backend and rider app. The Phase-2 path — salon RGB camera and
the ordinal drift corrector (DINOv2/ConvNeXt-Tiny + CORN) — is drawn dashed
and is dormant until municipal camera permission. Footer cites the measured
basis (10% coverage, r = 0.184, findings.md §5). Intended for single-column
width.

### Figure H2 — `figures/sensor_placement.svg`

Side view (A) and top view (B) of the ASSUMED 12 m low-floor bus with two
double doors. All dimensions are parametric and collected in the in-figure
parameter box (L_bus = 12 m, W_bus = 2.55 m, H_ceil ≈ 2.3 m, H_door ≈ 2.0 m,
W_door ≈ 1.25 m, door positions x_d1/x_d2 — all ASSUMED). Each door carries a
face-down ToF node above the lintel with its 60°×60° FoV cone (confirmed, 90°
diagonal) reaching a floor footprint of F ≈ 2.7 m side length (derived
from H_ceil; equations in node_design.md). The virtual
counting line is drawn inside the vestibule within the hatched 1.2-2.2 m band
from the door plane (Finding C, log 2026-08-10); its exact offset d_line is
TBD, and the node must be mounted inboard by s_off ≥ d_line − F/2 for the
footprint to cover the line. The Phase-2 wide-angle RGB camera and its FoV
are dashed. Intended for two-column width.

### Figure H3 — `figures/wiring_schematic.svg`

Power and data wiring; caption lives in `wiring.md`.

### Figure H4 — `figures/dataflow.svg`

Data flow with every derivable rate shown as explicit arithmetic. Per node:
64 zones × 2 B = 128 B/frame at f = 15 Hz (vendor-rated max at 8×8,
node_design.md §6) → ≈15.4 kbit/s payload, ≈31 kbit/s after CAN framing. The gateway fusion loop chains ingest + raw logging,
virtual-line crossing (line inside the vestibule per Finding C), B/A events,
the cumulative counter with GPS terminus reset, and the count-to-level map
whose denominator (cabin capacity) is an open product decision
(findings.md §8). Raw 8×8 frames are retained on the gateway as future
licence-clean training data, because PCDS is CC BY-NC-SA and cannot supply
production weights (log 2026-08-10). Uplink report ≈100 B (ASSUMED), cadence
per stop or 30 s (TBD). The Phase-2 ordinal corrector enters dashed. Intended
for two-column width.

## 5. Assumptions register (this document and its figures)

ASSUMED (reasonable default, replace with real data):
- 12 m low-floor city bus, 2 double doors; W_bus 2.55 m; H_ceil ≈ 2.3 m;
  H_door ≈ 2.0 m; W_door ≈ 1.25 m; door positions x_d1 ≈ 0.9 m,
  x_d2 ≈ 5.9 m from the front bumper.
- Vehicle electrical system 24 V DC nominal.
- CAN bitrate 500 kbit/s; CAN framing overhead ≈ ×2; bus-length rule of
  thumb.
- Uplink report size ≈100 B.

CONFIRMED since first draft (via node_design.md §6 / trade study):
- VL53L7CX FoV 60°×60° square (90° diagonal); max ranging rate 15 Hz at
  8×8. The earlier 60°-diagonal ASSUMED figure is superseded.

TBD (see research/trade_study.md unless noted):
- VL53L7CX effective range at a 2.3 m mount (reflectance-dependent).
- ESP32-S3 TWAI controller and max bitrate (Espressif datasheet).
- CAN transceiver part, gateway SBC board, LTE modem, GPS module.
- Variant (a) depth camera module, its interface and power.
- Avtobys backend API contract (Innoforce).
- GPS terminus-geofence radius (product).
- Count-to-level denominator = cabin capacity (product decision,
  findings.md §8).
- d_line exact placement inside the 1.2-2.2 m band; s_off; uplink cadence.
