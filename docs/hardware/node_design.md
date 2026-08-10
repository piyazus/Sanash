# Sanas hardware — ToF door-node design (variant b)

Written 2026-08-10 by CADy. The node is the paper's novelty claim: a
low-cost 64-zone ToF APC node — **ST VL53L7CX** (8×8 zones) on an
**ESP32-S3** carrier — mounted face down above each door, replacing a
RealSense-class depth camera (variant a). Donatello is validating the 8×8
resolution by simulation in parallel; nothing in this document has been
validated on hardware.

---

## 1. Function

Stream raw 8×8 depth frames at the sensor's maximum rate to the gateway over
CAN. The node does **no counting** in the MVP — see §5 for why.

## 2. Electrical design

- **Sensor:** VL53L7CX, 8×8 = 64 ranging zones, FoV 60°×60° square
  (90° diagonal) — CONFIRMED from ST product documentation via
  docs/trade_study.md (was ASSUMED 60° diagonal in the first draft)
  (**ASSUMED** until the trade study confirms the datasheet figure).
  Interface: I²C to the ESP32-S3, plus INT (frame-ready), LPn and reset
  GPIOs. I²C speed and the sensor's max ranging frequency in 8×8 mode:
  **TBD from the VL53L7CX datasheet**. Rail configuration (AVDD/IOVDD):
  TBD from the datasheet; the wiring schematic assumes a single-3.3 V
  carrier module.
- **MCU:** ESP32-S3. Roles: sensor init over I²C, frame capture on INT,
  framing onto CAN via the on-chip TWAI controller (per Espressif family
  docs, **UNVERIFIED** — confirm against the datasheet) through an external
  3.3 V CAN transceiver (part TBD). Wi-Fi/BLE stays off in normal operation;
  Wi-Fi is an option for bench flashing only (§5).
- **Power:** 24 V from the trunk → local buck 24→5 V → LDO 3.3 V. Currents
  and fuse rating TBD (see wiring.md §1).

## 3. Mounting geometry (all parametric)

Orientation: **sensor face down**, boresight vertical, above the door lintel
on the cabin side. Optional tilt θ_tilt toward the cabin (default 0, TBD).

Derived coverage arithmetic — FoV now CONFIRMED (60°×60° square per axis,
ST documentation via docs/trade_study.md); H_ceil and head height remain
ASSUMED, so treat the outputs as sizing guides, not spec:

- Per-axis half-angle = 30°, tan(30°) ≈ 0.577.
- Floor footprint side at mount height H_ceil = 2.3 m (ASSUMED):
  F = 2 · H_ceil · tan(30°) ≈ **2.66 m** (half-extent ≈ 1.33 m).
- Coverage half-width shrinks with target height z:
  w(z) = (H_ceil − z) · tan(30°). At head height z = 1.7 m (ASSUMED):
  w ≈ **0.35 m**, i.e. a ±0.35 m strip (0.7 m band).

Measured cross-check (experiments/log.md 2026-08-10, counting-band
ablation): detection stays flat down to a 0.7 m band and collapses at
0.5 m. The head-height strip of a vertically mounted node (0.7 m) sits
**exactly at the measured working minimum** — no margin. Earlier draft
computed ±0.25 m from an incorrectly assumed 60° diagonal; that value is
superseded.

Consequence, stated honestly: the counting line sits inside the vestibule at
d_line from the door plane, within the 1.2-2.2 m band (Finding C,
experiments/log.md 2026-08-10). With a vertical boresight the node must
therefore be mounted **inboard of the door plane by s_off ≥ d_line − F/2**
(at floor level), and at head height the covered strip around the line is
±0.35 m with zero margin against the measured 0.7 m minimum, so either
s_off ≈ d_line, or a modest tilt θ_tilt, or accepting torso/feet-level
detection is required. Alternatives if the
structure above the vestibule prevents inboard mounting: tilt the boresight
(θ_tilt > 0) or accept feet-level detection. Which of these wins is exactly
what Donatello's simulation should settle; the parameters (H_ceil, d_line,
s_off, θ_tilt, FoV) are all exposed, none is final.

Also open: the VL53L7CX's effective ranging distance at a 2.3 m mount with
bus-interior reflectance — **TBD from the datasheet/trade study**; if 2.3 m
is marginal, the mount drops below the ceiling line, which the parametric
design absorbs.

## 4. Enclosure concept

- Small ABS box above the lintel, sensor aperture facing down.
- **The sensor cannot sit behind an arbitrary window.** ToF through a cover
  degrades unless the material and air gap follow ST's cover-window
  guidance: either an open aperture or an IR-transmissive window per the ST
  application note (note reference and material spec: TBD). This is a real
  failure mode, not a finish detail.
- Vented against condensation (buses are washed and see large temperature
  swings); IP target TBD.
- Vibration: locking connectors, threadlocker on mounting hardware, the PCB
  mounted on standoffs, no component relying on friction fit. Relevant
  standard to consult: ISO 16750-3 (not yet reviewed).
- No user-visible optics pointing along the cabin: the node looks straight
  down at the vestibule floor, which is also the privacy argument — 64 depth
  zones cannot image a face.

## 5. Firmware responsibilities

1. Init: configure 8×8 mode at the sensor's **max ranging rate (TBD from
   datasheet)**; self-test at boot (frame received within timeout, plausible
   range distribution) and report status.
2. Capture: on INT, read the 64-zone frame — per-zone distance (2 B) and,
   optionally, per-zone target status (1 B).
3. Stream: fragment onto CAN and send. No local buffering beyond a few
   frames; the gateway timestamps on receipt (avoids node clock sync;
   receipt-jitter contribution TBD).
4. Heartbeat: 1 Hz status frame (uptime, sensor state, temperature if
   available) so the gateway can flag a dead node immediately.
5. Watchdog reset; sensor re-init on repeated frame timeouts.
6. OTA: option via CAN bootloader or bench Wi-Fi (ESP32-S3), TBD — not
   MVP-critical since the node contains no algorithm to update.

**Recommendation: raw 8×8 streaming to the gateway, no on-node
line-crossing.** Reasons:

- The crossing algorithm stays updatable centrally — one gateway deployment
  updates the counting logic for every door, and no dual implementation
  (node firmware vs gateway) can drift apart. Same anti-drift argument that
  motivated `scripts/build_kernels.py` (log 2026-08-10).
- Raw frames are the licence-clean training/validation corpus this project
  otherwise lacks: PCDS is CC BY-NC-SA and cannot supply production weights,
  so our own recordings are the only path to shippable weights
  (log 2026-08-10, PCDS entry). Streaming-and-logging collects them as a
  side effect of operation.
- Bandwidth is trivial at 64 zones: 64 × 2 B = 128 B/frame; at f = 15 Hz
  (**ASSUMED**, f TBD from datasheet) ≈ 15.4 kbit/s payload per node,
  ≈ 31 kbit/s after CAN framing; two nodes together ≈ 12.5% of a
  500 kbit/s bus (arithmetic in system_architecture.md §3).
- On-node preprocessing (crossing events only) would cut this to a few
  bytes per event, which matters for none of the above and freezes the
  algorithm into firmware. Revisit only if the bus load measurement ever
  says otherwise.

## 6. Assumptions register (this document)

CONFIRMED: FoV 60°×60° square (90° diagonal), 15 Hz max at 8×8 (ST docs
via trade study). ASSUMED: f = 15 Hz in worked examples; H_ceil 2.3 m;
head height 1.7 m; single-3.3 V carrier; 500 kbit/s CAN.

TBD (docs/trade_study.md unless noted): VL53L7CX max ranging rate, I²C
speed, rail configuration, effective range at 2.3 m, cover-window material
per ST app note (datasheet/app-note reads); ESP32-S3 TWAI confirmation
(Espressif datasheet); CAN transceiver part; node power draw and fuse
rating; d_line, s_off, θ_tilt (Donatello's simulation + PCDS validation);
enclosure IP rating; ISO 16750-3 review; OTA mechanism; gateway receipt
jitter.
