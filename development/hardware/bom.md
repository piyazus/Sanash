# Sanas hardware — bill of materials (2-door bus, variant b baseline)

Written 2026-08-10 by CADy. Quantities are for one 12 m two-door bus
(platform ASSUMED). **No prices in this file by design** — Donatello sources
prices in parallel; every price cell points to `research/trade_study.md`.
The only fixed part numbers are the two named ICs (VL53L7CX, ESP32-S3);
everything else is a role with a TBD part.

## Phase 1 — door APC (variant b baseline)

| # | Component | Qty | Role | Price |
|---|---|---|---|---|
| 1 | ST VL53L7CX ToF module (8×8, 64 zones) | 2 | per-door depth sensing, face down | see research/trade_study.md |
| 2 | ESP32-S3 module on carrier board | 2 | node MCU: I²C capture, CAN streaming | see research/trade_study.md |
| 3 | CAN transceiver, 3.3 V, automotive (part TBD) | 3 | bus PHY: 2 nodes + gateway | see research/trade_study.md |
| 4 | Buck converter 24→5 V, automotive input (part TBD) | 3 | local supplies: 2 nodes + gateway | see research/trade_study.md |
| 5 | LDO 3.3 V (part TBD) | 2 | node logic + sensor rail | see research/trade_study.md |
| 6 | Node enclosure, ABS, vented, IR-safe aperture (TBD) | 2 | above-door housing (node_design.md §4) | see research/trade_study.md |
| 7 | Gateway SBC, CPU-class (board TBD, findings.md §7) | 1 | APC engine, raw log, uplink | see research/trade_study.md |
| 8 | Gateway CAN attachment (USB/SPI adapter or native, TBD) | 1 | trunk interface | see research/trade_study.md |
| 9 | GPS receiver + antenna (module TBD) | 1 | terminus-reset geofence | see research/trade_study.md |
| 10 | LTE modem + antenna (module TBD) | 1 | uplink to Avtobys backend | see research/trade_study.md |
| 11 | Storage for raw frame log (SSD/SD, size TBD) | 1 | licence-clean data collection (node_design.md §5) | see research/trade_study.md |
| 12 | Fuses F1-F4 + holders (ratings TBD after power budget) | 1 set | supply protection | see research/trade_study.md |
| 13 | Input protection: TVS + reverse-polarity (parts TBD) | 1 | 24 V transient clamp (ISO 7637-2 to verify) | see research/trade_study.md |
| 14 | Trunk harness, 4-conductor with twisted pair (TBD) | ~15 m (ASSUMED) | 24 V + GND + CAN_H/L | see research/trade_study.md |
| 15 | Termination resistors 120 Ω | 2 | CAN trunk ends | see research/trade_study.md |
| 16 | Connectors, automotive-grade locking (series TBD) | 1 set | all interconnects | see research/trade_study.md |
| 17 | Mounting brackets + hardware, door lintel (TBD) | 2 sets | node mounting, s_off adjustable | see research/trade_study.md |
| 18 | Gateway enclosure + mounting (TBD) | 1 | driver-area or equipment-bay install | see research/trade_study.md |

## Variant (a) — alternative sensor, replaces rows 1-2 per door

| # | Component | Qty | Role | Price |
|---|---|---|---|---|
| A1 | 3D depth camera module, RealSense-class (module TBD) | 2 | per-door depth sensing | see research/trade_study.md |
| A2 | Camera-to-gateway link (USB3 or other, TBD — CAN trunk insufficient for depth-camera bandwidth) | 2 | data path | see research/trade_study.md |

Note: variant (a) bypasses the CAN data path for sensing (power distribution
unchanged) and removes the per-node MCU; it is the fallback if the 64-zone
simulation fails, and the comparison baseline in the paper.

## Phase 2 — dormant, do not procure

| # | Component | Qty | Role | Price |
|---|---|---|---|---|
| P1 | Salon RGB camera, wide-angle, ×1 (module, FoV TBD) | 1 | ordinal drift corrector input; dormant until municipal camera permission | see research/trade_study.md |
| P2 | Accelerator (Jetson-class) — only if the Phase-2 CNN activates; argument is TensorRT maturity, not throughput (findings.md §7) | 0-1 | CNN inference | see research/trade_study.md |
