# Sanas hardware — power and data wiring

Written 2026-08-10 by CADy. Companion figure: `figures/wiring_schematic.svg`
(Figure H3). Only fixed ICs anywhere in this document: **VL53L7CX** and
**ESP32-S3**. Every other part (transceivers, converters, fuses, connectors,
SBC) is deliberately unnamed and TBD in `docs/trade_study.md` — no invented
part numbers.

---

## 1. Power architecture

Topology: **distribute 24 V, convert locally.**

```
vehicle 24 V DC (nominal, ASSUMED — verify with operator)
  └─ F1 (main fuse, rating TBD)
      └─ input protection: reverse-polarity + TVS transient clamp
         (automotive transient environment: ISO 7637-2 / ISO 16750-2
          apply — standards to be consulted, not yet reviewed)
          └─ 24 V distribution rail (in the 4-conductor trunk)
              ├─ F2 → buck 24→5 V → gateway (SBC, GPS, LTE; optional
              │                      12 V aux rail TBD)
              ├─ F3 → door 1 node: buck 24→5 V → LDO 3.3 V
              │        → ESP32-S3 + CAN transceiver + VL53L7CX
              └─ F4 → door 2 node: identical
```

Why distribute 24 V rather than a central 5 V supply: at 5 V the same power
means ~5× the current, so IR drop over a ~10 m run becomes significant, and a
single regulator is a single point of failure for all nodes. Local bucks per
node are standard automotive practice. (Qualitative reasoning, no numbers
invented; the actual power budget is TBD — node and gateway current draws
must come from the VL53L7CX and ESP32-S3 datasheets and the chosen SBC.)

Rails per node: buck 24→5 V (automotive input range, part TBD), then LDO to
3.3 V for the ESP32-S3 and the VL53L7CX carrier. The VL53L7CX's exact rail
configuration (AVDD/IOVDD) is **TBD from the datasheet**; carrier modules
commonly run from a single 3.3 V rail, which is what the schematic assumes.

Fusing: F1 protects the trunk at the vehicle tap; F2-F4 protect each branch.
**All ratings TBD** — they follow from the power budget, which follows from
the datasheet currents. Do not size fuses before that exists.

Grounding: single-point chassis ground at the gateway; the GND conductor is
carried in the trunk so all CAN transceivers share a common reference.
Chassis is not used as a return conductor.

## 2. Data wiring

- **Trunk:** one 4-conductor harness — 24 V, GND, CAN_H, CAN_L (CAN pair
  twisted; automotive-grade cable, type and gauge TBD after power budget).
- **Topology:** daisy-chain along the cabin ceiling: gateway → door 1 node →
  door 2 node. Stubs kept short (rule of thumb; max stub length at the chosen
  bitrate TBD).
- **Termination:** 120 Ω at both physical ends of the trunk (gateway end and
  the last node). Middle node unterminated.
- **Bitrate:** 500 kbit/s ASSUMED (see system_architecture.md §3 for the
  CAN-vs-RS485 tradeoff and the load arithmetic: ≈62 kbit/s for two nodes
  at the vendor-rated 15 Hz frame rate, ≈12.5% bus load).
- **Node interface:** ESP32-S3 TWAI controller (per Espressif family docs,
  UNVERIFIED — confirm) through a 3.3 V automotive CAN transceiver (part
  TBD). Gateway attaches via its own transceiver (USB/SPI CAN adapter or
  SBC-native, TBD with the SBC choice).
- **Phase 2 (dashed):** the salon RGB camera does not use the CAN trunk —
  bandwidth is orders of magnitude higher; it connects directly to the
  gateway (USB3 or GMSL, TBD) and is dormant until municipal permission.

## 3. Connectors and mechanics

All interconnects: automotive-grade, locking, vibration-rated — **specific
series TBD**. Requirements to carry into the trade study: positive lock,
strain relief at both ends, sealed variants at the door pillars (wash-down
and condensation), keyed so node and gateway drops cannot be swapped with
the power input.

## 4. Figure H3 caption — `figures/wiring_schematic.svg`

Power and data wiring for a 2-door bus. The 24 V vehicle supply (nominal,
ASSUMED) passes a main fuse F1 and an input-protection stage
(reverse-polarity + TVS; ISO 7637-2 to verify) onto a 24 V distribution rail.
Three fused branches (F2-F4, ratings TBD) feed local buck converters at the
gateway and at each door node; each node regulates further to 3.3 V for the
ESP32-S3, the CAN transceiver, and the VL53L7CX (I²C + INT/LPn to the MCU).
The CAN trunk (CAN_H/CAN_L twisted pair, carried in the same harness as
24 V/GND) daisy-chains gateway → node 1 → node 2 and is terminated 120 Ω at
both physical ends. The Phase-2 salon RGB camera connects to the gateway
directly (USB3/GMSL, TBD) and is drawn dashed. Only VL53L7CX and ESP32-S3 are
fixed parts; every rating, gauge and part number is TBD
(docs/trade_study.md). Intended for two-column width.

## 5. Assumptions register (this document)

ASSUMED: 24 V vehicle system; 500 kbit/s CAN; single-3.3 V-rail VL53L7CX
carrier; trunk length ~15 m for a 12 m bus (BOM line).

TBD: all fuse ratings; buck converter parts and currents; power budget
(datasheet-derived); wire gauges; CAN transceiver part; max stub length;
connector series; gateway SBC and its CAN attachment; RGB camera link
(USB3/GMSL); ISO 7637-2 / ISO 16750-2 compliance review; ESP32-S3 TWAI
confirmation.
