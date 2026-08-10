# CANCELLED 2026-08-10 — inactive branch, do not push

Cabin-wide multi-camera occupancy. **Never pushed, never ran.**

## Why

The product MVP deferred incident detection (safety) to a later phase. Cabin-wide
multi-sensor coverage only pays for itself if you need to see behaviour anywhere in
the vehicle. For counting alone, the industry-standard answer is a door-mounted 3D
APC (iris GmbH IRMA MATRIX, INIT, Dilax), which watches a ~1x2 m aperture instead of
a whole saloon.

A second, independent reason: the rig cannot cover the cabin anyway. Measured on a
30-frame local sample, the four depth cameras see only **28.8%** of the floor area
occupants actually use within the D435i 0.3-3.0 m window, and only **63.3%** even
with a 10 m gate. The kernel's step-1 gate would have stopped on that.

## Status

Code is retained, not deleted. If the safety track returns, cabin-wide occupancy is
the right shape for it and this is the starting point — but it would need more
cameras, or better-placed ones, or a longer-range sensor.

## Superseded by

Door-zone APC work. See `experiments/log.md`, entry 2026-08-10.
