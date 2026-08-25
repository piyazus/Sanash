# Related-work notes (coordinator, 2026-08-10)

Source: Consensus academic search (two queries). All entries below are from
abstracts unless marked otherwise — Danyshpan must verify against full text
before citing, and must not copy accuracy numbers without checking the paper.
Consensus result URLs are kept for retrieval; final citations need the real
venue/DOI.

## What this does to our novelty claim

Low-resolution ToF people counting is NOT new in general:

- Lu et al. 2021 (Energy and Buildings): sparse array of low-res ToF sensors
  for zone-level occupancy counting in offices, reported error rate ~0.4%.
  https://consensus.app/papers/details/e499f8462d2a5f0f9af2ae01fb5f17ba/
- Jeong et al. 2025 (ETRI Journal): ToF camera above a doorway, mean-shift
  clustering, >90% accuracy in single-entry/exit scenarios, privacy framing
  identical to ours. https://consensus.app/papers/details/2ef4f2344cbc5df79b686542e367b124/
- Stec et al. 2019 (DASIP): ToF people counting explicitly for public
  transport, embedded processing. https://consensus.app/papers/details/f6c3215f3c5f539c959952b088efc4c7/
- VL53L5CX itself has 2026 characterization papers (Caroleo et al., Sensors;
  Kalvodova et al.) — the sensor is known to research, mostly robotics.
  https://consensus.app/papers/details/589ee630f31858569292ed0a0e16b856/
- Papanashi et al. 2026 (IEEE Access): low-res THERMAL array + ESP32 counting,
  the closest "cheap grid sensor on microcontroller" system.
  https://consensus.app/papers/details/8042b2f0480e5fe4b563f7d169ae0a20/

Depth-based APC at transit doors is industry standard and well published:

- Seidel et al. 2021, NAPC (IEEE OJ-ITS): LSTM on low-res 3D LiDAR depth over
  TRAIN doorways, ~13,000 annotated recordings, ~96% correct door-phase
  counts, counts 0-67. Public privacy-friendly dataset.
  https://consensus.app/papers/details/be89ef2d8df85a7e9b771fd735f54af8/
- Seo et al. 2026, CLAPC (IEEE OJ-ITS): CNN-LSTM, ToF depth above train doors,
  99.79%/99.97% boarding/alighting; explicitly states meeting the +-1%
  requirement of **VDV Guideline 457** (this is the acceptance standard our
  trade study must reference; the +-1% figure now has a citable source).
  Also does spatial down-sampling of 2D video for privacy — partial overlap
  with our resolution-ablation idea, but on grayscale video, not depth grids.
  https://consensus.app/papers/details/388aa1ec3d56565ea4265315b7fd5b56/
- Yahiaoui et al. 2010 (J. Electronic Imaging): dense stereo over bus door,
  99%/97% on two datasets. https://consensus.app/papers/details/eb5dd65da290548cb8b03d6877a7cb02/
- Sun et al. 2018 (IEEE T-ITS): PCDS — public RGB-D dataset, >4,500 videos at
  BUS entrance doors. **Candidate future validation data for our counter —
  flag to Donatello for a later run.**
  https://consensus.app/papers/details/6a8012616f5e51c98b8a76732c7bafae/
- Pronello et al. 2023 (Sensors): commercial video APC claimed 98% by vendor,
  measured 53-55% in a real field trial; their RPi+YOLOv5 got 72-75%. Key
  honesty citation for the gap between vendor claims and field performance.
  https://consensus.app/papers/details/1cd73e6562255fd79564ed3af4265e32/
- Liu et al. 2023: 3D LiDAR + RGB APC for sightseeing trams; notes ToF and
  mmWave explored, LiDAR underexplored.
  https://consensus.app/papers/details/2a80c510a10c52aa98510c969841d479/
- Nurseitov et al. 2026 (J. Imaging): YOLOv8+DeepSORT passenger counting,
  Kazakhstan group — regionally relevant citation.
  https://consensus.app/papers/details/116add0819c855e899c51e43df2133c5/

## Resulting honest novelty framing (for the paper)

NOT claimable: "first low-res ToF people counter" (buildings did it), "first
depth APC at transit doors" (industry + NAPC/CLAPC).

Claimable, pending our results:
1. Feasibility of a **single-chip 64-zone ToF (VL53L7CX-class) as a bus APC**,
   i.e. the extreme low-cost end (~$7 sensor vs commercial APC units), via a
   **resolution-ablation study** (full depth camera -> 8x8 -> 4x4) on real
   transit-cabin data. CLAPC downsampled grayscale video; nobody in the found
   set ablates depth resolution down to single-chip multizone grids for door
   counting.
2. **Vestibule-interior virtual-line placement**: measured finding that an
   oblique in-cabin rig cannot reproduce top-down aperture geometry, and the
   line must move inside the cabin (Finding C in development/experiments/log.md) — a
   deployment-geometry result the APC literature (which assumes top-down
   mounting) does not cover.
3. The measured **negative result** for cabin-wide oblique depth occupancy
   (10% coverage, r=0.184, non-monotonic — development/findings.md) as evidence FOR
   door-based counting over cabin-wide depth sensing.
4. System integration: terminus-reset cumulative count + ordinal 5-level
   crowding output for a rider-facing app (Avtobys), with Phase-2 RGB ordinal
   head as drift corrector — design contribution, clearly marked unvalidated.

## Verification status

All of the above: inferred-from-abstract via Consensus. Scite is out of monthly
quota until 2026-09-01, so full-text verification must go through the papers
themselves (arXiv/OA links) before the reference list is final.
