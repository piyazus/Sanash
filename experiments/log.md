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
