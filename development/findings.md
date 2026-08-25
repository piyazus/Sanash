# Sanas — findings so far

> **ИСТОРИЯ, НЕ CURRENT SPEC.** Этот файл фиксирует измерения архитектур до
> пивота 2026-08-25. Door/depth выводы не являются результатами текущей
> потолочной RGB-системы. Актуальное состояние: [`../GROUND_TRUTH.md`](../GROUND_TRUTH.md).

Everything here is measured or cited. Numbers without a source are not in this
document. Where something is a guess or an untuned default it says so.

Written 2026-08-10. Companion documents: `datasets.md` (external data
provenance), `../experiments/log.md` (run log).

---

## 1. Headline

Three things changed the plan.

1. **The substitute dataset is not a crowding dataset.** It tops out at four
   occupants, and the public field has nothing better — this is representative,
   not unlucky.
2. **P2PNet cannot ship.** Its licence restricts use to academic research, and
   Sanas ships inside Avtobys. The same applies to most crowd-counting
   benchmarks.
3. **Depth-based occupancy from one in-cabin camera does not work.** Measured:
   10% floor coverage, Pearson 0.184, and the score stops rising past two
   occupants.

---

## 2. Substitute dataset (Zenodo 10.5281/zenodo.20559664)

Read by HTTP range requests against the ZIP central directory. The 73.5 GB
archive was never downloaded in full.

### Record

| Field | Value |
|---|---|
| Title | Multi-View In-Cabin Monitoring System for Public Transport Vehicles |
| Licence | CC BY 4.0 |
| Files | one: `beintelli_v1.zip`, 73,515,214,805 bytes |
| md5 | `74924b5be59e706a5a89affba48f6b87` (matches `development/scripts/download_dataset.sh`) |
| Members | 298,694, 133,366,903,156 bytes uncompressed |
| Published | 2026-06-05 |

Paper: arXiv 2606.11739v1, five authors (GT-ARC / TU Berlin, plus MAN Truck &
Bus SE). The Zenodo record lists one creator; the paper lists five, so
"Gorelik et al." is correct.

### Contents

| Path | Files | Size | Content |
|---|---|---|---|
| `body_3d/` | 52,636 | 66.30 GB | ~1.26 MB JSON per person per frame |
| `lidar/` | 10,035 | 42.09 GB | Ouster OS0-128, 4,194,304 B each |
| `cameras/color/` | 50,578 | 13.01 GB | 4 RGB cameras, 1280x720 JPEG |
| `cameras/depth/` | 51,640 | 11.67 GB | 4 depth cameras |
| `segmentations/` | 103,752 | 0.14 GB | per-person PNG masks + JSON boxes |
| `poses/` | 10,035 | 0.11 GB | 3D keypoints |
| `bboxes_3d/` | 10,035 | 0.011 GB | oriented 3D boxes, label `"human"` |
| `states/` | 9,977 | 0.002 GB | per-person state/action labels |

### The occupancy problem

```
persons per frame (person_states.json, 9,975 frames): {1: 4989, 2: 4418, 3: 546, 4: 22}
unique person IDs across the whole dataset: 26
max simultaneous masks in one camera view: 6 (center_left)
frames with an empty cabin: 898
```

The paper states it plainly: "scenes with 1-2 occupants". The action vocabulary
(`mugging`, `vandalism`, `littering`, `smoke`, `punch`, `push`) shows what this
is — a staged behaviour and safety recording with actors.

Consequences:

- Levels 3, 4 and 5 of the intended five-level scale have **zero** training
  examples.
- Recording is a **single continuous 32-minute daytime session**, 2026-05-21
  09:10:44 to 09:43:04 UTC, split into 11 sub-sequences, ~6.6 fps effective per
  camera. There is no night, dusk or artificial-light data, so low-light
  robustness cannot be evaluated here at all.
- The vehicle was **stationary**. No vibration, no motion blur, no changing
  daylight.
- Annotations are **pseudo-labels**: 2D detector, manual filtering, SAM 3 Body
  3D for mesh and skeleton, ByteTrack for association, boxes derived from
  poses. Only the action and state labels are hand-made.

### Practical gotchas

- On disk the layout is a custom ROS tree, **not nuScenes**. nuScenes is an
  export target produced by a script in the toolkit repo, and that repo carries
  **no licence file** — the data is CC BY 4.0 but its tooling is not.
- Depth files are named `*.jpg` but their magic bytes are `89504e47`. They are
  PNGs. 1,204 of them were renamed by content sniffing in our extraction run.
- **Depth intrinsics differ from colour intrinsics**: depth is 848x480 with
  fx≈422, colour is 1280x720 with fx≈920. Using the colour intrinsics for
  deprojection fails silently.
- Extrinsics **are** in the archive (`frame_ids.json` plus the URDF chain in
  `target_transforms.json`), so no unlicensed toolkit code is needed.
  `front_left_depth` resolves to `[-1.176, 4.220, 1.953]` in `base_link`.
- A **HuggingFace mirror** exists and is faster from Almaty: 2.81 MB/s versus
  325 KB/s from Zenodo. Both honour HTTP range requests.

### Sensor rig used by the authors

Four Intel RealSense D435i, each on a Raspberry Pi, PoE switch, NUC 11 running
ROS 2, NTP-synchronised. Ouster OS0-128 LiDAR. RGB at 15 Hz, LiDAR at 10 Hz.
The paper gives the D435i's reliable depth window as approximately 0.3 to
3.0 m.

The LiDAR is **not** their label source. It is used for ICP calibration, for
qualitative validation of the reconstructed meshes, and as an input modality in
their BEVFusion benchmark.

---

## 3. Model candidates and licences

| Candidate | Verified numbers | Licence | Verdict |
|---|---|---|---|
| CSRNet | 16.26M params; ShanghaiTech A MAE 68.2 / MSE 115.0, B 10.6 / 16.0 | reference repo has **no licence file** | reference only; reimplement if needed |
| P2PNet | ShanghaiTech A MAE 52.74 / MSE 85.06, B 6.25 / 9.9 — best accuracy here | **"Use in source and binary forms shall only be for the purpose of academic research"** | **paper track only, never the product** |
| DINOv2 ViT-S/14 + CORN head | 21M params, ImageNet linear 81.1% | Apache-2.0; `coral-pytorch` head MIT | chosen for the first run |
| ConvNeXt-Tiny + same head | no benchmark exists for this task | MIT; timm Apache-2.0 | the edge-deployable variant |

Neither paper states inference speed, and **no latency number can be produced
for any candidate until the edge device is chosen**.

The P2PNet finding contradicts the stack recorded in the global project notes
("P2PNet, ShanghaiTech Part B"). Both halves are problems for a shipping
product: P2PNet is academic-only, and ShanghaiTech has **no stated licence at
all**, which in a due-diligence review is worse than a restrictive one.

---

## 4. Compute limits (Kaggle)

| Limit | Value | Source |
|---|---|---|
| Max size, single dataset | 200 GB | kaggle.com/docs/datasets |
| Private dataset storage | 200 GB | kaggle.com/docs/datasets |
| Top-level files per dataset | 50 | kaggle.com/docs/datasets |
| Notebook runtime | 12 h CPU/GPU, 9 h TPU | kaggle.com/docs/notebooks |
| `/kaggle/working` | 20 GB, output also capped at 20 GB | kaggle.com/docs/notebooks |
| GPU quota | ~30 h/week, "sometimes higher depending on demand" | kaggle.com/docs/efficient-gpu-usage |
| Sessions | P100, or 2×T4; 4 cores, 29 GB RAM | kaggle.com/docs/notebooks |

Not documented anywhere: the scratch disk size outside `/kaggle/working`, and
the maximum size of an individual output file. **GPU quota remaining is not
exposed by the API** and has to be read at kaggle.com/settings.

### Kaggle mechanics learned the hard way

`kernel_sources` **does** mount the source kernel's output. It appears at
`/kaggle/input/notebooks/<owner>/<slug>/`, alongside five code artifacts
(`__output__.json`, `__results__.html`, `__script__.ipynb`, `__script__.py`,
`custom.css`). An earlier conclusion in this project that it mounts only code
was wrong — the directory walk that produced it capped recursion at three
levels and the data sits at four.

`kernels output` paginates and will not finish on a large output. Fetch a
single file instead:

```python
api.kernels_output('owner/slug', path='outputs', file_pattern=r'.*\.log$')
```

Note `file_pattern` is a regex, not a glob.

---

## 5. Runs executed

### `sanas-extract-subset` — COMPLETE

Range-fetches a stride-10 subset straight from the HuggingFace mirror. Nothing
larger than the 44.9 MB central directory ever touches a local disk.

```
extracted 35,404 files, 0.686 GB in 1730.6 s
frames                : 1,289
sub-sequences         : 11
missing view label    : 0
count_view  dist      : {0: 218, 1: 632, 2: 389, 3: 50}
count_cabin dist      : {0: 120, 1: 656, 2: 453, 3: 60}
corrected 1,204 file extensions after magic-byte sniff
```

Both histograms sum to 1,289. The subset tops out at three occupants — the 22
four-occupant frames in the full dataset did not survive stride-10 sampling.

**Corrected 2026-08-10 (see `../experiments/log.md`, entry "two-phase plan
recorded"). The subtraction `218 - 120 = 98` is wrong** — it under-reports,
because 4 frames have `view > 0` while `cabin == 0` (pseudo-label
disagreement), so the zero-sets are not nested. Measured directly:

- `view == 0` and `cabin > 0`: **102 frames (7.9%)** — the camera sees nobody
  at all while someone is aboard.
- `view < cabin` (any undercount): **181 frames (14.0%)**. `view > cabin`:
  7 frames (0.5%).

14.0% is the operative figure. A weak-label scheme binding cabin-level counts
to a single view inherits that, not 7.6%. Either way it is a measured cost of
single-camera occlusion, not an estimate.

### `sanas-depth-occupancy` — COMPLETE (third attempt)

Geometric baseline: deproject depth with the correct depth intrinsics,
transform into `base_link`, build a BEV occupancy grid, correlate against
ground truth. No training, no GPU.

```
front_left_depth   zero 15.5%   in 0.3-3.0 m 42.2%   beyond 42.3%

STEP 1: FLOOR COVERAGE (118 empty-cabin frames)
  occupied-region cells (every 3D box, dilated 0.3 m): 4,110
  front_left_depth covers 10.0% of the region; 3,699 cells never seen

STEP 2/3: OCCUPANCY vs GROUND TRUTH (n = 1,204)
  Pearson  r   : 0.184   95% CI [0.133, 0.231]
  Spearman rho : 0.139   95% CI [0.081, 0.199]
  monotonic in count: False

   count      n     mean    median      std
       0    118   0.0014   0.0013   0.0003
       1    615   0.0017   0.0015   0.0007
       2    412   0.0019   0.0015   0.0010
       3     59   0.0017   0.0014   0.0009
```

**Single-camera depth occupancy does not work.** Three independent reasons:

- It sees 10% of the region occupants actually use. Nine tenths of the cabin is
  outside the reliable window.
- Correlation of 0.184 excludes zero but explains roughly 3% of the variance.
- The score is **not monotonic**: it rises from zero to two occupants and falls
  at three. This is the saturation predicted for a surface sensor — a depth
  camera sees the shell of a group, not its volume — and it appeared at three
  people rather than thirty.

Absolute values are 0.0014 to 0.0019, meaning 0.14% to 0.19% of cells occupied.
The differences are near noise.

Failed attempts before this: two ERROR runs caused by the recursion-depth bug
described in section 4, not by anything in the data or the geometry.

### Preliminary four-camera coverage (30-frame local sample, NOT the sanctioned run)

| Range gate | front_left alone | union of 4 |
|---|---|---|
| 3.0 m (spec) | 6.7% | 28.8% |
| 6.0 m | 16.1% | 54.9% |
| 10.0 m | 21.5% | 63.3% |

Only one empty-cabin frame in that sample, so treat as indicative. The full run
is built and awaiting approval. If it holds, four cameras still leave two
thirds of the occupant region unseen at spec range.

---

## 6. External data for the crowded end

Full provenance in `datasets.md`. Summary of the search:

**Nothing public supervises a crowded transit cabin.** Every in-cabin dataset
with real annotations hits the same ~4-occupant ceiling or has no count labels.
PMOF, published two months ago and purpose-built for this, also stops at four.
BUS-HAR (overhead views of crowded buses — exactly the target) is not public.
PHD is the only genuinely crowded carriage dataset found; its GitHub repo is an
empty stub, distribution is Baidu Netdisk only, and it has no licence.

Downloaded and verified, 3.47 GB total:

| Dataset | Verified content | Licence | Use |
|---|---|---|---|
| RPEE-HEADS | 1,886 images, 109,913 heads, mean **58.28** (published figure of 56.2 does not reproduce); 85.1% of images carry >30 heads; 666 dark images | CC BY-SA 4.0 per the page section — the wiki footer says CC BY 4.0 and is wrong | pretraining, crowded end |
| DISCO | **8,116** JPEGs, not the 1,935 advertised; 1,935 annotated, 6,181 unannotated; 170,269.9 instances; 593 dark, 141 very dark | CC BY 4.0, cleanest licence surveyed | pretraining, low light |

Neither archive ships a licence file. The licence exists only on the
publisher's page — worth recording before anyone asks in a review.

**RPEE-HEADS is two datasets in one.** 1,307 `field_natural` images versus 579
`lab_capped` images where participants wear bright red, fluorescent green or
numbered white caps. A detector trained on the lab third learns "saturated
coloured blob = head". Train on the field subset only, and at minimum keep the
lab images out of validation.

Camera geometry: RPEE field cameras sit 3-6 m up, angled 40-60° down with
fisheye distortion. That is the closest public geometry to an in-cabin camera
found anywhere. DISCO is taken for darkness and licence, not viewpoint.

---

## 7. Hardware implications

The evidence so far bears directly on the proposed rig.

**Camera count.** One camera covers 10% of the occupant region at spec range.
The preliminary four-camera union reaches 28.8%. A two-camera rig — one front,
one rear — is not supportable by any number measured here. The sanctioned
multizone run will settle this properly, and it costs nothing but CPU time.

**Compute.** A geometric pipeline is array arithmetic and is CPU-bound; it does
not need an NPU. An accelerator is justified by the CNN branch, and the
strongest argument for Jetson is TensorRT maturity rather than raw throughput.
"The only affordable board that will cope" is not accurate — the choice is
about toolchain risk.

**Depth as the primary signal.** Weak, on the evidence. It buys darkness
tolerance and privacy, but it saturates early, and saturation lands exactly on
the levels the product exists to distinguish. A hybrid — geometry for the
sparse end and in darkness, a network for the dense end — matches the
measurements better than either alone.

None of this is validated on a full cabin, because no such data exists yet.

---

## 8. What is still unknown

- **GPU quota remaining.** Not exposed by the API; blocks the training kernel.
- **Edge device.** Undecided, so no latency comparison is possible.
- **Four-camera coverage on the full subset.** Kernel built, not yet run.
- **Behaviour in a genuinely full cabin.** Unobservable with any public data.
  Requires own recording once municipal approval lands.
- **Occupancy definition.** Every dataset gives counts; none gives a level. The
  count-to-level mapping needs a cabin-capacity denominator that has to come
  from the product side, and whether a pram or a suitcase counts as occupied
  space is a product decision, not a modelling one.
- **RPEE-HEADS ShareAlike.** Whether trained weights constitute an adaptation
  is unsettled. Fine for the paper track; needs a legal read before shipping.

## 9. Untuned defaults currently in the code

These are guesses, exposed as parameters, and none has been tuned:

- `z_min` 0.6 m for "above seat height", inferred from box geometry; the
  archive documents no seat height.
- Depth values treated as uint16 millimetres — the RealSense convention,
  consistent with observed ranges, but stated nowhere in the archive.
- Pseudo-label score threshold 0.0, so every detection counts.
- BEV cell 0.1 m, minimum 3 points per cell, background quantile 0.5.
- Coverage region derived from every 3D box dilated 0.3 m, because the archive
  contains no cabin floor polygon.
- Validation fraction 0.25, held out by sub-sequence.
