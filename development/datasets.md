# External datasets

> **PROVENANCE, НЕ СПИСОК РАЗРЕШЁННЫХ TRAINING DATA.** Этот файл подтверждает,
> что скачано и измерено. Текущая применимость, domain gap и legal blockers
> определены в [`../GROUND_TRUTH.md`](../GROUND_TRUTH.md). Полный Gorelik
> archive не скачан; локально есть только прежняя APC-выборка и ZIP index.

Provenance record for pretraining data pulled into `data/external/`.
`data/` is gitignored, so the archives themselves are not in git. This file is
the tracked record of what was fetched, from where, under what licence, and
what the files actually contain after verification.

All statistics below were measured from the downloaded artifacts, not copied
from the publications. Where a measured value disagrees with the published
value, both are given.

Download date: 2026-08-09. Fetched with `curl -L -C -`.
Total on disk: 3,471,468,120 bytes (3.47 GB).

Verification scripts used are throwaway and live in the session scratchpad,
not in the repo. They are trivial to re-derive from the numbers below.

---

## 1. RPEE-HEADS

Pedestrian head detection in crowded railway platforms and event entrances.

| Field | Value |
| --- | --- |
| Landing page | https://ped.fz-juelich.de/da/doku.php?id=rpee_heads |
| Dataset DOI | 10.34735/ped.2024.2 |
| Direct file URL | http://ped.fz-juelich.de/data/machine_learning/2024_11_Recognition_In_Field_Studies/data/2024rpee_heads_dataset.zip |
| Paper | IEEE Access, DOI 10.1109/ACCESS.2025.3563311; preprint arXiv:2411.18164 |
| Publisher | IAS-7 Civil Safety Research, Forschungszentrum Jülich |
| Local file | `data/external/2024rpee_heads_dataset.zip` |
| Size | 1,163,119,976 bytes (1.163 GB) |
| MD5 | `2b10f2cdc6cea78b748ea415de33e870` |
| SHA256 | `d8f0ac0ea3998300e77ed24f6068c35e4b38918309b1fe0628c4eef46b3a073c` |
| Measured throughput | 1.91 MB/s, 609 s |

### Licence

**CC BY-SA 4.0**, taken from the "License" section of the DOI landing page,
which reads in full: `Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).`

Two caveats that matter for the product path:

1. **No licence file ships inside the archive.** The zip contains only images
   and label files. There is no LICENSE, README, COPYING or CITATION file. The
   licence exists only on the landing page and in the paper, so the artifact
   itself is unmarked.
2. **The site footer contradicts the page.** The DokuWiki footer says
   "Except where otherwise noted, content on this wiki is licensed under
   CC Attribution 4.0 International" and links to `creativecommons.org/licenses/by/4.0/`.
   The page-level License section is the "otherwise noted" case, so CC BY-SA 4.0
   is the governing term. Anyone skimming the footer would read CC BY 4.0 and be
   wrong. Assume ShareAlike.

ShareAlike is the open legal question for shipping. Whether model weights
trained on this constitute an adaptation of the dataset is unsettled. Fine for
the paper track and internal pretraining; needs a legal read before it goes
into an Avtobys build.

### Verified contents

1,886 JPEG images and 1,886 label files, one label per image.

| Split | Images | Head annotations |
| --- | --- | --- |
| training | 1,346 | 78,606 |
| validation | 246 | 16,022 |
| testing | 294 | 15,285 |
| **Total** | **1,886** | **109,913** |

Image count, annotation total and per-split breakdown all match the published
figures exactly.

**One discrepancy.** The paper and landing page state an average of
approximately 56.2 heads per image. The measured average is **58.28**
(109,913 / 1,886). The published average does not reproduce from the published
totals either, so this is an error in the source, not in our download.

Per-image head count: min 0, max 255, mean 58.28, median 51. Twenty-four images
contain 0 to 5 heads.

Distribution, which is the reason we took this dataset:

| Heads per image | Images | Share |
| --- | --- | --- |
| 0-5 | 24 | 1.3% |
| 6-15 | 60 | 3.2% |
| 16-30 | 198 | 10.5% |
| 31-60 | 975 | 51.7% |
| 61-100 | 471 | 25.0% |
| 101+ | 158 | 8.4% |

85.1% of images carry more than 30 heads. This is genuine crowded-end coverage.

Resolutions are heterogeneous, 12+ distinct sizes. Most common: 981x552 (387),
3840x1944 (346), 4000x2640 (333), 736x552 (192), 4000x3000 (105). Roughly 31%
of images are small (under 1 MP), so resize policy matters.

### Scene families

The dataset is not homogeneous. Classified by filename prefix and confirmed by
viewing samples, it splits into two visually distinct populations:

| Family | Images | Heads | Heads/img | Images with mean luma < 60 |
| --- | --- | --- | --- | --- |
| field_natural | 1,307 (69.3%) | 75,413 | 57.7 | 666 |
| lab_capped | 579 (30.7%) | 34,500 | 59.6 | 0 |

- **field_natural** (prefixes `DMAGB41`, `DMAGB72`, `DMLGB72`, `DMLGP41`,
  `DMPG41`, `DMSA0442`, `DMSA0472`, `DMSA0542`, `DMSA0572`, `DMSAF951`,
  `DMWW71`, `DMWWX2`): real crowds at railway platforms and outdoor event
  entrances, ordinary clothing, elevated oblique cameras.
- **lab_capped** (prefixes `2C100`, `3C080`, `3C121`, `D2`, `EXP`, `entrance`,
  `predict`): controlled corridor and entrance experiments filmed near-nadir
  from a gymnasium ceiling, in which **participants wear brightly coloured or
  numbered caps** (red, fluorescent green, white with printed numbers).

The caps are a serious domain artifact. A detector trained on the full set can
learn "saturated coloured blob = head", which will not transfer to a bus cabin.
Recommend training on `field_natural` only, or at minimum holding
`lab_capped` out of validation so the metric is not flattered by it.

### Low light

666 images (35.3%) have mean luma below 60; 47 (2.5%) below 40; darkest is 19.0.
All 666 fall in the `field_natural` family. Visual check confirms real night
conditions: wet pavement, car headlights, artificial street lighting, crowds
queueing in the dark. This is authentic low-light crowd data, not underexposed
daylight.

### Annotation format

YOLO normalized detection format, one line per head:

```
0 0.448012 0.539855 0.015291 0.021739
```

`class cx cy w h`, all normalized to [0,1]. Single class, id `0`, across all
109,913 boxes.

---

## 2. DISCO (audiovisual crowd counting)

| Field | Value |
| --- | --- |
| Record | https://zenodo.org/records/3828468 |
| Concept DOI | 10.5281/zenodo.3828467 |
| Version DOI | 10.5281/zenodo.3828468 (v1.0) |
| Paper | "Ambient Sound Helps: Audiovisual Crowd Counting in Extreme Conditions", arXiv:2005.07097 |
| Local files | `data/external/disco_images.zip`, `data/external/disco_density_maps.zip` |

| File | Size (bytes) | MD5 | Matches Zenodo MD5 | SHA256 | Throughput |
| --- | --- | --- | --- | --- | --- |
| disco_images.zip | 2,186,215,586 | `ab28329777d05af94eb0992d8de1da89` | yes | `258d31f8af1357401c40bf5d044cc1199e0e83de40d3ab371e0b9d5917ebe99d` | 2.18 MB/s, 1002 s |
| disco_density_maps.zip | 122,132,558 | `0a9633a4e190414f9c6a41aee4e06a38` | yes | `690fbf007b2856c60d32f6b8028586f89c40dc1ad0e14ba263ca4db25ac3cd7c` | 0.62 MB/s, 196 s |

`audio.zip` (1,431,120,033 bytes) was deliberately not downloaded. We use RGB only.

Both MD5s match the checksums published in the Zenodo record metadata, so
integrity is confirmed against the publisher, not just internally.

### Licence

**CC BY 4.0**, from the Zenodo record metadata (`"license": {"id": "cc-by-4.0"}`),
access right `open`. Verified against the versioned record 3828468, not only the
concept DOI.

As with RPEE, **no licence file ships inside either archive**. The licence is
carried in the Zenodo record, which is the authoritative distribution record and
is adequate for due diligence, but the artifacts on disk are unmarked.

CC BY 4.0 permits commercial use with attribution and has no ShareAlike clause.
This is the cleanest licence of anything surveyed for the product path.

### Verified contents

**The archive ships far more images than are annotated.**

| Item | Count |
| --- | --- |
| JPEG images in `images.zip` | 8,116 |
| Density maps in `density_maps.zip` | 1,935 |
| Images with a matching density map | 1,935 |
| Density maps with no image | 0 |
| **Images with no annotation** | **6,181** |

The Zenodo description says "1,935 annotated images". That is accurate about
annotations but understates the image payload by 4.2x. 2,277 image ids carry a
`stage2-` prefix; of the 1,935 annotated, 321 are `stage2-` and 1,614 are plain
numeric.

The 6,181 unannotated images are a bonus, usable for self-supervised or
domain-adaptation pretraining, but they cannot supervise counting.

Density map split: train 1,435 / val 200 / test 300 = 1,935.

Verified statistics over the annotated subset:

- Total instances, summed over all density maps: **170,269.9**, against a
  published 170,270. Matches within float accumulation error.
- Per image: min 0.3, max 708.4, mean **87.99**, median 54.0.
- Published average of 88 per image reproduces exactly.

Images are 1920x1080 (confirmed from density map array shape (1080, 1920)).

### Low light

Measured over the 1,935 annotated images:

| Threshold (mean luma) | Images | Share |
| --- | --- | --- |
| < 30 | 141 | 7.3% |
| < 40 | 273 | 14.1% |
| < 50 | 422 | 21.8% |
| < 60 | 593 | 30.6% |
| < 70 | 697 | 36.0% |

Darkest 13.4, median 92.7, brightest 234.8. Visual check of the darkest samples
confirms genuine night scenes: dense crowds on urban plaza steps under
artificial light, faces and bodies barely separable from background.

This is the reason we took DISCO and it holds up. Roughly 30% of the annotated
set is meaningfully dark, and unlike RPEE the licence has no ShareAlike clause.

### Annotation format

MATLAB v5 `.mat`, one per image, single variable `map`, a float64 array of
shape (1080, 1920) matching the image dimensions. The array is a Gaussian
density map. Per-image count is the sum of the array.

**Point coordinates are not included in this archive.** Only rasterized density
maps. If we need head points rather than counts, they are not here.

---

## 3. PCDS (People Counting DataSet) — NOT DOWNLOADED, blocked on host

Bus-door RGB-D people counting. Phase 1 source for the door-mounted APC, and the
only public dataset we have found with the overhead-at-the-door geometry the
Gorelik rig lacks.

**Nothing has been fetched.** This entry records what was verified remotely on
2026-08-10 and why the download has not happened.

| Field | Value |
| --- | --- |
| Repo | https://github.com/shijieS/people-counting-dataset |
| Project page | https://shijies.github.io/people-counting-dataset/ |
| Paper | Sun, Akhtar, Song, Zhang, Li, Mian, IEEE T-ITS 20(10), Oct 2019; preprint arXiv:1804.04339 (submitted 12 Apr 2018, revised 28 Oct 2018) |
| Sensor | Kinect V1, mounted on the ceiling of front/back bus doors, non-zero pitch angle |
| Scale (README) | 5,464 video pairs, 10,908 videos, ~20,908 people, 30 scenes |
| Scale (paper body) | "4,689 videos"; abstract says "over 4500 videos" |
| Routes | No. 25, No. 301, No. 106 in Xi'An, XiNing and YinChuan, China |
| Collection | "three different bus routes at different times of the day up to 6 different days" |
| Local file | none — not downloaded |
| Size | **not stated anywhere and not measurable remotely** — see below |

### Licence

**CC BY-NC-SA 3.0.** From the repo README, verbatim:

> The datasets provided on this page are published under the Creative Commons
> Attribution-NonCommercial-ShareAlike 3.0 License. This means that you must
> attribute the work in the manner specified by the authors, you may not use
> this work for commercial purposes and if you alter, transform, or build upon
> this work, you may distribute the resulting work only under the same license.

**NonCommercial is a hard blocker for the product, and it is stricter than the
RPEE-HEADS case.** RPEE-HEADS is CC BY-SA 4.0, where only ShareAlike is
unsettled. PCDS adds NonCommercial on top, which is not an open question at all:
a commercial Avtobys deployment is exactly the excluded use.

Permitted: smoke test, method validation, Danyshpan's paper.
Forbidden: training or shipping any model that goes into commercial Avtobys.

Same class of blocker as the P2PNet "academic research only" clause, with the
same consequence: **the shipped model must be retrained from scratch on
licence-clean data, i.e. our own recording.** PCDS can validate that our
line-crossing logic is correct; it cannot supply production weights.

ShareAlike compounds it. If model weights are held to be a derivative, any
release would have to carry CC BY-NC-SA 3.0 too. The authors offer a commercial
route, shijieSun@chd.edu.cn, linked in the README as "contact us" for commercial
usage. That is the only clean path if PCDS-trained weights are ever wanted in
the product.

### Download status — checked 2026-08-10

| Host | Status |
| --- | --- |
| Google Drive | **DEAD.** HTTP 404. The README already says "has been removed for the space limit"; now confirmed rather than just claimed. |
| Baidu Pan | **ALIVE but gated.** https://pan.baidu.com/s/10O2JJrTC3WJJvweW8XGWVA with code 2s31. |
| GitHub releases | none. 0 releases, 0 assets. The repo is 8.6 MB of README and figures. |
| Project page | points at a different, older Baidu link (/s/1eR3fmdO), not the README one. |

The Baidu share resolves and is not flagged as expired: the shorturlinfo API
returns shareid 6773605951, uk 2972546568, expired_type 0. Listing the contents
returns errno -9 (extraction code required) and the verify endpoint returns
errno 105 (anti-automation). The data is there, but enumerating or downloading
it needs a browser session and in practice a Baidu account; free-tier Baidu also
throttles large transfers heavily.

**Consequence: the volume cannot be measured remotely, and no published source
states it** — not the README, not the paper, not the project page.

### Size estimate — ESTIMATE ONLY, not a measurement

Do not treat as fact. Grounded on the authors' own demo clips, whose durations
are real (YouTube metadata, 2026-08-10): depth demos 16, 22, 23, 29 s; colour
demos 17, 31, 41, 45 s. These may be montages, so treat them as a hint at clip
length rather than a mean.

Kinect V1 depth at 640x480, 16-bit, 30 fps is 18.4 MB/s raw, so a 20 s depth
clip is roughly 368 MB raw and 5,464 such clips would be about 2 TB
uncompressed. The archive is therefore certainly compressed, and the real figure
depends entirely on a codec nobody documents. Plausible range **150 GB to
800 GB**; the honest summary is that the uncertainty spans more than a factor of
five.

**Do not start a download on this estimate.** Someone must open the Baidu link
in a browser and read the actual folder size first. If it is in the hundreds of
GB, a full pull is off the table on a residential line and we need a scene-level
subset from the authors or a different plan.

### Sensor caveats that limit transfer to our hardware

**1. Kinect V1 is structured-light IR and degrades in sunlight — and the dataset
does contain sunlit door scenes.** The authors are explicit, from the paper:

> The rationale of dividing the dataset into noisy and clean videos is that
> Kinect V1 camera is sensitive to illumination conditions. For strong
> illumination, there is often noise in the videos [...] The videos in our
> dataset are mainly recorded in either direct sunlight or diffused sunlight,
> resulting in a natural division of corresponding levels of noise.

So N+/N- is literally a sunlight axis: N+ is strong/direct sunlight and noisy,
N- is mild/diffused and clean. That is good news, in that the failure mode is
present and labelled rather than designed out. But it is heavily imbalanced:

| category | condition | people |
| --- | --- | --- |
| N+C+ | strong sunlight, crowded | 2,086 |
| N+C- | strong sunlight, sequential | 1,284 |
| N-C+ | mild sunlight, crowded | 12,074 |
| N-C- | mild sunlight, sequential | 5,464 |

Only **3,370 of 20,908 people (16.1%) are in strong sunlight**. An aggregate
accuracy figure over the whole dataset is dominated by the benign condition and
**will be optimistic for a real Almaty door**. Any result we report must be
stratified N+ against N-, with N+ quoted separately.

**2. Kinect V1 is not the sensor we have been assuming.**

| | Kinect V1 (PCDS) | RealSense D435i (our assumption) |
| --- | --- | --- |
| Depth principle | structured light IR | active IR stereo |
| Depth range | 0.8-4.0 m default; 0.4-3.0 m near mode | ideal 0.3-3 m, max about 10 m |
| Depth resolution | 640x480 / 320x240 / 80x60 | 848x480 as measured in the Gorelik data |
| Depth FOV | 57 deg H x 43 deg V | wider |
| Hardware era | 2010 | current |

Two consequences. Structured light and active stereo fail *differently* in
sunlight: structured light loses its projected pattern outright, active stereo
degrades but can still exploit ambient texture. PCDS N+ noise is therefore not a
direct predictor of D435i behaviour, in either direction. Separately, Kinect V1's
0.8 m near limit is much higher than the D435i's 0.3 m, which matters for a
camera directly above a door that passengers pass close underneath.

The paper does **not** state depth resolution, frame rate or range anywhere. The
figures above are Kinect V1 hardware specifications, not PCDS measurements. An
automated summary claimed the paper says "VGA resolution (640x480) at 30 fps";
that string does not occur in the paper and was discarded.

Also unconfirmed: **the Gorelik cameras are never identified as D435i either.**
848x480 is a characteristic RealSense D400-series depth mode, which is why we
inferred it, but it remains an inference.

**3. The data is from 2016.** Scene names encode the date, e.g.
`25_20160411_front`. The paper body never states the recording year; 2016 comes
from the scene naming and the README. Note that the README misreads its own
example, glossing `20160411` as "04, Nov. 2016" when the format gives 11 April
2016. Ten-year-old Kinect V1 footage of Chinese city buses: the hardware is
obsolete, and Almaty door geometry, crowding and lighting are all unverified
against it.

### Ground truth format (from README, not yet verified against a real file)

Per scene directory, `label.txt`: the first 4 lines are the camera extrinsics as
a 4x3 matrix, then one line per video:

    DepthVideoName, EnteringNumber, ExitingNumber, VideoType

VideoType index: 0 = N-C-, 1 = N-C+, 2 = N+C-, 3 = N+C+.
Scene naming: `BUS_DATETIME_[front|back]`, e.g. `25_20160411_front`.

Take extrinsics from those four lines rather than deriving them.

### RGB is not synchronised with depth

From the README, verbatim, typo included:

> We only focus on the deth video and the color video is an accessory. We cannot
> guarantee the synchronization of color video and depth video.

Our door counter is a depth method, so this is survivable. **Never treat the RGB
channel as frame-aligned with depth**; eyeballing only. It also rules PCDS out as
a source of paired RGB training data for Phase 2.

### Internal inconsistencies to be aware of

- The video count is stated three ways: 5,464 pairs / 10,908 videos (README),
  4,689 videos (paper body), "over 4500" (abstract). And 5,464 x 2 = 10,928, not
  the 10,908 the README claims, so its own arithmetic is off by 20.
- The README per-category table gives N-C- a total of 5,464, numerically
  identical to the video-pair count. The four category totals do sum to 20,908,
  matching the stated headline, so it is probably genuine rather than a copy
  error, but it is worth remembering.

---

## Converting both to our target label

Scope: RPEE-HEADS and DISCO only. PCDS (section 3) is a depth-only
line-crossing dataset with integer entering/exiting counts, not a
crowd-density source, so it does not feed this conversion.

Neither dataset gives an occupancy level. Both give counts. No public dataset
supplies a cabin-capacity denominator, so the 5-level ordinal scale cannot be
derived from either without our own capacity assumption.

Conversion code needed, both small:

1. **RPEE to per-image count.** Read the `.txt` label, count non-empty lines.
   Exact integer. No parsing of coordinates needed for the count target,
   though the boxes are there if we want a detection-based head.
2. **DISCO to per-image count.** Load the `.mat`, take variable `map`, sum,
   round to nearest integer. The sums are floats and are not exactly integral
   (observed min 0.3), so rounding policy needs to be fixed once and recorded.
3. **Common count-to-level mapping.** A single function
   `level = f(count, capacity)` producing the 5 ordinal levels, with `capacity`
   a per-vehicle constant we choose. This is the piece no dataset provides and
   it is a modelling decision, not a data property. It should be defined once
   and reused, and it must be re-opened when real cabin data arrives.
4. **A shared loader** emitting `(image, count)` so the two sources can be
   mixed in one sampler despite different annotation formats.

Nothing here needs to be clever. The risk is not the code, it is silently
picking a capacity denominator and forgetting it was arbitrary.

---

## Viewpoint reality check

Scope: RPEE-HEADS and DISCO only, written before PCDS was adopted. PCDS is
the one source whose viewpoint does match the Phase 1 target, being mounted
on the ceiling of a bus door looking down at a pitch angle.

Honest assessment after viewing samples from every scene family in both
datasets, against the target of a bus cabin camera at 1.5 to 2.5 m looking
down an aisle.

**RPEE field_natural (railway platforms, event entrances).** Cameras are
elevated and oblique, roughly 3 to 6 m up on masts or scaffolds, looking down at
about 40 to 60 degrees, with visible fisheye barrel distortion. People present
as head-and-shoulders from above and behind, with real inter-person occlusion in
queues. This is the closest public geometry we have found to an in-cabin camera,
and closer than any street-level crowd benchmark. It is still not the same
thing: the floor is open, sightlines are long, and heads are smaller in the frame
than they would be in a cabin. My earlier characterisation of these as
"platform-height over open floor" was right about the open floor and wrong to
imply the camera is at head height. It is well above head height and angled down.

**RPEE lab_capped.** Near-nadir from a high ceiling, tiny heads, coloured caps.
Geometrically this is a drone-style top-down view, not a cabin view, and the caps
make it unrepresentative. Low value for us.

**DISCO.** Elevated oblique over urban plazas and steps, but from higher and
further back than RPEE, maybe 8 to 15 m. Head sizes are small, crowds are wide
and deep. Geometrically this is the weakest match of the three families. We are
taking it for the low-light supervision and the clean licence, not the viewpoint.

**What none of it reproduces.** Seat-back and stanchion occlusion, a confined
metal box with walls close to the lens, heads at 40 to 200 px, interior lamp
lighting, and the near-field extreme perspective of a camera 2 m from the
nearest passenger and 10 m from the furthest. These datasets buy crowded-scene
head features and low-light robustness. They do not buy cabin geometry, and no
public dataset does.
