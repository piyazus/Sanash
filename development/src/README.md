# sanas_baseline

Minimal, reproducible, deliberately dumb counting baseline. It exists to close
step 5 of the minimal MVP path in `GROUND_TRUTH.md` section 11: a simplest
reproducible baseline end to end, before any complex model is chosen.

Status of what is here: **verified as running code**, nothing else. No claim
about accuracy, no architecture decision, no product metric.

## What it is

- Data: DISCO, the public audiovisual crowd counting set already on disk
  (`data/external/disco_images.zip`, `data/external/disco_density_maps.zip`,
  CC BY 4.0, provenance and checksums in `development/datasets.md`). Outdoor
  plazas from 8 to 15 m, elevated oblique. Not a bus cabin.
- Split: the train/val/test directories shipped inside the density archive,
  1,435 / 200 / 300. Not re-partitioned here.
- Target: the per image person count, which is the sum of the density map.
- Model: `DensityCNN`, a randomly initialised ResNet-18 trunk truncated at
  stride 8 plus a two layer density head. `weights=None`, so no ImageNet
  checkpoint is downloaded and no third party weight licence is involved.
- Floor: `ConstantPredictor`, which always answers the train split mean count.
  Every future candidate, CSRNet + PFCASA included, has to beat it on the same
  split before its numbers mean anything.
- Metrics: MAE and RMSE over counts, plus `level_accuracy` with caller supplied
  thresholds.

## Run it

CPU only. From `development/src`, in PowerShell, one command per line:

```
cd C:\Users\User\OneDrive\Desktop\me\Sanash\development\src
C:\Python313\python.exe -m sanas_baseline.prepare --train 16 --val 8 --test 8
C:\Python313\python.exe -m sanas_baseline.train --run-id smoke-20260902
C:\Python313\python.exe -m sanas_baseline.evaluate --checkpoint ..\..\outputs\baseline\smoke-20260902\checkpoint.pt --split test
```

`prepare` reads only the members it needs out of the 2.1 GB image archive; it
never unpacks it. `--train 0` means the whole split. Extracted samples land in
`data/interim/disco_baseline/` and run artifacts in `outputs/baseline/<run id>/`,
both gitignored.

Requires torch, torchvision, numpy, scipy and Pillow, all already installed
locally. There is no lockfile yet.

## The one subtle piece

A density map is people per pixel, so the label is its sum, not its mean.
Resizing it like an image divides that sum by the area ratio. At this geometry,
1080x1920 down to 27x48, a plain area resize turns a count of 112.0 into 0.070.
`data.resize_density_preserving_sum` aggregates by area and then renormalises
the total back, which keeps the count invariant to within float error
(112.0 becomes 112.00002). Anything that changes resolution later has to do the
same thing.

## What this does NOT prove

- Nothing about bus cabins. DISCO is outdoor crowd from a high oblique view.
  `GROUND_TRUTH.md` section 7.2 records the domain gap, and no own in-cabin
  footage exists yet.
- Nothing about the product target. What the `0..1` score means and where the
  five level boundaries sit are open (`GROUND_TRUTH.md` section 3.2), which is
  exactly why `metrics.level_accuracy` takes thresholds as an argument instead
  of defining them.
- Nothing about architecture. ResNet-18 here is a reference point, not a
  proposal, and it is not a comparison against CSRNet + PFCASA.
- Nothing about the edge device. No latency, memory, power, TensorRT export or
  Jetson measurement of any kind.
- The smoke test numbers are a wiring check on 16 training images and 2 epochs.
  They are not counting performance and must not be quoted as such.
