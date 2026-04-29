# Almaty / Sanash P2PNet Dataset Integration

Dataset root:

```powershell
p2pnet_almaty_dataset_block_stratified
```

Dataset class:

```python
from datasets.almaty_transit import AlmatyTransitDataset, collate_fn

train_dataset = AlmatyTransitDataset(
    root="p2pnet_almaty_dataset_block_stratified",
    split="train",
    transforms=your_train_transforms,
)
val_dataset = AlmatyTransitDataset(
    root="p2pnet_almaty_dataset_block_stratified",
    split="val",
    transforms=your_val_transforms,
)
```

If the training code has a dataset registry or `build_dataset()` function, register
`AlmatyTransitDataset` there and use `collate_fn` for variable point counts.

First local dataloader check:

```powershell
python -B debug_almaty_dataset.py `
  --root p2pnet_almaty_dataset_block_stratified `
  --out debug_outputs/almaty_dataset_overlays `
  --num-overlays 10 `
  --seed 42 `
  --point-radius 6
```

Smoke fine-tuning command shape, adjusted to your P2PNet repo flags:

```powershell
python train.py `
  --dataset almaty_transit `
  --data-root p2pnet_almaty_dataset_block_stratified `
  --resume path\to\shanghaitech_part_b_p2pnet.pth `
  --epochs 3 `
  --batch-size 1 `
  --lr 1e-4 `
  --lr-backbone 1e-5 `
  --eval-every 1 `
  --save-every 1 `
  --output-dir runs\almaty_smoke
```

Prediction overlay command shape:

```powershell
python -B visualize_p2pnet_predictions.py `
  --root p2pnet_almaty_dataset_block_stratified `
  --checkpoint runs\almaty_smoke\checkpoint_epoch_003.pth `
  --model-module models.p2pnet `
  --model-factory build_model `
  --num-images 20 `
  --out debug_outputs/prediction_overlays
```
