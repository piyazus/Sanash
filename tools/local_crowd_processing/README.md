# Local Crowd Processing Tools

This folder contains the local processing scripts prepared in the Codex workspace for:

- recursive MP4 frame extraction with blur filtering
- person-only frame filtering with YOLOv8n
- local P2PNet batch inference over selected frames

## Included files

- `extract_frames.py`
- `filter_people_frames.py`
- `p2pnet/requirements.txt`
- `p2pnet/models/*`
- `p2pnet/run_frames_inference.py`

## Not included

Large local artifacts were intentionally excluded from this branch:

- source MP4 videos
- extracted `frames/` images
- `frames_with_people/` and `frames_no_people/`
- generated prediction outputs
- downloaded checkpoint binaries such as `SHTechA.pth`

## Checkpoint note

The local P2PNet workflow was adapted around the official upstream P2PNet code path, but the checkpoint file itself is not committed here. Download the checkpoint separately and place it under `p2pnet/weights/` if you want to run the script as-is.
