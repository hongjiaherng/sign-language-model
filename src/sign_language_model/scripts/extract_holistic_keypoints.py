import os
import argparse
import shutil
from multiprocessing import Lock, Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_holistic = mp.solutions.holistic
WARN_LOCK = Lock()
BBOX_EXPAND = 1.1  # Expand by 10%


def extract_xy(results):
    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        return None

    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )  # visibility is included for pose keypoints, as they might not be visible sometimes

    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )  # if the code is here, visibility must be 1, so we can ignore it to not add into the dimension

    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )  # if the code is here, visibility must be 1, so we can ignore it to not add into the dimension

    return np.concatenate([pose, lh, rh])  # (75*3 + 33,) -> (258,)


def sample_and_pad(frames, target_frames=32):
    """Sample and pad frames to target_frames length."""
    num_frames = frames.shape[0]

    if num_frames >= target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames).astype(np.int32)
        return frames[indices]

    # Last pad the keypoints
    pad = np.repeat(frames[-1][np.newaxis, :], target_frames - num_frames, axis=0)

    return np.concatenate([frames, pad], axis=0)  # (target_frames, 258)


def crop_to_bbox(frame, bbox, expand=BBOX_EXPAND):
    if bbox is None:
        return frame

    h, w, _ = frame.shape
    bx, by, bw, bh = bbox  # normalized [0,1]

    # center in normalized coords
    cx = bx + bw / 2
    cy = by + bh / 2

    # expand size
    bw = bw * expand
    bh = bh * expand

    # new top-left (normalized)
    x1 = max(0.0, cx - bw / 2)
    y1 = max(0.0, cy - bh / 2)
    x2 = min(1.0, cx + bw / 2)
    y2 = min(1.0, cy + bh / 2)

    # convert to pixel coords
    px1 = int(x1 * w)
    py1 = int(y1 * h)
    px2 = int(x2 * w)
    py2 = int(y2 * h)

    return frame[py1:py2, px1:px2]


def _worker_star(task):
    """
    Top-level helper for Windows multiprocessing.
    """
    return process_video(*task)


def log_warning(warn_file, message):
    with WARN_LOCK:
        with open(warn_file, "a") as f:
            f.write(message + "\n")
            f.flush()


def process_video(
    video_path,
    out_path,
    bbox=None,
    target_frames=32,
    warn_file=None,
):
    if out_path.exists():
        return True  # Skip if already processed

    try:
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Unable to open video file: {video_path}"

        valid_frames = []

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cropped = crop_to_bbox(frame, bbox)
                img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                results = holistic.process(img)

                # If no hands detected, even if pose is detected, that means no gesture is made
                if (
                    results.left_hand_landmarks is None
                    and results.right_hand_landmarks is None
                ):
                    continue

                kp = extract_xy(results)
                if kp is not None:
                    valid_frames.append(kp)
        cap.release()

        assert len(valid_frames) > 0, (
            f"No valid frames with detected keypoints: {video_path}"
        )

        valid_frames = np.array(valid_frames)  # (T, 258)
        output = sample_and_pad(valid_frames, target_frames)  # (target_frames, 258)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, output)
        return True

    except Exception as e:
        if warn_file is not None:
            log_warning(warn_file, str(e))
        return False


def main():
    command_used = " ".join(["python"] + os.sys.argv)
    parser = argparse.ArgumentParser(
        "Extract MediaPipe keypoints using Holistic model (pose + hands)"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--crop-to-bbox", action="store_true", default=False)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    in_root = Path(args.dataset_root)
    meta = pd.read_csv(in_root / "metadata.csv")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for child in out_root.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

    # Create warning file
    warn_file = out_root / "warnings.txt"

    tasks = []
    for row in meta.itertuples():
        video_path = in_root / row.filepath
        out_path = out_root / row.gloss / (Path(row.filepath).stem + ".npy")
        bbox = (
            (row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h)
            if args.crop_to_bbox
            else None
        )
        tasks.append((video_path, out_path, bbox, args.num_frames, warn_file))

    if args.num_workers is None or args.num_workers == 0:
        for task in tqdm(tasks, desc="Extracting keypoints"):
            _ = process_video(*task)
    else:
        with Pool(args.num_workers) as pool:
            list(
                tqdm(
                    pool.imap(_worker_star, tasks),
                    total=len(tasks),
                    desc="Extracting keypoints",
                )
            )

    # README
    if args.crop_to_bbox:
        crop_text = (
            "- Videos are cropped using the provided bounding boxes.\n"
            f"- Bounding boxes are expanded by **{(BBOX_EXPAND - 1) * 100:.0f}% on all sides** before cropping."
        )
    else:
        crop_text = "- Videos are not cropped; full frames are used."

    readme_path = out_root / "README.md"
    readme_text = f"""# Holistic Keypoint Features

This directory contains pre-extracted MediaPipe keypoint features derived from
the reduced WLASL dataset.

## Feature Description

- **Modality**: MediaPipe Holistic
- **Keypoints included**:
  - Pose (33 points): with visibility `(x, y, z, visibility)` (4 dims)
  - Left hand (21 points): without visibility `(x, y, z)` (3 dims)
  - Right hand (21 points): without visibility `(x, y, z)` (3 dims)
- **Total keypoints**: 258 per frame
- **Frames per video**: {args.num_frames} (fixed length)

Each feature file is stored as a NumPy array with shape: `({args.num_frames}, 258)`

## Preprocessing Details

{crop_text}
- Keypoints are extracted from **all frames** in the video.
- Frames without detected landmarks are discarded.
- From the remaining frames:
  - If at least 32 frames are available, 32 frames are **uniformly sampled**.
  - If fewer than 32 frames are available, **last-frame padding** is applied.

## Directory Structure

```bash
{out_root.as_posix()}/
├── <gloss_label>/
│   ├── <video_id>.npy  # shape: ({args.num_frames}, 258)
│   ├── ...
├── warnings.txt # Optional: Warnings during extraction
└── README.md
```

## Command Used

The features were generated using the following command:

```bash
{command_used}
```

"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_text)
    print(f"README saved to {Path(readme_path).as_posix()}")


if __name__ == "__main__":
    main()
