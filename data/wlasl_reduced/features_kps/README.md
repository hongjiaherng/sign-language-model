# Holistic Keypoint Features

This directory contains pre-extracted MediaPipe keypoint features derived from
the reduced WLASL dataset.

## Feature Description

- **Modality**: MediaPipe Holistic
- **Keypoints included**:
  - Pose (33 points): `(x, y, z)` (4 dims)
  - Left hand (21 points): `(x, y, z)` (3 dims)
  - Right hand (21 points): `(x, y, z)` (3 dims)
- **Frames per video**: 32 (fixed length)

Each feature file is stored as a NumPy array with shape: `(C, T, V)` = `(3, 32, 75)`, where:

- `C`: Coordinate dimensions (x, y, z)
- `T`: Number of frames (fixed to 32)
- `V`: Number of keypoints (75 total)

## Preprocessing Details

- Videos are cropped using the provided bounding boxes.
- Bounding boxes are expanded by **10% on all sides** before cropping.
- Keypoints are extracted from **all frames** in the video.
- Frames without detected landmarks are discarded.
- From the remaining frames:
  - If at least 32 frames are available, 32 frames are **uniformly sampled**.
  - If fewer than 32 frames are available, **last-frame padding** is applied.

## Directory Structure

```bash
data/wlasl_reduced/features_kps/
├── <gloss_label>/
│   ├── <video_id>.npy  # shape: (3, 32, 75)
│   ├── ...
├── warnings.txt # Optional: Warnings during extraction
└── README.md
```

## Command Used

The features were generated using the following command:

```bash
python .\src\sign_language_model\scripts\extract_holistic_keypoints.py --dataset-root .\data\wlasl_reduced\ --output-root .\data\wlasl_reduced\features_kps --crop-to-bbox --num-workers 4
```

