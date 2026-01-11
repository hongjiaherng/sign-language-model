# I3D Video Features

This directory contains pre-extracted I3D spatiotemporal features derived from
the reduced WLASL dataset.

## Feature Description

- **Backbone**: Pretrained I3D
- **Input FPS**: 24 FPS
- **Streams**: RGB only
- **Clip length**: 32 frames
- **Clip stride**: 32 frames
- **Pooling strategy**:
  - Global average pooling per clip
  - Average pooling across clips

Each video is represented as a single NumPy array:

- RGB only: `(1024,)`

## Preprocessing Details

- Videos are cropped using the provided bounding boxes.
- Bounding boxes are expanded by **10% on all sides** before cropping.
- Videos are resized to 224x224 pixels.
- Videos are segmented into non-overlapping 32-frame clips.
- If the video length (number of frames) is not a multiple of 32, it is padded by repeating the last frame. One extra clip (32 frames) is always added as a safety margin to every video.


## Directory Structure

```bash
data/wlasl_reduced/features_i3d_rgb/
├── <gloss_label>/
│   ├── <video_id>.npy # shape: (1024,) or (2, 2048)
│   ├── ...
├── warnings.txt # Optional: Warnings during extraction
└── README.md
````

## Command Used

The features were extracted using the following command:

```bash
python .\src\sign_language_model\scripts\extract_i3d_features.py --dataset-root .\data\wlasl_reduced\ --output-root .\data\wlasl_reduced\features_i3d_rgb --crop-to-bbox --num-frames 32 --device cuda --streams rgb
```
