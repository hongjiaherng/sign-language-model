# Data Preparation

## Dataset Download

We use a curated version of the WLASL dataset provided by FiftyOne on Hugging Face.
Since the original script to download the dataset has too many broken links, we use the curated version from FiftyOne.

```bash
uvx hf download Voxel51/WLASL --repo-type=dataset --local-dir=./data/WLASL
```

Prepare the reduced WLASL dataset by running the following script or download from [here](https://huggingface.co/datasets/jherng/wlasl_reduced).

```bash
# Build reduced WLASL dataset with selected glosses
uv run ./src/sign_language_model/scripts/build_reduced_wlasl.py \
    --wlasl-root ./data/WLASL/ \
    --output-root ./data/wlasl_reduced \
    --glosses "before,cool,thin,go,drink,help,computer,cousin,who,bowling,trade,bed,accident,tall,thanksgiving,candy,short,pizza,man,no,wait,good,bad,son,like,doctor,now,find,you,thank you,please,hospital,bathroom,me,i" \
    --target-fps 24

# Build train/test splits and extract features
uv run ./src/sign_language_model/scripts/build_dataset_splits.py \
    --dataset-root ./data/wlasl_reduced/ \
    --test-size 0.15 \
    --seed 42

# Extract I3D features and Holistic keypoints
# I3D RGB + Flow features
uv run ./src/sign_language_model/scripts/extract_i3d_features.py \
    --dataset-root ./data/wlasl_reduced/ \
    --output-root ./data/wlasl_reduced/features_i3d_rgb_flow \
    --crop-to-bbox \
    --num-frames 32 \
    --device cuda \
    --streams rgb+flow

# I3D RGB only features
uv run ./src/sign_language_model/scripts/extract_i3d_features.py \
    --dataset-root ./data/wlasl_reduced/ \
    --output-root ./data/wlasl_reduced/features_i3d_rgb \
    --crop-to-bbox \
    --num-frames 32 \
    --device cuda \
    --streams rgb

# Holistic Keypoint features
uv run ./src/sign_language_model/scripts/extract_holistic_keypoints.py \
    --dataset-root ./data/wlasl_reduced/ \
    --output-root ./data/wlasl_reduced/features_kps \
    --crop-to-bbox \
    --num-workers 4

# Combine features into tensor dataset
# KPS only
uv run python .\src\sign_language_model\scripts\build_tensor_dataset.py --dataset-root .\data\wlasl_reduced --features-kps features_kps --output-dir tensors/kps --overwrite

# RGB only
uv run python .\src\sign_language_model\scripts\build_tensor_dataset.py --dataset-root .\data\wlasl_reduced --features-rgb features_i3d_rgb --output-dir tensors/rgb --overwrite

# RGB + Flow only
uv run python .\src\sign_language_model\scripts\build_tensor_dataset.py --dataset-root .\data\wlasl_reduced --features-rgb-flow features_i3d_rgb_flow --output-dir tensors/rgb_flow --overwrite

# KPS + RGB
uv run python .\src\sign_language_model\scripts\build_tensor_dataset.py --dataset-root .\data\wlasl_reduced --features-kps features_kps --features-rgb features_i3d_rgb --output-dir tensors/kps_rgb --overwrite

# KPS + RGB + Flow
uv run python .\src\sign_language_model\scripts\build_tensor_dataset.py --dataset-root .\data\wlasl_reduced --features-kps features_kps --features-rgb-flow features_i3d_rgb_flow --output-dir tensors/kps_rgb_flow --overwrite

```

## Dataset Structure

```bash
data/
├── WLASL/
│   ├── .cache/                      # Hugging Face dataset cache
│   ├── data/
│   │   ├── data_0/
│   │   │   ├── *.mp4
│   │   │   └── ...
│   │   ├── data_1/
│   │   │   ├── *.mp4
│   │   │   └── ...
│   │   └── ...
│   ├── fiftyone.yml
│   ├── metadata.json
│   ├── frames.json
│   ├── samples.json
│   └── README.md
│
└── wlasl_reduced/
    ├── videos/                      # Reduced video dataset (by gloss)
    │   ├── accident/
    │   │   ├── *.mp4
    │   │   └── ...
    │   └── ...
    │
    ├── features_i3d_rgb_flow/        # I3D features (RGB + Flow)
    │   ├── accident/
    │   │   ├── *.npy                # shape: (2, 1024) → [rgb, flow]
    │   │   └── ...
    │   └── README.md
    │
    ├── features_i3d_rgb/             # I3D features (RGB only)
    │   ├── accident/
    │   │   ├── *.npy                # shape: (1024,)
    │   │   └── ...
    │   └── README.md
    │
    ├── features_kps/                 # MediaPipe keypoint features
    │   ├── accident/
    │   │   ├── *.npy                # shape: (3, T, 75) = (C, T, V)
    │   │   └── ...
    │   └── README.md
    │
    ├── splits/                       # Dataset splits
    │   ├── train.csv
    │   └── test.csv
    │
    ├── tensors/                      # Preloaded tensor datasets (.npz)
    │   ├── kps/
    │   │   ├── train.npz
    │   │   └── test.npz
    │   │   # Access:
    │   │   # data = np.load("train.npz")
    │   │   # X_kps = data["X_kps"]   # (N, 3, T, 75)
    │   │   # y     = data["y"]       # (N,)
    │   │
    │   ├── rgb/
    │   │   ├── train.npz
    │   │   └── test.npz
    │   │   # Access:
    │   │   # data = np.load("train.npz")
    │   │   # X_rgb = data["X_rgb"]   # (N, 1024)
    │   │   # y     = data["y"]
    │   │
    │   ├── rgb_flow/
    │   │   ├── train.npz
    │   │   └── test.npz
    │   │   # Access:
    │   │   # data = np.load("train.npz")
    │   │   # X_rgb  = data["X_rgb"]  # (N, 1024)
    │   │   # X_flow = data["X_flow"] # (N, 1024)
    │   │   # y      = data["y"]
    │   │
    │   ├── kps_rgb/
    │   │   ├── train.npz
    │   │   └── test.npz
    │   │   # Access:
    │   │   # data = np.load("train.npz")
    │   │   # X_kps = data["X_kps"]   # (N, 3, T, 75)
    │   │   # X_rgb = data["X_rgb"]   # (N, 1024)
    │   │   # y     = data["y"]
    │   │
    │   └── kps_rgb_flow/
    │       ├── train.npz
    │       └── test.npz
    │       # Access:
    │       # data = np.load("train.npz")
    │       # X_kps  = data["X_kps"]  # (N, 3, T, 75)
    │       # X_rgb  = data["X_rgb"]  # (N, 1024)
    │       # X_flow = data["X_flow"] # (N, 1024)
    │       # y      = data["y"]
    │
    ├── gloss_map.json                # gloss → class_id mapping
    ├── metadata.csv                  # per-video metadata
    └── README.md
```
