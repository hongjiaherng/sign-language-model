# Data Preparation

## Dataset Download

We use a curated version of the WLASL dataset provided by FiftyOne on Hugging Face.
Since the original script to download the dataset has too many broken links, we use the curated version from FiftyOne.

```bash
uvx hf download Voxel51/WLASL --repo-type=dataset --local-dir=./data/WLASL
```

Prepare the reduced WLASL dataset by running the script:

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
```

## Dataset Structure

```bash
data/
├── WLASL/
│   ├── .cache/                 # Hugging Face cache
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
└── wlasl_reduced/
    ├── videos/
    │   ├── accident/
    │   │   ├── *.mp4
    │   │   └── ...
    │   ├── bathroom/
    │   │   ├── *.mp4
    │   │   └── ...
    │   └── ...
    ├── features_i3d_rgb_flow/ # I3D features rgb + flow (raft)
    │   ├── accident/
    │   │   ├── *.npy  # shape: (2, C) where 2 repreesents flow & rgb, C is feature dim (e.g., 1024)
    │   │   └── ...
    │   ├── bathroom/
    │   │   ├── *.npy
    │   │   └── ...
    │   ├── ...
    │   └── README.md
    ├── features_i3d_rgb/ # I3D features with rgb only
    │   ├── accident/
    │   │   ├── *.npy  # shape: (C,) where C is feature dim (e.g., 1024)
    │   │   └── ...
    │   ├── bathroom/
    │   │   ├── *.npy
    │   │   └── ...
    │   ├── ...
    │   └── README.md
    ├── features_kps/ # Keypoint features
    │   ├── accident/
    │   │   ├── *.npy  # shape: (C, T, V) = (3, T, 75), where T is number of frames, V is number of keypoints, C is coordinate dims (x,y,z)
    │   │   └── ...
    │   ├── bathroom/
    │   │   ├── *.npy
    │   │   └── ...
    │   ├── ...
    │   └── README.md
    ├── splits/
    │   ├── train.csv
    │   └── test.csv
    ├── gloss_map.json
    ├── metadata.csv
    └── README.md
```
