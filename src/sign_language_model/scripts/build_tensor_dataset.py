import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_feature(feature_root, gloss, video_id):
    path = feature_root / gloss / f"{video_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return np.load(path)


def build_split(
    split_csv,
    gloss_map,
    features_rgb=None,
    features_flow=None,
    features_rgb_flow=None,
    features_kps=None,
):
    df = pd.read_csv(split_csv)

    X_rgb = []
    X_flow = []
    X_kps = []
    y = []

    for row in tqdm(df.itertuples(), total=len(df), desc=f"Loading {split_csv.name}"):
        gloss = row.gloss
        video_id = Path(row.filepath).stem
        label = gloss_map[gloss]

        # --- I3D RGB only ---
        if features_rgb is not None:
            rgb = load_feature(features_rgb, gloss, video_id)
            assert rgb.shape == (1024,)
            X_rgb.append(rgb)

        # --- I3D Flow only ---
        if features_flow is not None:
            flow = load_feature(features_flow, gloss, video_id)
            assert flow.shape == (1024,)
            X_flow.append(flow)

        # --- I3D RGB + Flow ---
        if features_rgb_flow is not None:
            rf = load_feature(features_rgb_flow, gloss, video_id)
            assert rf.shape == (2, 1024)
            X_rgb.append(rf[0])
            X_flow.append(rf[1])

        # --- Keypoints ---
        if features_kps is not None:
            kps = load_feature(features_kps, gloss, video_id)
            X_kps.append(kps)

        y.append(label)

    output = {
        "y": np.asarray(y, dtype=np.int64),
    }

    if X_rgb:
        output["X_rgb"] = np.stack(X_rgb, axis=0)

    if X_flow:
        output["X_flow"] = np.stack(X_flow, axis=0)

    if X_kps:
        output["X_kps"] = np.stack(X_kps, axis=0)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Build tensor datasets from pre-extracted WLASL features"
    )

    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--gloss-map", default="gloss_map.json")

    parser.add_argument("--features-rgb", default=None)
    parser.add_argument("--features-flow", default=None)
    parser.add_argument("--features-rgb-flow", default=None)
    parser.add_argument("--features-kps", default=None)

    parser.add_argument("--output-dir", default="tensors")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    root = Path(args.dataset_root)
    splits_dir = root / args.splits_dir
    out_dir = root / args.output_dir

    if out_dir.exists() and args.overwrite:
        for f in out_dir.glob("*"):
            f.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gloss map
    with open(root / args.gloss_map, "r") as f:
        gloss_map = json.load(f)

    features_rgb = root / args.features_rgb if args.features_rgb else None
    features_flow = root / args.features_flow if args.features_flow else None
    features_rgb_flow = (
        root / args.features_rgb_flow if args.features_rgb_flow else None
    )
    features_kps = root / args.features_kps if args.features_kps else None

    for split in ["train", "test"]:
        split_csv = splits_dir / f"{split}.csv"
        assert split_csv.exists(), f"Missing split file: {split_csv}"

        data = build_split(
            split_csv,
            gloss_map,
            features_rgb=features_rgb,
            features_flow=features_flow,
            features_rgb_flow=features_rgb_flow,
            features_kps=features_kps,
        )

        out_path = out_dir / f"{split}.npz"
        np.savez_compressed(out_path, **data)
        print(f"Saved {split} dataset to {out_path}")

    print("All done.")


if __name__ == "__main__":
    main()
