import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Build train/test splits for reduced WLASL dataset"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    root = Path(args.dataset_root)
    meta_path = root / "metadata.csv"
    gloss_map_path = root / "gloss_map.json"
    splits_dir = root / "splits"

    assert meta_path.exists(), f"Missing {meta_path}"
    assert gloss_map_path.exists(), f"Missing {gloss_map_path}"

    if splits_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{splits_dir} already exists. Use --overwrite to regenerate."
        )

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df = pd.read_csv(meta_path)

    # Load gloss map
    with open(gloss_map_path, "r") as f:
        gloss_map = json.load(f)

    # Basic validation
    missing_glosses = set(df["gloss"]) - set(gloss_map.keys())
    if missing_glosses:
        raise ValueError(f"Glosses missing from gloss_map.json: {missing_glosses}")

    # Stratified split by gloss
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["gloss"],
    )

    # Reset indices for cleanliness
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save splits
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Print summary
    print("Dataset splits created successfully.")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"Saved to: {splits_dir.as_posix()}")

    # Optional: class distribution sanity check
    print("\nClass distribution (train):")
    print(train_df["gloss"].value_counts().sort_index())

    print("\nClass distribution (test):")
    print(test_df["gloss"].value_counts().sort_index())


if __name__ == "__main__":
    main()
