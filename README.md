# 🤟 Sign Language Real-Time Translator

A computer vision project to recognize 35 American Sign Language (ASL) glosses on a reduced set of WLASL dataset.

This project demonstrates that for small-scale datasets (~400 samples), **geometric feature extraction (Keypoints)** significantly outperforms raw pixel-based methods (RGB/Optical Flow) due to the Curse of Dimensionality & small number of samples.

Preprocessed dataset is available here: https://huggingface.co/datasets/hongjiaherng/wlasl_reduced/tree/main.

Please refer to [data/README.md](data/README.md) for more details.

## Features

- Real-time gesture detection
- User-friendly Streamlit UI
- Localhost interface for easy testing

## How to run

1. Install required dependencies

```bash
uv sync --dev --extra cu130
# Or
# uv sync --dev --extra cpu
```

1. Run the app

```bash
uv run streamlit run app.py
```

## 🧑‍🤝‍🧑 Contribution Guide

- Fork the repository
- Create new feature branch
- Commit and push to remote feature branch
- Create Pull Request before merging into main branch

## 🤝 Team Members

Contributed by Jia Herng, Jasper, Zoe, Zen Yu, Zhi Yang
