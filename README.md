# 🤟 Sign Language Detection

Real-time ASL (American Sign Language) finger-spelling recognition using EfficientNet-B0 and MediaPipe hand tracking.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project classifies 24 ASL letters (A–Y, excluding J and Z which require motion) from a live webcam feed. It uses:

- **EfficientNet-B0** (via `timm`) fine-tuned on Sign MNIST
- **MediaPipe Hand Landmarker** (Tasks API) for hand detection and bounding box cropping
- **Two-stage training**: frozen backbone → full fine-tuning at epoch 5

---

## Demo

The webcam inference pipeline:
1. Detects the hand using MediaPipe and crops it
2. Preprocesses the crop to match training distribution
3. Runs EfficientNet-B0 and overlays the predicted letter + confidence

> **Note:** The model is trained on Sign MNIST (clean, grayscale, studio images). Real-world webcam accuracy may vary. See [Known Limitations](#known-limitations).

---

## Project Structure

```
sign-language-detection/
├── data/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── checkpoints/
│   └── best_model.pth          # saved after training
├── config.py                   # hyperparameters & paths
├── dataset.py                  # SignDataset + DataLoaders
├── model.py                    # SignEfficientNet definition
├── train.py                    # training loop
├── utils.py                    # train_one_epoch, validate
├── inference.py                # real-time webcam inference
└── requirements.txt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/sign-language-detection.git
cd sign-language-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Get [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) from Kaggle and place the CSVs in `data/`:

```
data/
├── sign_mnist_train.csv
└── sign_mnist_test.csv
```

### 4. Create the checkpoints directory

```bash
mkdir checkpoints
```

---

## Training

```bash
python train.py
```

Training runs for 15 epochs. The backbone is frozen for the first 4 epochs, then fully unfrozen for fine-tuning. The best model (by validation accuracy) is saved to `checkpoints/best_model.pth`.

**Default hyperparameters** (edit in `config.py`):

| Parameter     | Value  |
|---------------|--------|
| Batch size    | 32     |
| Epochs        | 15     |
| LR (head)     | 1e-4   |
| LR (finetune) | 1e-5   |
| Weight decay  | 1e-4   |
| Image size    | 224×224|

---

## Inference (Webcam)

```bash
python inference.py
```

- The MediaPipe hand landmark model (`hand_landmarker.task`) is downloaded automatically on first run.
- Press **Q** to quit.

---

## Known Limitations

- **Domain gap**: Sign MNIST consists of clean, centered, grayscale images. Performance on webcam footage with real-world lighting and backgrounds will be lower than the reported test accuracy.
- **J and Z excluded**: These letters require motion and are not part of the dataset.
- **Single hand only**: Inference is configured for `num_hands=1`.

---

## Requirements

```
pandas
numpy
scikit-learn
torch
torchvision
timm
tqdm
opencv-python
mediapipe
```

---

## License

MIT
