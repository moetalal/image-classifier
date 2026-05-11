# 🖼️ Image Classifier — Transfer Learning with MobileNetV2

A production-ready image classification pipeline using **transfer learning** on MobileNetV2, with a built-in CIFAR-10 demo that requires zero external data.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **Transfer learning** on ImageNet-pretrained MobileNetV2 (224×224)
- **CIFAR-10 demo** — trains a CNN with no external dataset needed
- **Data augmentation** — random flip, rotation, zoom
- **Early stopping + LR scheduling** — trains smart, not just long
- **Single image prediction** with top-3 confidence scores

## 🚀 Quick Start

```bash
git clone https://github.com/moetalal/image-classifier
cd image-classifier
pip install -r requirements.txt

# Run CIFAR-10 demo (downloads dataset automatically)
python classifier.py demo

# Train on your own images
python classifier.py train ./data --epochs 20 --save my_model.keras

# Predict a single image
python classifier.py predict my_model.keras photo.jpg --classes cat dog bird
```

## 📁 Custom Dataset Structure

```
data/
├── train/
│   ├── cats/     *.jpg
│   └── dogs/     *.jpg
└── val/
    ├── cats/     *.jpg
    └── dogs/     *.jpg
```

## 📊 Results (CIFAR-10, 5 epochs)

| Metric | Score |
|--------|-------|
| Test Accuracy | ~72% |
| Training Time | ~3 min (GPU) |

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `TensorFlow / Keras` | Model building & training |
| `MobileNetV2` | Pre-trained ImageNet backbone |
| `tf.data` | Efficient data pipeline |

## 📄 License

MIT © [Mohamed Hamid](https://github.com/moetalal)
