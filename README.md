# NYCU-CV-2026-HW2
DETR-based object detection project for digit detection on a custom dataset, including training, inference, and COCO-style submission generation.

# NYCU-CV-2026-HW2 Object detection using DETR (DEtection TRansformer) with PyTorch - NYCU Visual Recognition HW2 2026

# NYCU Computer Vision 2026 HW2

- Student ID: 114550826
- Name: Jakob Cleve

## Introduction

Object detection task using DETR (ResNet-50 backbone) with transfer learning. The model detects digits (0–9) in RGB images using a transformer-based architecture. It is based on a pretrained DETR model from Facebook Research and fine-tuned on a custom dataset.

## Environment Setup
bash
pip install torch torchvision pycocotools scipy pillow

## Usage

### Training
bash
python train.py

### Inference
bash
python inference.py

### Submission
bash
zip submission.zip pred.json

## Performance Snapshot
<img width="1920" height="6147" alt="image" src="https://github.com/user-attachments/assets/23c8f249-feb7-4953-b8a2-3ad920dae4e8" />

