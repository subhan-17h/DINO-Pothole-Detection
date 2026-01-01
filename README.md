# DINO Pothole Detection

A fine-tuned DINO (DETR with Improved DeNoising Anchor Boxes) implementation for automatic pothole detection in road images. This project leverages transfer learning from COCO-pretrained weights to achieve accurate pothole localization with minimal training data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements **DINO (DETR with Improved DeNoising Anchor Boxes)**, a state-of-the-art object detection architecture based on transformers, fine-tuned specifically for pothole detection in road images.

### Key Features
- **Transfer Learning**: Utilizes COCO-pretrained weights (11 epochs) for efficient fine-tuning
- **State-of-the-Art Architecture**: 4-scale deformable transformer with ResNet-50 backbone
- **Data Efficient**: Achieves good results with small datasets (100-500 images)
- **Production Ready**: COCO-style evaluation metrics and inference pipeline
- **Mixed Precision Training**: AMP support for faster training with reduced memory usage

### Use Cases
- Road maintenance automation
- Autonomous vehicle safety systems
- Infrastructure monitoring
- Mobile applications for pothole reporting
- Municipal damage assessment

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 8GB+ GPU memory (16GB+ recommended)

### Setup

1. **Clone the repository**
```bash
cd DINO
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- transformers
- opencv-python
- pycocotools
- scipy
- Pillow

---

## Dataset Preparation

### Dataset Structure

Organize your pothole dataset in COCO format:

```
datasets/pothole/
├── train2017/                # Training images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val2017/                  # Validation images
│   ├── val_image1.jpg
│   └── ...
└── annotations/              # COCO format annotations
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Annotation Format

Your JSON files must follow COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 800,
      "height": 600
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "pothole"
    }
  ]
}
```

### Data Requirements
- **Minimum**: 100 training images, 20 validation images
- **Recommended**: 500+ training images, 100+ validation images
- **Image size**: Any size (resized during training)
- **Format**: JPG, PNG

### Annotation Tools
- [CVAT](https://github.com/opencv/cvat) - Free, open-source annotation tool
- [LabelMe](https://github.com/wkentaro/labelme) - Simple web-based tool
- [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) - Browser-based

---

## Training

### Quick Start

Basic training with default parameters:

```bash
python main.py \
  --config_file config/DINO/DINO_4scale.py \
  --output_dir outputs/pothole_finetune \
  --coco_path datasets/pothole \
  --pretrain_model_path checkpoints/checkpoint0011_4scale.pth \
  --options epochs=10 lr=1e-5 batch_size=1 \
  --device cuda \
  --amp
```

### Training Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--config_file` | `config/DINO/DINO_4scale.py` | Model architecture config (4-scale DINO) |
| `--output_dir` | `outputs/pothole_finetune` | Directory for checkpoints and logs |
| `--coco_path` | `datasets/pothole` | Path to COCO-format dataset |
| `--pretrain_model_path` | `checkpoints/checkpoint0011_4scale.pth` | Pretrained COCO weights |
| `--options epochs` | 10 | Number of training epochs |
| `--options lr` | 1e-5 | Learning rate (low for fine-tuning) |
| `--options batch_size` | 1 | Batch size (adjust based on GPU memory) |
| `--options lr_drop` | 8 | Epoch to reduce learning rate (default: 8) |
| `--num_workers` | 2 | Data loading workers |
| `--device` | cuda | Use GPU (set to 'cpu' for CPU training) |
| `--amp` | - | Enable mixed precision training |

### Advanced Training

#### Custom Learning Rate Schedule
```bash
python main.py \
  --config_file config/DINO/DINO_4scale.py \
  --coco_path datasets/pothole \
  --pretrain_model_path checkpoints/checkpoint0011_4scale.pth \
  --options epochs=15 lr=5e-5 batch_size=2 lr_drop=10 \
  --output_dir outputs/custom_run \
  --device cuda \
  --amp
```

#### Resume Training from Checkpoint
```bash
python main.py \
  --config_file config/DINO/DINO_4scale.py \
  --coco_path datasets/pothole \
  --resume outputs/pothole_finetune/checkpoint.pth \
  --output_dir outputs/pothole_finetune \
  --device cuda \
  --amp
```

#### Training with Different Backbone
```bash
# Use ConvNeXt backbone
python main.py \
  --config_file config/DINO/DINO_4scale_convnext.py \
  --coco_path datasets/pothole \
  --pretrain_model_path checkpoints/checkpoint_convnext.pth \
  --options epochs=10 lr=1e-5 \
  --output_dir outputs/pothole_convnext \
  --device cuda \
  --amp
```

### Training Workflow

1. **Initialization** (Lines 141-180 in `main.py`)
   - Setup distributed training
   - Load configuration from config file
   - Initialize logging

2. **Model Building** (Lines 186-221)
   - Build DINO architecture (ResNet-50 + Transformer)
   - Setup loss functions (Focal loss + L1 + GIoU)
   - Initialize optimizer and scheduler

3. **Dataset Loading** (Lines 227-252)
   - Load training and validation datasets
   - Apply data augmentations
   - Create data loaders

4. **Training Loop** (Lines 370-498)
   - For each epoch:
     - Train on training set
     - Update learning rate
     - Validate on validation set
     - Save checkpoints
     - Track best model

### Expected Training Timeline

| Epochs | Phase | Expected mAP | Description |
|--------|-------|--------------|-------------|
| 1-3 | Initial Adaptation | 0.1 → 0.3 | Rapid loss decrease, class head adaptation |
| 4-7 | Refinement | 0.3 → 0.4 | Better box localization |
| 8 | LR Drop | - | Learning rate reduced by 10x |
| 9-10 | Convergence | 0.4 → 0.6 | Fine-tuning, final optimization |

---

## Evaluation

### Evaluate Trained Model

```bash
python main.py \
  --config_file config/DINO/DINO_4scale.py \
  --coco_path datasets/pothole \
  --resume outputs/pothole_finetune/checkpoint_best_regular.pth \
  --output_dir outputs/evaluation \
  --eval \
  --device cuda
```

### Evaluation Metrics

The model evaluates using COCO metrics:

#### Primary Metric: mAP@[0.5:0.95]
- Mean Average Precision across IoU thresholds from 0.5 to 0.95
- **Range**: 0.0 (worst) to 1.0 (perfect)
- **Good performance**: > 0.3 for small datasets, > 0.5 for larger datasets

#### Additional Metrics
- **mAP@0.50**: Precision at loose IoU threshold (0.5)
- **mAP@0.75**: Precision at strict IoU threshold (0.75)
- **mAP small**: Precision for small objects (< 32² pixels)
- **mAP medium**: Precision for medium objects (32² - 96² pixels)
- **mAP large**: Precision for large objects (> 96² pixels)

### Example Results

```
Epoch 10 Validation Results:
┌─────────────────────────┬────────┐
│ Metric                  │ Score  │
├─────────────────────────┼────────┤
│ mAP@[0.5:0.95]          │ 0.456  │
│ mAP@0.50                │ 0.678  │
│ mAP@0.75                │ 0.512  │
│ mAP (small)             │ 0.234  │
│ mAP (medium)            │ 0.567  │
│ mAP (large)             │ 0.689  │
└─────────────────────────┴────────┘
```

### Visualizing Results

Results are saved in `outputs/evaluation/eval/`:
- `latest.pth`: Latest evaluation results
- COCO-format metrics in JSON format

---

## Inference

### Run Inference on Single Image

Create an inference script `inference.py`:

```python
import torch
from main import build_model_main
from util.slconfig import SLConfig
from util.visualizer import visualization
import cv2

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = SLConfig.fromfile('config/DINO/DINO_4scale.py')
model, criterion, postprocessors = build_model_main(cfg)
model.load_state_dict(torch.load('outputs/pothole_finetune/checkpoint_best_regular.pth', map_location='cpu')['model'])
model.to(device)
model.eval()

# Load image
image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
with torch.no_grad():
    outputs = model(image)

# Visualize results
visualization(image, outputs, 'output.jpg')
```

### Batch Inference

```python
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

images = [Image.open(f) for f in image_paths]
batch = torch.stack([transform(img) for img in images]).to(device)

with torch.no_grad():
    outputs = model(batch)

# Postprocess predictions
probas = outputs['pred_logits'].softmax(-1)[..., :-1].cpu().numpy()
keep = probas.max(-1) > 0.3  # Confidence threshold
```

---

## Project Structure

```
DINO/
├── main.py                          # Main training script
├── engine.py                        # Training loop and evaluation
├── requirements.txt                 # Python dependencies
│
├── config/
│   └── DINO/
│       ├── DINO_4scale.py          # 4-scale DINO config (used here)
│       ├── DINO_5scale.py          # 5-scale variant
│       ├── DINO_4scale_convnext.py # ConvNeXt backbone
│       └── DINO_4scale_swin.py     # Swin Transformer backbone
│
├── models/
│   └── dino/
│       ├── dino.py                 # Core DINO detector
│       ├── deformable_transformer.py # Transformer architecture
│       ├── backbone.py             # ResNet-50 backbone
│       ├── matcher.py              # Hungarian matching
│       ├── dn_components.py        # Denoising components
│       ├── attention.py            # Multi-scale deformable attention
│       └── position_encoding.py    # Positional embeddings
│
├── util/
│   ├── box_ops.py                  # Bounding box operations
│   ├── box_loss.py                 # Loss functions
│   ├── utils.py                    # General utilities
│   ├── logger.py                   # Logging setup
│   ├── slconfig.py                 # Configuration loader
│   └── vis_utils.py                # Visualization tools
│
├── checkpoints/
│   └── checkpoint0011_4scale.pth   # Pretrained COCO weights
│
├── outputs/
│   └── pothole_finetune/
│       ├── checkpoint.pth           # Latest model
│       ├── checkpoint_best_regular.pth  # Best validation mAP model
│       ├── checkpoint0008.pth       # Milestone checkpoint
│       ├── log.txt                  # Training metrics (JSON)
│       ├── info.txt                 # Detailed logs
│       └── eval/                    # Evaluation results
│
└── datasets/
    └── pothole/
        ├── train2017/              # Training images
        ├── val2017/                # Validation images
        └── annotations/            # COCO annotations
```

---

## Results

### Training Outputs

After training, you'll find:

#### Checkpoints
- `checkpoint.pth` - Latest model (saved every epoch)
- `checkpoint_best_regular.pth` - **Best model** (highest validation mAP)
- `checkpoint{epoch:04}.pth` - Milestone checkpoints (e.g., epoch 8, 10)

#### Logs
- `log.txt` - JSON format metrics per epoch
- `info.txt` - Detailed training log with timestamps

#### Example Log Entry
```json
{
  "train_loss": 2.3456,
  "train_class_error": 12.34,
  "train_loss_ce": 0.456,
  "train_loss_bbox": 0.234,
  "train_loss_giou": 0.123,
  "test_coco_eval_bbox": [0.456, 0.678, 0.512, 0.234, 0.567, 0.689],
  "epoch": 5,
  "n_parameters": 47000000,
  "best_regular_map": 0.678,
  "best_regular_epoch": 5
}
```

### Performance Tips

#### Improve Accuracy
1. **More training data** - 500+ images recommended
2. **Better annotations** - Precise bounding boxes
3. **Data augmentation** - Already enabled, can be customized
4. **Longer training** - Try 15-20 epochs
5. **Higher resolution** - Modify image size in config

#### Improve Speed
1. **Larger batch size** - Use gradient accumulation
2. **Fewer workers** - Reduce `num_workers` if IO is fast
3. **Smaller backbone** - Use ResNet-18 instead of ResNet-50
4. **Disable AMP** - If GPU doesn't support it well

---

## Configuration

### Model Config (`config/DINO/DINO_4scale.py`)

Key configuration options:

```python
# Model parameters
model = dict(
    type='DINO',
    backbone=dict(
        type='ResNet',
        depth=50,              # ResNet-50
        frozen_stages=-1,      # -1 = train all layers
    ),
    encoder=dict(
        num_layers=6,          # Transformer encoder layers
        num_feature_levels=4,  # Multi-scale features
    ),
    decoder=dict(
        num_layers=6,          # Transformer decoder layers
        num_queries=300,       # Object queries
    ),
)

# Training parameters
lr = 1e-5
batch_size = 1
epochs = 10
lr_drop = 8
weight_decay = 1e-4

# Loss weights
loss_weights = dict(
    loss_class=2.0,
    loss_bbox=5.0,
    loss_giou=2.0,
)
```

### Customizing Configuration

Edit the config file or use command-line options:

```bash
# Override config values via command line
python main.py \
  --config_file config/DINO/DINO_4scale.py \
  --options \
    epochs=20 \
    lr=5e-5 \
    batch_size=2 \
    lr_drop=15 \
    weight_decay=1e-3 \
  --coco_path datasets/pothole \
  --device cuda
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--options batch_size=1`
- Use gradient accumulation
- Reduce image size in config
- Use smaller backbone (ResNet-18/34)

#### 2. Loss is NaN
**Error**: Loss becomes NaN during training

**Solutions**:
- Lower learning rate: `--options lr=1e-6`
- Check for corrupted images
- Verify annotation format
- Enable gradient clipping

#### 3. mAP not Improving
**Issue**: Validation mAP stagnates

**Solutions**:
- Train for more epochs
- Increase learning rate slightly
- Check annotation quality
- Add more training data
- Try different backbone (Swin, ConvNeXt)

#### 4. Slow Training
**Issue**: Training takes too long

**Solutions**:
- Enable mixed precision: `--amp`
- Increase `num_workers`
- Use faster GPU
- Reduce image resolution
- Use smaller backbone

#### 5. No Objects Detected
**Issue**: Model outputs empty predictions

**Solutions**:
- Check confidence threshold
- Verify class labels match
- Ensure pretrained weights loaded correctly
- Increase training epochs
- Add more diverse training data

### Getting Help

1. Check `log.txt` and `info.txt` for detailed error messages
2. Verify dataset format with COCO API
3. Test with pretrained COCO model first
4. Review GitHub issues in original DINO repository

---

## Fine-Tuning Strategy

### Why Low Learning Rate (1e-5)?
- Pretrained weights are already good for general object detection
- High learning rate would destroy learned features
- Low rate allows small, targeted adjustments for potholes

### Why 10 Epochs?
- Pretrained model converges quickly on new task
- More epochs risk overfitting on small dataset
- Early stopping based on validation mAP

### Why Batch Size 1?
- **Memory constraints**: DINO is memory-intensive
- **Small dataset**: Limited number of pothole images
- **Gradient accumulation**: Can simulate larger effective batches

### Transfer Learning Process

```
ImageNet Pretrained (ResNet-50)
         ↓
COCO Pretrained (11 epochs, 80 classes)
         ↓
Load checkpoint0011_4scale.pth
         ↓
Adapt class head (pothole classes)
         ↓
Fine-tune with low learning rate (1e-5)
         ↓
Pothole Detection Model
```

**Benefits**:
- **Feature reuse**: Potholes share visual features with COCO objects
- **Spatial understanding**: Model already learned object localization
- **Data efficiency**: Need fewer pothole images than training from scratch

---

## Citation

If you use this implementation in your research, please cite the original DINO paper:

```bibtex
@inproceedings{dino2022,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Yue and Kang, Bing and Zhang, Liangwei and Yang, Wenjia and Li, Junyu and Yang, Zhiyuan and Zhang, Wenyu},
  booktitle={TPAMI},
  year={2022}
}
```

Also cite Deformable DETR:

```bibtex
@inproceedings{deformable_detr,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  booktitle={ICLR},
  year={2021}
}
```

---

## License

This project is based on the [DINO implementation](https://github.com/IDEA-Research/DINO). Please refer to the original repository for licensing information.

---

## Acknowledgments

- Original DINO implementation by [IDEA Research](https://github.com/IDEA-Research/DINO)
- Deformable DETR by [SenseTime Research](https://github.com/fundamentalvision/Deformable-DETR)
- DETR by [Facebook AI Research](https://github.com/facebookresearch/detr)

---

## Contact

For questions about this pothole detection implementation, please open an issue in the repository or refer to the original DINO repository for general questions about the architecture.

---

**Last Updated**: January 2026
**Project**: DINO Pothole Detection Fine-tuning
**Pretrained Checkpoint**: `checkpoint0011_4scale.pth` (COCO weights)
