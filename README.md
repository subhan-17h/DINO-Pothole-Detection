# DINO Pothole Detection

Fine-tuned DINO (DETR with Improved DeNoising Anchor Boxes) for automatic pothole detection. Uses COCO-pretrained weights (ResNet-50, 4-scale deformable transformer) with transfer learning.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset Structure

```
datasets/pothole/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Annotations must be in COCO format. Minimum: 100 train / 20 val images.

---

## Training

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

**Resume from checkpoint:**
```bash
python main.py \
  --config_file config/DINO/DINO_4scale.py \
  --coco_path datasets/pothole \
  --resume outputs/pothole_finetune/checkpoint.pth \
  --output_dir outputs/pothole_finetune \
  --device cuda \
  --amp
```

---

## Evaluation

```bash
python main.py \
  --config_file config/DINO/DINO_4scale.py \
  --coco_path datasets/pothole \
  --resume outputs/pothole_finetune/checkpoint_best_regular.pth \
  --output_dir outputs/evaluation \
  --eval \
  --device cuda
```

Metrics reported: mAP@[0.5:0.95], mAP@0.50, mAP@0.75, mAP small/medium/large.

---

## Local CPU Video Inference (M1/Mac)

Use the fine-tuned checkpoint to run inference on a local video and force displayed label text to `pothole`.

```bash
source /Users/rowdy/Projects/DINO/venv/bin/activate
export PYTHONPATH=/Users/rowdy/Projects/DINO
export TORCH_HOME=/Users/rowdy/Projects/DINO/.torchcache
export MPLCONFIGDIR=/Users/rowdy/Projects/DINO/.mplconfig

python /Users/rowdy/Projects/DINO/tools/video_inference.py \
  --config /Users/rowdy/Projects/DINO/outputs/pothole_finetune/config_cfg.py \
  --checkpoint /Users/rowdy/Projects/DINO/outputs/pothole_finetune/checkpoint.pth \
  --input-video /Users/rowdy/Projects/DINO/video.mp4 \
  --output-video /Users/rowdy/Projects/DINO/outputs/pothole_finetune/local_cpu_result_pothole.mp4 \
  --device cpu \
  --score-threshold 0.30 \
  --single-class-name pothole
```

---

## Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 10 | Training epochs |
| `lr` | 1e-5 | Learning rate |
| `batch_size` | 1 | Batch size |
| `lr_drop` | 8 | Epoch to drop LR by 10x |
| `--amp` | — | Mixed precision training |

---

## Project Structure

```
DINO/
├── main.py                     # Training & evaluation entry point
├── engine.py                   # Training loop
├── config/DINO/DINO_4scale.py  # Model config
├── models/dino/                # DINO architecture
├── datasets/pothole/           # Dataset
├── checkpoints/                # Pretrained weights
└── outputs/pothole_finetune/   # Checkpoints & logs
```

---

## Outputs

- `checkpoint.pth` — latest checkpoint
- `checkpoint_best_regular.pth` — best validation mAP
- `log.txt` — per-epoch metrics (JSON)

---

## Troubleshooting

- **OOM**: reduce batch size or image resolution
- **NaN loss**: lower learning rate to `1e-6`
- **Low mAP**: train longer, add more data, check annotation quality
- **Slow training**: enable `--amp`, increase `--num_workers`

---

## License

Based on [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO).
