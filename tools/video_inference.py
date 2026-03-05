import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig


def parse_args():
    parser = argparse.ArgumentParser("Run DINO pothole inference on video.")
    parser.add_argument(
        "--config",
        default="outputs/pothole_finetune/config_cfg.py",
        help="Path to model config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/pothole_finetune/checkpoint.pth",
        help="Path to model checkpoint .pth file.",
    )
    parser.add_argument(
        "--input-video",
        required=True,
        help="Path to input video file. Use 0 for webcam.",
    )
    parser.add_argument(
        "--output-video",
        default="outputs/pothole_finetune/inference_output.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.30,
        help="Keep detections above this confidence.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--class-map",
        default="util/coco_id2name.json",
        help="JSON mapping from class id to class name.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=800,
        help="Short side resize used before inference.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1333,
        help="Max side resize used before inference.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional frame limit for quick tests (-1 = all frames).",
    )
    parser.add_argument(
        "--single-class-name",
        default=None,
        help="If set, use this class name for every detection (e.g., pothole).",
    )
    return parser.parse_args()


def load_class_map(path):
    class_map_path = Path(path)
    if not class_map_path.exists():
        return {}
    with class_map_path.open("r") as f:
        data = json.load(f)
    return {int(k): str(v) for k, v in data.items()}


def build_transform(resize, max_size):
    return T.Compose(
        [
            T.RandomResize([resize], max_size=max_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_model(config_path, checkpoint_path, device):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, postprocessors


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    return x1, y1, x2, y2


@torch.no_grad()
def infer_frame(model, postprocessors, transform, frame_bgr, device, score_threshold):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    image_tensor, _ = transform(pil_img, None)
    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor[None])
    h, w = frame_bgr.shape[:2]
    target_sizes = torch.tensor([[h, w]], dtype=torch.float32, device=device)
    output = postprocessors["bbox"](outputs, target_sizes)[0]

    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()

    keep = scores > score_threshold
    return boxes[keep], labels[keep], scores[keep]


def draw_detections(frame, boxes, labels, scores, id2name, single_class_name=None):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = clamp_box(box, frame.shape[1], frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 255, 40), 2)
        if single_class_name:
            class_name = single_class_name
        else:
            class_name = id2name.get(int(label), f"class_{int(label)}")
        text = f"{class_name}: {float(score):.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )
    return frame


def open_capture(input_video):
    if str(input_video) == "0":
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(str(input_video))


def main():
    args = parse_args()
    id2name = load_class_map(args.class_map)
    transform = build_transform(args.resize, args.max_size)
    model, postprocessors = load_model(args.config, args.checkpoint, args.device)

    cap = open_capture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes, labels, scores = infer_frame(
            model=model,
            postprocessors=postprocessors,
            transform=transform,
            frame_bgr=frame,
            device=args.device,
            score_threshold=args.score_threshold,
        )
        frame = draw_detections(
            frame, boxes, labels, scores, id2name, single_class_name=args.single_class_name
        )
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"Processed {frame_idx} frames...", flush=True)
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"Done. Processed {frame_idx} frames.")
    print(f"Saved output video to: {output_path}")


if __name__ == "__main__":
    main()
