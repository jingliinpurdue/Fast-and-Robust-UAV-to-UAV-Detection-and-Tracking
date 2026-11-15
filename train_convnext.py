"""Train and validate torchvision ConvNeXt UAV detectors.

Each frame may contain zero or more UAV boxes. The detector predicts a fixed
number of objectness/box slots; box coordinates use normalized center-x,
center-y, width, and height values. Frames are sampled from each clip and the
train/validation split is always made at the clip level.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_small,
    convnext_tiny,
)
from torchvision.models.convnext import ConvNeXt

from yolo_detector import find_video_annotation_pairs, parse_annotation_file


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "Data" / "Videos"
DEFAULT_ANNOTATION_DIR = ROOT / "Data" / "Annotation_update_180925"
DEFAULT_OUTPUT_DIR = ROOT / "Experiment_Results" / "ConvNeXt"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SLOT_COMPONENTS = 5

LOGGER = logging.getLogger("convnext_training")


@dataclass(frozen=True)
class FrameRecord:
    """One cached frame and its normalized target boxes."""

    image_path: Path
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class VideoRecord:
    """A clip, its annotation file, and parsed frame-indexed boxes."""

    video_id: int
    video_path: Path
    annotation_path: Path
    annotations: dict[int, list[tuple[int, int, int, int]]]


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_video_ids(video_ids: Sequence[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Return disjoint deterministic train/validation clip IDs."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be greater than 0 and less than 1")
    unique_ids = sorted(set(video_ids))
    if len(unique_ids) < 2:
        raise ValueError(f"At least two matched clips are required; found {len(unique_ids)}")

    shuffled_ids = unique_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)
    val_count = min(len(shuffled_ids) - 1, max(1, round(len(shuffled_ids) * val_fraction)))
    val_ids = sorted(shuffled_ids[:val_count])
    train_ids = sorted(shuffled_ids[val_count:])
    return train_ids, val_ids


def _source_signature(path: Path) -> dict[str, int]:
    """Return metadata used to detect stale extracted-frame caches."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _normalize_boxes(
    boxes: Sequence[tuple[int, int, int, int]], width: int, height: int, context: str
) -> tuple[tuple[float, float, float, float], ...]:
    """Convert top/left/bottom/right boxes to normalized center/size boxes."""
    normalized = []
    for top, left, bottom, right in boxes:
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"{context}: invalid or out-of-frame box {(top, left, bottom, right)}")
        normalized.append(
            (
                (left + right) / (2.0 * width),
                (top + bottom) / (2.0 * height),
                (right - left) / width,
                (bottom - top) / height,
            )
        )
    return tuple(sorted(normalized, key=lambda box: (box[0], box[1], box[2], box[3])))


def extract_annotated_frames(
    video: VideoRecord, cache_dir: Path, frame_stride: int
) -> list[FrameRecord]:
    """Sequentially extract annotated frames, reusing a valid on-disk cache."""
    selected_frames = [
        frame_number
        for frame_number in sorted(video.annotations)
        if (frame_number - 1) % frame_stride == 0
    ]
    if not selected_frames:
        raise ValueError(f"{video.annotation_path}: no annotated frames selected by stride {frame_stride}")

    clip_cache = cache_dir / f"video_{video.video_id:03d}"
    clip_cache.mkdir(parents=True, exist_ok=True)
    manifest_path = clip_cache / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring invalid frame-cache manifest: %s", manifest_path)

    expected_manifest = {
        "video": _source_signature(video.video_path),
        "annotations": _source_signature(video.annotation_path),
        "frame_stride": frame_stride,
        "frame_numbers": selected_frames,
    }
    dimensions = manifest.get("dimensions", {})
    cache_is_valid = all(manifest.get(key) == value for key, value in expected_manifest.items())
    cache_is_valid = cache_is_valid and isinstance(dimensions, dict)
    cache_is_valid = cache_is_valid and all(
        (clip_cache / f"frame_{frame_number:06d}.jpg").is_file()
        and str(frame_number) in dimensions
        for frame_number in selected_frames
    )

    if not cache_is_valid:
        capture = cv2.VideoCapture(str(video.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video.video_path}")
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_frames > 0 and max(video.annotations) > reported_frames:
            capture.release()
            raise ValueError(
                f"{video.video_path}: annotations extend through frame {max(video.annotations)}, "
                f"but the video reports {reported_frames} frames"
            )

        dimensions = {}
        selected_set = set(selected_frames)
        try:
            for frame_number in range(1, max(video.annotations) + 1):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not decode frame {frame_number} from {video.video_path}"
                    )
                if frame_number not in selected_set:
                    continue
                height, width = image.shape[:2]
                image_path = clip_cache / f"frame_{frame_number:06d}.jpg"
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Could not write cached frame: {image_path}")
                dimensions[str(frame_number)] = [width, height]
        finally:
            capture.release()

        manifest = {**expected_manifest, "dimensions": dimensions}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    records = []
    for frame_number in selected_frames:
        width, height = dimensions[str(frame_number)]
        image_path = clip_cache / f"frame_{frame_number:06d}.jpg"
        records.append(
            FrameRecord(
                image_path=image_path,
                width=int(width),
                height=int(height),
                boxes=_normalize_boxes(
                    video.annotations[frame_number],
                    int(width),
                    int(height),
                    f"{video.annotation_path}: frame {frame_number}",
                ),
            )
        )
    return records


class AirToAirFrameDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load cached RGB frames and letterbox images without distorting UAV boxes."""

    def __init__(self, records: Sequence[FrameRecord], image_size: int, max_boxes: int) -> None:
        if not records:
            raise ValueError("A train and validation dataset must each contain at least one frame")
        self.records = list(records)
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read cached image: {record.image_path}")
        height, width = image.shape[:2]
        if width != record.width or height != record.height:
            raise ValueError(
                f"Cached image dimensions changed for {record.image_path}: "
                f"expected {(record.width, record.height)}, got {(width, height)}"
            )

        rgb_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        scale = min(self.image_size / width, self.image_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = transforms.functional.resize(
            rgb_image, [resized_height, resized_width], antialias=True
        )
        left_pad = (self.image_size - resized_width) // 2
        top_pad = (self.image_size - resized_height) // 2
        right_pad = self.image_size - resized_width - left_pad
        bottom_pad = self.image_size - resized_height - top_pad
        padded = transforms.functional.pad(
            resized,
            [left_pad, top_pad, right_pad, bottom_pad],
            fill=(124, 116, 104),
        )
        image_tensor = self.normalize(transforms.functional.to_tensor(padded))

        scale_x = resized_width / width
        scale_y = resized_height / height
        target = torch.zeros((self.max_boxes, SLOT_COMPONENTS), dtype=torch.float32)
        for slot, (center_x, center_y, box_width, box_height) in enumerate(record.boxes):
            if slot >= self.max_boxes:
                raise ValueError(
                    f"{record.image_path}: found more than {self.max_boxes} boxes; "
                    "the model output size was computed incorrectly"
                )
            target[slot] = torch.tensor(
                [
                    1.0,
                    (center_x * width * scale_x + left_pad) / self.image_size,
                    (center_y * height * scale_y + top_pad) / self.image_size,
                    box_width * width * scale_x / self.image_size,
                    box_height * height * scale_y / self.image_size,
                ],
                dtype=torch.float32,
            )
        return image_tensor, target


def create_model(model_name: str, max_boxes: int, pretrained: bool) -> ConvNeXt:
    """Load a torchvision ConvNeXt and replace its classifier with box slots."""
    model_specs: dict[str, Callable[[bool], ConvNeXt]] = {
        "tiny": lambda use_pretrained: convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT if use_pretrained else None
        ),
        "small": lambda use_pretrained: convnext_small(
            weights=ConvNeXt_Small_Weights.DEFAULT if use_pretrained else None
        ),
        "base": lambda use_pretrained: convnext_base(
            weights=ConvNeXt_Base_Weights.DEFAULT if use_pretrained else None
        ),
    }
    model = model_specs[model_name](pretrained)
    classifier = model.classifier[-1]
    if not isinstance(classifier, nn.Linear):
        raise TypeError(f"Unexpected {model_name} ConvNeXt classifier: {type(classifier).__name__}")
    model.classifier[-1] = nn.Linear(classifier.in_features, max_boxes * SLOT_COMPONENTS)
    return model


def detector_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Combine per-slot objectness BCE with regression on occupied box slots."""
    predictions = predictions.float().reshape_as(targets)
    objectness_loss = nn.functional.binary_cross_entropy_with_logits(
        predictions[..., 0], targets[..., 0]
    )
    positive = targets[..., 0] > 0.5
    if positive.any():
        predicted_boxes = predictions[..., 1:][positive].sigmoid()
        target_boxes = targets[..., 1:][positive]
        box_loss = nn.functional.smooth_l1_loss(predicted_boxes, target_boxes)
    else:
        box_loss = predictions[..., 1:].sum() * 0.0
    return objectness_loss + box_loss


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU for normalized center-x, center-y, width, height boxes."""
    ax1 = max(0.0, box_a[0] - box_a[2] / 2.0)
    ay1 = max(0.0, box_a[1] - box_a[3] / 2.0)
    ax2 = min(1.0, box_a[0] + box_a[2] / 2.0)
    ay2 = min(1.0, box_a[1] + box_a[3] / 2.0)
    bx1 = max(0.0, box_b[0] - box_b[2] / 2.0)
    by1 = max(0.0, box_b[1] - box_b[3] / 2.0)
    bx2 = min(1.0, box_b[0] + box_b[2] / 2.0)
    by2 = min(1.0, box_b[1] + box_b[3] / 2.0)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def decode_boxes(prediction: Tensor, confidence_threshold: float) -> list[list[float]]:
    """Convert one frame's output slots into confident normalized boxes."""
    slot_probabilities = prediction[:, 0].sigmoid()
    boxes = prediction[:, 1:].sigmoid()
    decoded = []
    for probability, box in zip(slot_probabilities.tolist(), boxes.tolist()):
        if probability >= confidence_threshold:
            decoded.append([min(1.0, max(0.0, value)) for value in box])
    return decoded


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, float]:
    """Compute validation loss and greedy IoU-matched detection metrics."""
    model.eval()
    loss_total = 0.0
    sample_count = 0
    true_positives = false_positives = false_negatives = 0
    matched_ious: list[float] = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            predictions = model(images).reshape_as(targets)
            loss_total += detector_loss(predictions, targets).item() * images.shape[0]
            sample_count += images.shape[0]

            for prediction, target in zip(predictions.cpu(), targets.cpu()):
                predicted_boxes = decode_boxes(prediction, confidence_threshold)
                target_boxes = target[target[:, 0] > 0.5, 1:].tolist()
                candidates = sorted(
                    (
                        (box_iou(predicted_box, target_box), pred_index, target_index)
                        for pred_index, predicted_box in enumerate(predicted_boxes)
                        for target_index, target_box in enumerate(target_boxes)
                    ),
                    reverse=True,
                )
                matched_predictions: set[int] = set()
                matched_targets: set[int] = set()
                for overlap, pred_index, target_index in candidates:
                    if overlap < iou_threshold:
                        break
                    if pred_index not in matched_predictions and target_index not in matched_targets:
                        matched_predictions.add(pred_index)
                        matched_targets.add(target_index)
                        matched_ious.append(overlap)
                true_positives += len(matched_predictions)
                false_positives += len(predicted_boxes) - len(matched_predictions)
                false_negatives += len(target_boxes) - len(matched_targets)

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    metric_suffix = f"@{iou_threshold:g}"
    return {
        "loss": loss_total / max(1, sample_count),
        f"precision{metric_suffix}": precision,
        f"recall{metric_suffix}": recall,
        f"f1{metric_suffix}": f1,
        f"mean_iou{metric_suffix}": sum(matched_ious) / max(1, len(matched_ious)),
    }


class EarlyStopping:
    """Save the best weights and stop after validation loss stops improving."""

    def __init__(self, patience: int, checkpoint_path: Path) -> None:
        if patience < 1:
            raise ValueError("--patience must be at least 1")
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, validation_loss: float, model: nn.Module) -> bool:
        """Save a new best checkpoint; return True when patience is exhausted."""
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.bad_epochs = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_one_model(
    model_name: str,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    max_boxes: int,
) -> dict[str, float]:
    """Train one architecture and return metrics from its best checkpoint."""
    LOGGER.info("Starting ConvNeXt-%s", model_name.title())
    model = create_model(model_name, max_boxes, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = args.output_dir / f"convnext_{model_name}_best.pth"
    early_stopping = EarlyStopping(args.patience, checkpoint_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        sample_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                predictions = model(images).reshape_as(targets)
                loss = detector_loss(predictions, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_total += loss.detach().item() * images.shape[0]
            sample_count += images.shape[0]

        train_loss = train_loss_total / max(1, sample_count)
        validation = evaluate(
            model,
            val_loader,
            device,
            args.confidence_threshold,
            args.iou_threshold,
        )
        if not math.isfinite(validation["loss"]):
            raise RuntimeError(f"Validation loss became non-finite for ConvNeXt-{model_name.title()}")
        scheduler.step()
        metric_suffix = f"@{args.iou_threshold:g}"
        LOGGER.info(
            "ConvNeXt-%s epoch %d/%d: train_loss=%.5f val_loss=%.5f "
            "precision%s=%.4f recall%s=%.4f f1%s=%.4f mean_iou%s=%.4f lr=%.3g",
            model_name.title(),
            epoch,
            args.epochs,
            train_loss,
            validation["loss"],
            metric_suffix,
            validation[f"precision{metric_suffix}"],
            metric_suffix,
            validation[f"recall{metric_suffix}"],
            metric_suffix,
            validation[f"f1{metric_suffix}"],
            metric_suffix,
            validation[f"mean_iou{metric_suffix}"],
            optimizer.param_groups[0]["lr"],
        )
        if early_stopping.step(validation["loss"], model):
            LOGGER.info("Early stopping ConvNeXt-%s after epoch %d", model_name.title(), epoch)
            break

    best_weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_weights)
    best_metrics = evaluate(
        model,
        val_loader,
        device,
        args.confidence_threshold,
        args.iou_threshold,
    )
    LOGGER.info(
        "Best ConvNeXt-%s checkpoint: %s; validation metrics: %s",
        model_name.title(),
        checkpoint_path,
        best_metrics,
    )
    del model, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics


def select_device(device_name: str) -> torch.device:
    """Resolve an explicit device or choose CUDA, MPS, then CPU."""
    if device_name != "auto":
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("An MPS device was requested, but MPS is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for data preparation and training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-stride", type=int, default=10, help="Use every Nth annotated frame")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a specific device")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    """Prepare clip-level splits and train Tiny, Small, and Base in sequence."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    if args.frame_stride < 1 or args.image_size < 32 or args.batch_size < 1:
        raise ValueError("--frame-stride and --batch-size must be positive; --image-size must be at least 32")
    if args.epochs < 1 or args.num_workers < 0 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive; --num-workers cannot be negative")
    if args.seed < 0:
        raise ValueError("--seed cannot be negative")
    if not 0.0 <= args.confidence_threshold <= 1.0 or not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("Confidence and IoU thresholds must be between 0 and 1")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("--learning-rate must be positive and --weight-decay non-negative")

    set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    LOGGER.info("Training on %s", device)

    pairs = find_video_annotation_pairs(args.videos_dir, args.annotations_dir)
    videos = [
        VideoRecord(video_id, video_path, annotation_path, parse_annotation_file(annotation_path))
        for video_id, video_path, annotation_path in pairs
    ]
    train_ids, val_ids = split_video_ids([video.video_id for video in videos], args.val_fraction, args.seed)
    train_id_set, val_id_set = set(train_ids), set(val_ids)
    if train_id_set & val_id_set:
        raise RuntimeError("Train and validation clip splits overlap")
    LOGGER.info("Train clips (%d): %s", len(train_ids), train_ids)
    LOGGER.info("Validation clips (%d): %s", len(val_ids), val_ids)

    cache_dir = args.output_dir / "frame_cache"
    records_by_video = {
        video.video_id: extract_annotated_frames(video, cache_dir, args.frame_stride)
        for video in videos
    }
    train_records = [record for video_id in train_ids for record in records_by_video[video_id]]
    val_records = [record for video_id in val_ids for record in records_by_video[video_id]]
    max_boxes = max(len(record.boxes) for records in records_by_video.values() for record in records)
    if max_boxes < 1:
        raise ValueError("The selected annotation files contain no bounding boxes")
    LOGGER.info(
        "Prepared %d training and %d validation frames; model predicts %d boxes per frame",
        len(train_records),
        len(val_records),
        max_boxes,
    )

    train_dataset = AirToAirFrameDataset(train_records, args.image_size, max_boxes)
    val_dataset = AirToAirFrameDataset(val_records, args.image_size, max_boxes)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    results = {}
    for model_name in ("tiny", "small", "base"):
        results[model_name] = train_one_model(model_name, train_loader, val_loader, args, device, max_boxes)
    summary = {
        "train_video_ids": train_ids,
        "validation_video_ids": val_ids,
        "frame_stride": args.frame_stride,
        "image_size": args.image_size,
        "max_boxes": max_boxes,
        "results": results,
    }
    summary_path = args.output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Training complete; validation summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
