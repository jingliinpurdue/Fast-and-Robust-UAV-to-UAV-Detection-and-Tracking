"""Prepare, train, evaluate, and run a frame-wise YOLO UAV detector."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "Data" / "Videos"
ANNOTATION_DIR = ROOT / "Data" / "Annotation_update_180925"
YOLO_DIR = ROOT / "Experiment_Results" / "YOLO"
DATASET_DIR = YOLO_DIR / "dataset"
RUNS_DIR = YOLO_DIR / "runs"
DEFAULT_WEIGHTS = RUNS_DIR / "uav_detector" / "weights" / "best.pt"

VIDEO_NAME_RE = re.compile(r"^Clip_(\d+)\.mov$")
ANNOTATION_LINE_RE = re.compile(r"^time_layer:\s*(\d+)\s+detections:\s*(.*)$")
BOX_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def parse_annotation_file(path: Path) -> dict[int, list[tuple[int, int, int, int]]]:
    """Read boxes keyed by frame number; coordinates are top,left,bottom,right."""
    annotations: dict[int, list[tuple[int, int, int, int]]] = {}
    previous_frame = 0
    with path.open(encoding="utf-8") as annotation_file:
        for line_number, raw_line in enumerate(annotation_file, start=1):
            match = ANNOTATION_LINE_RE.fullmatch(raw_line.strip())
            if match is None:
                raise ValueError(f"{path}:{line_number}: malformed annotation row")
            frame_number = int(match.group(1))
            if frame_number <= previous_frame:
                raise ValueError(f"{path}:{line_number}: frame IDs must increase")
            previous_frame = frame_number
            box_text = match.group(2)
            boxes = [tuple(map(int, values)) for values in BOX_RE.findall(box_text)]
            remainder = BOX_RE.sub("", box_text).strip(" ,\t")
            if remainder:
                raise ValueError(f"{path}:{line_number}: malformed box data {remainder!r}")
            annotations[frame_number] = boxes
    if not annotations:
        raise ValueError(f"{path}: no annotation rows found")
    return annotations


def find_video_annotation_pairs(
    video_dir: Path, annotation_dir: Path
) -> list[tuple[int, Path, Path]]:
    videos = {
        int(match.group(1)): path
        for path in video_dir.glob("Clip_*.mov")
        if (match := VIDEO_NAME_RE.fullmatch(path.name))
    }
    annotations = {}
    for path in annotation_dir.glob("Video_*_gt.txt"):
        match = re.fullmatch(r"Video_(\d+)_gt\.txt", path.name)
        if match:
            annotations[int(match.group(1))] = path

    video_ids, annotation_ids = set(videos), set(annotations)
    if video_ids - annotation_ids:
        print(f"Skipping videos without annotations: {sorted(video_ids - annotation_ids)}", file=sys.stderr)
    if annotation_ids - video_ids:
        print(f"Skipping annotations without videos: {sorted(annotation_ids - video_ids)}", file=sys.stderr)
    pairs = [
        (video_id, videos[video_id], annotations[video_id])
        for video_id in sorted(video_ids & annotation_ids)
    ]
    if len(pairs) < 3:
        raise ValueError(f"At least three matched video/annotation pairs are required; found {len(pairs)}")
    return pairs


def split_video_ids(video_ids: Iterable[int], seed: int = 42) -> dict[str, list[int]]:
    """Create deterministic video-level splits and hold video 14 out for testing."""
    ids = sorted(set(video_ids))
    if len(ids) < 3:
        raise ValueError("At least three videos are required for train/val/test splits")
    test_count = max(1, len(ids) - int(len(ids) * 0.70) - int(len(ids) * 0.15))
    val_count = max(1, int(len(ids) * 0.15))
    if test_count + val_count >= len(ids):
        raise ValueError("Not enough videos for non-empty train, val, and test splits")

    test_ids = [14] if 14 in ids else []
    remaining = [video_id for video_id in ids if video_id not in test_ids]
    random.Random(seed).shuffle(remaining)
    extra_test_count = test_count - len(test_ids)
    test_ids.extend(remaining[:extra_test_count])
    remaining = remaining[extra_test_count:]
    return {
        "train": sorted(remaining[val_count:]),
        "val": sorted(remaining[:val_count]),
        "test": sorted(test_ids),
    }


def yolo_labels(
    boxes: Iterable[tuple[int, int, int, int]], width: int, height: int
) -> list[str]:
    labels = []
    for top, left, bottom, right in boxes:
        left, right = max(0, min(left, width)), max(0, min(right, width))
        top, bottom = max(0, min(top, height)), max(0, min(bottom, height))
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid or out-of-frame box: {(top, left, bottom, right)}")
        center_x, center_y = ((left + right) / 2) / width, ((top + bottom) / 2) / height
        box_width, box_height = (right - left) / width, (bottom - top) / height
        labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
    return labels


def _remove_previous_outputs(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / "splits.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("generator") != "yolo_detector.py":
        raise ValueError(f"Refusing to replace dataset without a recognized manifest: {manifest_path}")
    for relative_path in manifest.get("generated_files", []):
        output_path = (dataset_dir / relative_path).resolve()
        if not output_path.is_relative_to(dataset_dir.resolve()):
            raise ValueError(f"Unsafe generated-file path in {manifest_path}: {relative_path}")
        if output_path.is_file():
            output_path.unlink()


def prepare_dataset(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir
    frame_stride = args.frame_stride
    seed = args.seed
    if frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    pairs = find_video_annotation_pairs(VIDEO_DIR, ANNOTATION_DIR)
    splits = split_video_ids((video_id for video_id, _, _ in pairs), seed)
    split_by_id = {
        video_id: split_name
        for split_name, video_ids in splits.items()
        for video_id in video_ids
    }
    for split_name in splits:
        (dataset_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    parsed = [(video_id, video, parse_annotation_file(annotation)) for video_id, video, annotation in pairs]
    _remove_previous_outputs(dataset_dir)
    images_written = {name: 0 for name in splits}
    boxes_written = {name: 0 for name in splits}
    generated_files = []

    for video_id, video_path, annotations in parsed:
        split_name = split_by_id[video_id]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_frames > 0 and max(annotations) > reported_frames:
            capture.release()
            raise ValueError(
                f"{video_path}: annotations extend through frame {max(annotations)}, "
                f"but video reports {reported_frames} frames"
            )

        try:
            for frame_number, boxes in annotations.items():
                if (frame_number - 1) % frame_stride:
                    continue
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1):
                    raise RuntimeError(f"Could not seek to frame {frame_number} in {video_path}")
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame_number} from {video_path}")
                height, width = image.shape[:2]
                stem = f"video_{video_id:02d}_frame_{frame_number:06d}"
                image_path = dataset_dir / split_name / "images" / f"{stem}.jpg"
                label_path = dataset_dir / split_name / "labels" / f"{stem}.txt"
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Could not write image: {image_path}")
                labels = yolo_labels(boxes, width, height)
                label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                generated_files.extend((str(image_path.relative_to(dataset_dir)), str(label_path.relative_to(dataset_dir))))
                images_written[split_name] += 1
                boxes_written[split_name] += len(labels)
        finally:
            capture.release()

        if reported_frames and reported_frames > len(annotations):
            missing_frames = sorted(set(range(1, reported_frames + 1)) - set(annotations))
            print(
                f"Video {video_id}: {len(missing_frames)} frame(s) have no annotation row; "
                f"skipped frame IDs: {missing_frames}",
                file=sys.stderr,
            )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {json.dumps(str(dataset_dir.resolve()))}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
                "  0: UAV",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "generator": "yolo_detector.py",
        "seed": seed,
        "frame_stride": frame_stride,
        "split_method": "video-level; video 14 is held out for test when present",
        "splits": splits,
        "images_written": images_written,
        "boxes_written": boxes_written,
        "generated_files": generated_files,
    }
    (dataset_dir / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"YOLO dataset written to {dataset_dir.resolve()}")
    print(f"Data config: {(dataset_dir / 'dataset.yaml').resolve()}")
    for split_name, video_ids in splits.items():
        print(f"{split_name}: {len(video_ids)} videos, {images_written[split_name]} frames, {boxes_written[split_name]} boxes (IDs: {video_ids})")


def load_yolo(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO commands require the optional dependency; install with "
            "`python -m pip install -r requirements-yolo.txt`."
        ) from exc
    return YOLO(weights)


def train_model(args: argparse.Namespace) -> None:
    dataset_yaml = args.dataset_dir / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"{dataset_yaml} is missing; run `python yolo_detector.py prepare` first")
    for split_name in ("train", "val"):
        image_dir = args.dataset_dir / split_name / "images"
        if not any(image_dir.glob("*.jpg")):
            raise ValueError(f"No {split_name} images found in {image_dir}")
        label_files = (args.dataset_dir / split_name / "labels").glob("*.txt")
        if not any(path.read_text(encoding="utf-8").strip() for path in label_files):
            raise ValueError(f"No positive UAV labels found in {args.dataset_dir / split_name / 'labels'}")
    model = load_yolo(args.model)
    model.train(
        data=str(dataset_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(args.project_dir), name=args.name, exist_ok=True,
    )


def evaluate_model(args: argparse.Namespace) -> None:
    dataset_yaml = args.dataset_dir / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"{dataset_yaml} is missing; run `python yolo_detector.py prepare` first")
    if not any((args.dataset_dir / "test" / "images").glob("*.jpg")):
        raise ValueError("No test images found; prepare the dataset before evaluation")
    model = load_yolo(str(args.weights))
    model.val(
        data=str(dataset_yaml), split="test", imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(args.project_dir), name=f"{args.name}_test", exist_ok=True,
    )


def format_detection_line(
    frame_number: int,
    boxes: Iterable[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> str:
    line = f"time_layer: {frame_number} detections: "
    for left, top, right, bottom in boxes:
        left, right = max(0, min(round(left), width)), max(0, min(round(right), width))
        top, bottom = max(0, min(round(top), height)), max(0, min(round(bottom), height))
        if right > left and bottom > top:
            line += f"({top}, {left}, {bottom}, {right}), "
    return line + "\n"


def predict_video(args: argparse.Namespace) -> None:
    video_path = args.video_dir / f"Clip_{args.video_id}.mov"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if expected_frames < 1:
        raise RuntimeError(f"Could not determine frame count for {video_path}")

    model = load_yolo(str(args.weights))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(args.output.name + ".tmp")
    frame_count = 0
    completed = False
    try:
        with temp_output.open("w", encoding="utf-8") as output_file:
            results = model.predict(
                source=str(video_path), stream=True, vid_stride=1, conf=args.confidence,
                imgsz=args.imgsz, device=args.device, verbose=False,
            )
            for frame_count, result in enumerate(results, start=1):
                height, width = result.orig_shape
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                output_file.write(format_detection_line(frame_count, boxes, width, height))
        if frame_count != expected_frames:
            raise RuntimeError(
                f"YOLO processed {frame_count} frames, video reports {expected_frames}; "
                "refusing to publish incomplete output"
            )
        temp_output.replace(args.output)
        completed = True
    finally:
        if not completed and temp_output.exists():
            temp_output.unlink()
    print(f"Processed {frame_count} frames from {video_path}")
    print(f"Detection-only results written to {args.output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="extract frames and YOLO labels")
    prepare.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    prepare.add_argument("--frame-stride", type=int, default=10)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(run=prepare_dataset)

    train = commands.add_parser("train", help="fine-tune a small pretrained YOLO model")
    train.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=960)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default=None)
    train.add_argument("--project-dir", type=Path, default=RUNS_DIR)
    train.add_argument("--name", default="uav_detector")
    train.set_defaults(run=train_model)

    evaluate = commands.add_parser("evaluate", help="evaluate on held-out test videos")
    evaluate.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    evaluate.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    evaluate.add_argument("--imgsz", type=int, default=960)
    evaluate.add_argument("--batch", type=int, default=8)
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--project-dir", type=Path, default=RUNS_DIR)
    evaluate.add_argument("--name", default="uav_detector")
    evaluate.set_defaults(run=evaluate_model)

    predict = commands.add_parser("predict", help="write per-frame detections (no tracking)")
    predict.add_argument("--video-id", type=int, default=14)
    predict.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    predict.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    predict.add_argument("--output", type=Path, default=YOLO_DIR / "txt" / "14_dt_yolo.txt")
    predict.add_argument("--confidence", type=float, default=0.25)
    predict.add_argument("--imgsz", type=int, default=960)
    predict.add_argument("--device", default=None)
    predict.set_defaults(run=predict_video)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("epochs", "imgsz", "batch"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1")
    args.run(args)


if __name__ == "__main__":
    main()
