# Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking
This repository contains the code (in Keras) for "Fast and Robust UAV to UAV Detection and Tracking" by Jing Li.

## Citation
If you use our software or our data sets, please cite:

J. Li, D. Ye, M. Kolsch, J. Wachs and C. Bouman, "Fast and Robust UAV to UAV Detection and Tracking from Video" in IEEE Transactions on Emerging Topics in Computing.
doi: 10.1109/TETC.2021.3104555
url: https://doi.ieeecomputersociety.org/10.1109/TETC.2021.3104555

@ARTICLE {9519550,
author = {J. Li and D. Ye and M. Kolsch and J. P. Wachs and C. A. Bouman},
journal = {IEEE Transactions on Emerging Topics in Computing},
title = {Fast and Robust UAV to UAV Detection and Tracking from Video},
year = {5555},
volume = {},
number = {01},
issn = {2168-6750},
pages = {1-1},
keywords = {target tracking;cameras;detectors;unmanned aerial vehicles;optical imaging;radar tracking;optical detectors},
doi = {10.1109/TETC.2021.3104555},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {aug}
}

https://www.computer.org/csdl/journal/ec/5555/01/09519550/1wc8Vbe1r7G


@inproceedings{li2016multi,<br>
  title={Multi-target detection and tracking from a single camera in Unmanned Aerial Vehicles (UAVs)},<br>
  author={Li, Jing and Ye, Dong Hye and Chung, Timothy and Kolsch, Mathias and Wachs, Juan and Bouman, Charles},<br>
  booktitle={2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},<br>
  pages={4992--4997},<br>
  year={2016},<br>
  organization={IEEE}
}



## Content
1. Requirements
2. Usage
3. Results
4. Contacts
## Requirements

The recommended development environment is CPU-only and is defined in `environment-cpu.yml` (Python 3.11). The original `environment.yml` is a historical Linux/CUDA setup and is not suitable for macOS.



## Usage
### 
Create and activate the CPU environment, then run the example:

```sh
conda env create -f environment-cpu.yml
conda activate uav-cpu
python main.py
```

On Apple Silicon, use `environment-cpu-macos-arm64.lock.yml` instead to recreate the exact tested package set.

The saved AdaBoost classifiers emit a scikit-learn version warning because they were serialized with 0.19.1. The demo has been smoke-tested on video 14; output parity with the original environment has not been established. See `MODERNIZATION.md` for details.


### YOLO detection-only experiment

The YOLO workflow is separate from `main.py` and does not change the existing detector/tracker. It uses the existing per-frame annotations to create a single-class YOLO dataset, splitting by video so frames from a clip cannot leak across splits. With all 50 clips present, the default deterministic split has 35 training videos, 7 validation videos, and 8 test videos; video 14 is reserved for test. Frames are sampled every 10 frames by default to keep the generated dataset manageable. Use `--frame-stride 1` to extract every annotated frame.

The default 960-pixel input helps preserve detail for small UAV boxes; reduce `--imgsz` or `--batch` if training exceeds available memory. Annotation rows are aligned by their `time_layer` values. Video 39 has no annotation row for frame 162, which is skipped rather than assigning the next frame's box to it.

Install the optional dependency in the active project environment:

```sh
python -m pip install -r requirements-yolo.txt
```

Prepare the dataset, fine-tune the small pretrained YOLO11 model, and evaluate on the held-out test split:

```sh
python yolo_detector.py prepare
python yolo_detector.py train
python yolo_detector.py evaluate
```

Generate frame-by-frame predictions for video 14:

```sh
python yolo_detector.py predict --video-id 14
```

This writes `Experiment_Results/YOLO/txt/14_dt_yolo.txt` using the baseline's `time_layer` and `(top, left, bottom, right)` box convention. It is detection-only: there is no association or interpolation between frames. Use `--weights`, `--confidence`, `--imgsz`, or `--device` to change inference settings. The script downloads pretrained YOLO weights on first use if they are not already cached. Test metrics cover sampled test frames; compare baseline results on exactly those frames for a detector-only comparison. The generated dataset, trained weights, run results, and prediction file are local artifacts, not source files to commit.


The existing `main.py` still only runs its video-14 demo. This supports a first same-video comparison, not a complete baseline benchmark; a full comparison needs baseline outputs for the same held-out videos and sampled frames.


### ConvNeXt detection experiment

`train_convnext.py` trains torchvision ConvNeXt Tiny, Small, and Base as multi-box UAV detectors. It reads `Clip_<id>.mov` alongside `Video_<id>_gt.txt`, whose rows are keyed by `time_layer` and whose boxes use `(top, left, bottom, right)` coordinates. Empty rows are treated as background frames; multiple boxes are supported. The default split is deterministic and clip-level (80% train, 20% validation), and annotated frames are extracted sequentially every 10 frames into a reusable cache. Use `--frame-stride 1` to train on every annotated frame.

Install a PyTorch build suitable for your CPU/CUDA environment and the matching torchvision build, then run:

```sh
python train_convnext.py --device cuda --batch-size 8
```

The script downloads torchvision ImageNet weights on first use. Reduce `--batch-size` or `--image-size` if GPU memory is insufficient; use `--no-pretrained` to avoid downloading weights. Best checkpoints (`convnext_{tiny,small,base}_best.pth`), extracted-frame caches, and `validation_summary.json` are written under `Experiment_Results/ConvNeXt/`. Validation reports loss and detection precision, recall, F1, and mean IoU at a 0.5 IoU threshold. The validation split is for model selection and is not an independent test set.

## Results
### 
Saved under Experiment_Results folder:<br>
.txt file with detected bounding boxes,<br>
videos with groundtruth and detection results

## Dataset
https://engineering.purdue.edu/~bouman/UAV_Dataset/

## Contacts
lijinginxjtu@gmail.com

Any discussions or concerns are welcomed!
