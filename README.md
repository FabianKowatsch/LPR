# License Plate Recognition

## Installation

Requires a Nvidia GPU with CUDA 11.8 support. Dependencies can be installed via conda:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate lpr
```

## Usage

Testing:

```
python main.py
```

Training:

```
python train.py
```

## Configuration

### General

Change LPR specific parameters in config/config.yaml:
```
data_path: ./data/test/images
lpd_checkpoint_path: checkpoints/test/LPD_best.pt
verbose: True
visualize: True
recognizer: 
  type: tesseract # parseq, easyocr, tesseract
  language: en
  parseq_img_size: [32, 128]
  tesseract_engine: 3 # 0=Legacy, 1=LSTM, 2=Legacy+LSTM 3=default
  tesseract_segmentation: 7 # For single lines
upscaler:
  type: "LANCZOS4" # LANCZOS4, bilinear, bicubic
  scale_factor: 2
image_processing:
  grayscale: True
  denoising: False
  normalize: True
  contrast: False
  thresholding: False
  threshold_value: 125
  rotation: False
  max_rotation_angle: 15
```

## Dataset

Specify dataset parameters in config/data.yaml. See [YOLO dataset docs](https://docs.ultralytics.com/datasets/detect/) for additional parameters.

```
names:
- License Plate
nc: 1
train: ../../data/train/images
test: ../../data/test/images
val: ../../data/valid/images
```

### Training

Specify training parameters in config/train.yaml. See [YOLO training docs](https://docs.ultralytics.com/modes/train/#train-settings) for additional parameters.

```
model: checkpoints/pretrained/yolo11n.pt
data: config/data.yaml
epochs: 400
imgsz: 640
batch: 32
device: 0
pretrained: True
single_cls: True
val: True
```

