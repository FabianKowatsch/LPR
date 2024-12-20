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
data_path: ./data/testing_set/images
lpd_checkpoint_path: checkpoints/test/LPD.pt
recognizer: tesseract # parseq | easyocr | tesseract
language: en
parseq_img_size: [32, 128]
verbose: True
visualize: True
```

## Dataset

Specify dataset parameters in config/data.yaml. See [YOLO dataset docs](https://docs.ultralytics.com/datasets/detect/) for additional parameters.

```
names:
- License Plate
nc: 1
train: ../../data/testing_set/images
test: ../../data/testing_set/images
val: ../../data/testing_set/images
```

### Training

Specify training parameters in config/train.yaml. See [YOLO training docs](https://docs.ultralytics.com/modes/train/#train-settings) for additional parameters.

```
model: ../checkpoints/pretrained/yolo11n.pt
data: ../config/data.yaml
epochs: 200
imgsz: 640
batch: 16
device: 0
pretrained: True
single_cls: True
```

