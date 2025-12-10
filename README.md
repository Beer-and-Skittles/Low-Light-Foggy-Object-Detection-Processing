# Low-Light & Foggy Object Detection -- Enhacement + YOLO Task-Aware Training

This repository contains a full pipeline for evaluating classical enhancement methods (CLAHE, Gamma, Retinex) and training task-aware neural enhancers (AOD-Net and a ResNet-based enhancer) for improving object detection on low-light (ExDark) and foggy (RTTS) datasets.

The final output evaluates detection performance using YOLOv8 (mAP50 / mAP50-95).

## Dataset Setup
Your directory structure must look like this:
```
/ExDark
/ExDark_Anno
/RTTS
/RTTS_Anno
/<project_directory>   ← this repo
```
Download these folders from links:
* [ExDark](https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view?usp=sharing)
* [ExDark_Anno](https://drive.google.com/file/d/1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i/view?usp=sharing)
* [RTTS](https://utexas.app.box.com/s/2yekra41udg9rgyzi3ysi513cps621qz) (download foler `"RTTS/JPEGImages"` and change name to `"RTTS"`)
* [RTTS_Anno](https://utexas.app.box.com/s/2yekra41udg9rgyzi3ysi513cps621qz) (download foler `"RTTS/Annotations"` and change name to `"RTTS_Anno"`)

## Repository High-Level Overview
Inside `<project_directory>`:
```
Low-Light-Foggy-Object-Detection-Processing/
│
├── configs/
│   ├── aod_train.yaml            # AOD-Net training config
│   ├── resnet_train.yaml         # ResNet training config
│   └── default.yaml              # (optional) shared config template
│
├── data/
│   ├── splits/                   # Pre-generated train/test split lists
│   │   ├── combined_train.txt
│   │   ├── combined_test.txt
│   │   ├── exdark_train.txt
│   │   ├── exdark_test.txt
│   │   ├── rtts_train.txt
│   │   ├── rtts_test.txt
│   │   └── sample.txt
│   └── class_names.json          # List of categories used by YOLO
│
├── datasets/
│   ├── __init__.py
│   ├── exdark.py                 # Loads ExDark images + txt annotations
│   ├── rtts.py                   # Loads RTTS images + xml annotations
│   ├── anno_parsers.py           # Converts annotations → YOLO label format
│   └── taskaware_dataset.py      # Dataset feeding YOLO supervision for enhancers
│
├── models/
│   ├── aod_net.py                # AOD-Net implementation
│   └── resnet_enhancer.py        # ResNet18 encoder–decoder enhancement network
│
├── preprocessing/
│   └── baselines_pil.py          # CLAHE, Gamma, Retinex implemented using PIL
│
├── scripts/
│   ├── split_data.py             # Create train/test splits
│   ├── run_baseline_cli.py       # Run classical baselines + YOLO mAP eval
│   ├── eval_aod_cli.py           # Evaluate trained AOD-Net
│   └── eval_resnet_cli.py        # Evaluate trained ResNet enhancer
│
├── train/
│   ├── eval_baselines.py         # (legacy) baseline evaluation code
│   ├── train_aod_taskaware.py    # Full AOD-Net training loop (YOLO-supervised)
│   └── train_resnet_taskaware.py # Full ResNet training loop (YOLO-supervised)
│
├── yolo/
│   ├── detector.py               # Thin wrapper for YOLOv8 inference
│   └── metrics.py                # Helper utils (optional)
│
├── outputs/

```
### configs/
All model + training hyperparameters (img size, lr, batch size, weights).

### data/splits/
Text files listing `<image>\t<label>` pairs for training/evaluation.
You never need to regenerate these unless rearranging datasets.

### datasets/
All dataset-specific logic:
* listing image/annotation pairs
* parsing ExDark `.txt` and RTTS `.xml`
* preparing tensors and labels for the enhancer training loop

### models/
Two enhancement architectures:
* AODNet (lightweight)
* ResNetEnhancer (heavy encoder–decoder)

### preprocessing/
Classical enhancement baselines (CLAHE / Gamma / Retinex).

### scripts/
User-facing CLI commands:
* baseline evaluation
* AOD eval
* ResNet eval
* dataset splitting

### train/
Core training loops:
* train AOD-Net
* train ResNet
* (legacy) baseline scripts

### yolo/
Small YOLO utilities for inference and metric computation.

### outputs/
Stores:
* checkpoints (*.pt)
* temporary YOLO-formatted datasets for eval
* sample enhanced images

### runs/detect/
YOLO’s native output directory for mAP tables, confusion matrices, and plots.


## AOD-Net
AOD-Net is a lightweight convolutional enhancement network frequently used in dehazing.
Structure:
* Shallow stack of conv layers
* Multi-branch feature fusion
* Final RGB reconstruction
* Adds tiny computational cost (~0.5M parameters)

Code:
```
models/aod_net.py
```

AOD-Net is trained with task-aware loss:
```
Loss = λ_det * YOLO_Detection_Loss(enhanced_image, labels)
        + λ_identity * L1(enhanced_image - original_image)
```
YOLO loss encourages better object detection after enhancement.

Configure AOD training in
```
configs/aod_train.yaml
```

## ResNet Enhancer
This model converts ResNet18 into an enhancement network:
* ResNet18 encoder
* 4-stage decoder (upsampling + conv)
* Skip-like progression improving global + local features
* Output passed through sigmoid → [0,1]

Code:
```
models/resnet_enhancer.py
```

Same task-aware detection loss as AOD-Net:
```
Loss = λ_det * YOLO_Loss + λ_identity * L1
```

Configure the ResNet training in:
```
configs/resnet_train.yaml
```

## Flow & Checklist
Below are tasks with exact commands where outputs will be stored

### 0. **[DONE]** Split Dataset
Split dataset:
```
python -m scripts.split_data --cfg configs/default.yaml
```

Training/testing splits are already created in the `data/splits` folder:
```
data/splits/combined_train.txt
data/splits/combined_test.txt
data/splits/exdark_test.txt
data/splits/exdark_train.txt
data/splits/rtts_test.txt
data/splits/rtts_train.txt
sample.txt
```
Each of these files contain:
```
/full/path/to/image.png    /full/path/to/annotation
```

### 1. Run Baselines
#### Preparation
```
python -m scripts.run_baseline_cli \
--stage prepare \
--test_list data/splits/<dataset> \
--method <method> \

```

#### Evaluation
```
python -m scripts.run_baseline_cli \
--stage eval \
--temp_dir outputs/<temp_dir>
```

* `<dataset>` = `exdark_test.txt` / `rtts_test.txt` / `comtined_test.txt`
* `<method>` = `clahe` / `gamma` / `retinex`

#### Preparation & Evaluation
```
python -m scripts.run_baseline_cli \
--stage all \
--test_list data/splits/<dataset> \
--method <method>
```

Outputs goes to
```
outputs/<temp_dir>/                ← temp YOLO dataset
runs/detect/<temp_dir>/            ← YOLO mAP results
```

### 2. **[TODO]** Train & Evaluate AOD-Net
Train Resnet:
```
python -m scripts.train_aod_taskaware \
  --cfg configs/aod_train.yaml \
  --log_path outputs/aod_loss.txt
```

Adjust input and model configuration in:
```
configs/aod_train.yaml
```

Evaluate AOD-Net:
```
python -m scripts.eval_aod_cli \
  --split_list data/splits/combined_test.txt \
  --ckpt outputs/aod_checkpoints/aod_epoch<N>.pt \
  --class_names_json data/class_names.json \
  --device cpu
```
* `<dataset>` = `exdark_test.txt` / `rtts_test.txt` / `comtined_test.txt`
* `<device>` = `cpu` / `cuda`


Results saved to:
```
outputs/tmp_aod_eval/
runs/detect/val*/            ← YOLO mAP tables
```

### 3. **[TODO]** Train & Evaluate ResNet Enhancer
```
python -m scripts.train_resnet_taskaware \
  --cfg configs/resnet_train.yaml
```

Adjust input and model configuration in:
```
configs/resnet_train.yaml
```

Evaluate ResNet:
```
python -m scripts.eval_resnet_cli \
  --split_list data/splits/combined_test.txt \
  --ckpt outputs/resnet_checkpoints/resnet_epoch<N>.pt \
  --class_names_json data/class_names.json \
  --device cpu
```
* `<dataset>` = `exdark_test.txt` / `rtts_test.txt` / `comtined_test.txt`
* `<device>` = `cpu` / `cuda`


Results saved to:
```
outputs/tmp_resnet_eval/
runs/detect/val*/            ← YOLO mAP tables
```
### Miscelaneous
* printing out yolo class names
  ```
    python3 - << 'PY'
from ultralytics import YOLO
m = YOLO("yolov8n.pt")
print("\nYOLO class names:")
for k, v in m.model.names.items():
    print(k, ":", v)
PY
  ```