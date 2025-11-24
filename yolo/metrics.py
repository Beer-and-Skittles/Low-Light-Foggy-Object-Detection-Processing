from ultralytics import YOLO
import numpy as np
from pathlib import Path

def compute_map(model, data_yaml, split="test"):
    """
    Runs YOLO's built-in validation to compute mAP@0.5.
    data_yaml: path to YOLO-style dataset YAML
    """
    res = model.val(data=data_yaml, split=split, verbose=False)
    return res.results_dict["metrics/mAP50"]
