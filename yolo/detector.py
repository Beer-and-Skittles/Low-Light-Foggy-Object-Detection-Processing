from ultralytics import YOLO
import numpy as np
from PIL import Image

class YoloDetector:
    def __init__(self, model="yolov8n.pt", device="cpu"):
        self.model = YOLO(model)
        self.device = device

    def infer_img(self, img_np):
        """
        img_np: RGB numpy array (H,W,3), uint8
        """
        return self.model.predict(img_np, device=self.device, verbose=False)[0]
