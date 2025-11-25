# <Project>/datasets/class_mapping.py
import json
from ultralytics import YOLO
from datasets.alias_mapping import ALIAS_TO_YOLO


def build_class2yolo_id(class_names_json, yolo_weights="yolov8n.pt"):
    """
    Converts dataset labels (including aliases) into YOLO class IDs.
    Returns:
      class2id: dict mapping dataset label -> YOLO class index
      class_names: list of canonical dataset labels (original order)
    """
    # Load your class list
    class_names = json.load(open(class_names_json))
    class_names = [c.lower() for c in class_names]

    # Load YOLO names
    yolo_model = YOLO(yolo_weights)
    yolo_name2id = {name.lower(): idx for idx, name in yolo_model.model.names.items()}

    class2id = {}

    for name in class_names:
        alias = name.lower()

        # Map alias → canonical YOLO name if needed
        canonical = ALIAS_TO_YOLO.get(alias, alias)

        if canonical not in yolo_name2id:
            print(f"[WARN] Label '{alias}' → canonical '{canonical}' not in YOLO classes. Skipping.")
            continue

        class2id[alias] = yolo_name2id[canonical]

    return class2id, class_names
