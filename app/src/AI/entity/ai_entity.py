from os import path

import numpy as np
import onnxruntime as rt

from app.src.AI.ai_utils import get_detections

ENTITY_FOLDER = path.dirname(path.abspath(__file__))
MODEL_PATH = path.join(ENTITY_FOLDER, "entity_model.onnx")
PROVIDERS = ["CPUExecutionProvider"]
OUTPUTS = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
MIN_SCORE = 0.8
sess = rt.InferenceSession(MODEL_PATH, providers=PROVIDERS)
ENTITY_CLASSES = [
    "ENTITY",
]


def get_entities(img):
    return get_detections(img, sess, OUTPUTS, MIN_SCORE, ENTITY_CLASSES)
