from os import path

import onnxruntime as rt

from app.src.AI.ai_utils import get_detections

RELATIONSHIP_FOLDER = path.dirname(path.abspath(__file__))
MODEL_PATH = path.join(RELATIONSHIP_FOLDER, "relationship_model.onnx")
PROVIDERS = ["CPUExecutionProvider"]
OUTPUTS = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
MIN_SCORE = 0.5
sess = rt.InferenceSession(MODEL_PATH, providers=PROVIDERS)
RELATIONSHIP_CLASSES = [
    "1M",
    "11",
    "01",
    "0M",
]


def get_relationships(img):
    return get_detections(img, sess, OUTPUTS, MIN_SCORE, RELATIONSHIP_CLASSES)
