from os import path

import numpy as np
import onnxruntime as rt

ENTITY_FOLDER = path.dirname(path.abspath(__file__))
MODEL_PATH = path.join(ENTITY_FOLDER, "entity_model.onnx")
PROVIDERS = ["CPUExecutionProvider"]
OUTPUTS = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
MIN_SCORE = 0.8
sess = rt.InferenceSession(MODEL_PATH, providers=PROVIDERS)


def get_entity_boxes(img):
    img_array = np.array(img)
    img_array = np.expand_dims(img_array.astype(np.uint8), axis=0)
    num_detections, detection_boxes, detection_scores, detection_classes = sess.run(
        OUTPUTS, {"input_tensor": img_array}
    )
    boxes = filter_boxes_by_score(detection_boxes[0], detection_scores[0])
    height = img_array.shape[1]
    width = img_array.shape[2]
    return get_real_region(height, width, boxes)


def filter_boxes_by_score(detection_boxes, detection_scores):
    boxes = []
    for idx, score in enumerate(detection_scores):
        if score > MIN_SCORE:
            boxes.append(detection_boxes[idx])
    return boxes


def get_real_region(height, width, boxes):
    return [box * [height, width, height, width] for box in boxes]
