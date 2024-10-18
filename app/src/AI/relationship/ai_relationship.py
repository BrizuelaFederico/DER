from os import path

import numpy as np
import onnxruntime as rt

RELATIONSHIP_FOLDER = path.dirname(path.abspath(__file__))
MODEL_PATH = path.join(RELATIONSHIP_FOLDER, "relationship_model.onnx")
PROVIDERS = ["CPUExecutionProvider"]
OUTPUTS = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
MIN_SCORE = 0.5
sess = rt.InferenceSession(MODEL_PATH, providers=PROVIDERS)
RELATIONSHIP_TYPE = [
    "1M",
    "11",
    "01",
    "0M",
]


def get_relationships(img):
    img_array = np.array(img)
    img_array = np.expand_dims(img_array.astype(np.uint8), axis=0)

    num_detections, detection_boxes, detection_scores, detection_classes = sess.run(
        OUTPUTS, {"input_tensor": img_array}
    )

    detections = filter_detections_by_score(
        detection_boxes[0], detection_scores[0], detection_classes[0]
    )

    return convert_detections_to_listdict(detections, img_array)


def filter_detections_by_score(detection_boxes, detection_scores, detection_classes):
    detections = list(zip(detection_boxes, detection_scores, detection_classes))
    return filter(lambda detection: detection[1] > MIN_SCORE, detections)


def convert_detections_to_listdict(detections, img_array):
    listdict = []
    for index, detection in enumerate(detections):
        listdict.append(convert_detection_to_dict(detection, index, img_array))
    return listdict


def convert_detection_to_dict(detection, index, img_array):
    return {
        "index": index,
        "type": RELATIONSHIP_TYPE[int(detection[2] - 1)],
        "box": get_region(detection[0], img_array),
    }


def get_region(box, img_array):
    height = img_array.shape[1]
    width = img_array.shape[2]
    return {
        "xmin": int(box[1] * width),
        "xmax": int(box[3] * width),
        "ymin": int(box[0] * height),
        "ymax": int(box[2] * height),
    }
