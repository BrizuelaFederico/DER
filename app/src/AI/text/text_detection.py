from os import path

import cv2

TEXT_FOLDER = path.dirname(path.abspath(__file__))
MODEL_PATH = path.join(TEXT_FOLDER, "DB_TD500_resnet18.onnx")
MIN_SCORE = 0.8

text_detection_model = cv2.dnn.TextDetectionModel_DB(MODEL_PATH)
text_detection_model.setInputScale(1.0 / 255.0)
text_detection_model.setInputSize(736, 736)
text_detection_model.setInputMean([122.67891434, 116.66876762, 104.00698793])


def detect_text(img):
    detections, confidences = text_detection_model.detectTextRectangles(img)
    return get_detections_by_score(detections, confidences)


def get_detections_by_score(detections, confidences):
    confidence_detections = []

    for index, confidence in enumerate(confidences):
        if confidence > MIN_SCORE:
            confidence_detections.append(detections[index])

    return confidence_detections
