from os import path

import cv2

LINE_FOLDER = path.dirname(path.abspath(__file__))
MODEL_PATH = path.join(LINE_FOLDER, "line_model.onnx")
LINE_CLASSES = [
    "CROSSING",
    "HORIZONTAL",
    "BOTTOM-LEFT",
    "TOP-LEFT",
    "NONE",
    "BOTTOM-RIGHT",
    "TOP-RIGHT",
    "VERTICAL",
]

recognizer = cv2.dnn.readNet(MODEL_PATH)


def recognize_line(img):
    gray_img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    blob = cv2.dnn.blobFromImage(gray_img, size=(50, 50))
    recognizer.setInput(blob)
    result = recognizer.forward()
    return get_line_class(result[0])


def get_line_class(recognizer_result):
    for index, value in enumerate(recognizer_result):
        if int(value) == 1:
            return LINE_CLASSES[index]
