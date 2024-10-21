import cv2
import numpy as np

from .text_detection import detect_text
from .text_recognition import recognize_text


def get_entity_text(img):
    text_detections = detect_text(img)

    if len(text_detections) == 0:
        return {"name": "", "attributes": []}

    # from top to bottom
    ordered_text_detections = sorted(text_detections, key=lambda x: x[0][1])

    text_recognitions = list(
        map(
            lambda text_detection: recognize_text_for_detection(text_detection, img),
            ordered_text_detections,
        )
    )

    return {"name": text_recognitions[0], "attributes": text_recognitions[1:]}


def recognize_text_for_detection(text_detection, img):
    increased_text_area = increase_text_area(text_detection)
    vertices = cv2.boxPoints(increased_text_area)
    cropped_text = four_points_transform(img, vertices)
    return recognize_text(cropped_text)


# to be able to better recognize the text
def increase_text_area(text_detection):
    return (
        (text_detection[0][0], text_detection[0][1]),
        (text_detection[1][0] * 1.1, text_detection[1][1] * 1.1),
        0.0,
    )


def four_points_transform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array(
        [
            [0, outputSize[1] - 1],
            [0, 0],
            [outputSize[0] - 1, 0],
            [outputSize[0] - 1, outputSize[1] - 1],
        ],
        dtype="float32",
    )

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result
