from os import path

import cv2
import numpy as np

TEXT_FOLDER = path.dirname(path.abspath(__file__))
MODEL_PATH = path.join(TEXT_FOLDER, "VGG_CTC.onnx")

text_recognition_model = cv2.dnn.readNet(MODEL_PATH)


def recognize_text(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(
        gray_image, size=(100, 32), mean=127.5, scalefactor=1 / 127.5
    )
    text_recognition_model.setInput(blob)
    result = text_recognition_model.forward()
    return decode_text(result)


def decode_text(scores):
    text = ""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += "-"

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != "-" and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return "".join(char_list)
