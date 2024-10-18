from os import path

import cv2

from ..src.AI.line.ai_line import recognize_line

TEST_FOLDER = path.dirname(path.abspath(__file__))

BOTTOM_LEFT_JPG_PATH = path.join(TEST_FOLDER, "img", "lines", "bottom-left.jpg")
BOTTOM_RIGHT_JPG_PATH = path.join(TEST_FOLDER, "img", "lines", "bottom-right.jpg")
CROSSING_JPG_PATH = path.join(TEST_FOLDER, "img", "lines", "crossing.jpg")
HORIZONTAL_JPG_PATH = path.join(TEST_FOLDER, "img", "lines", "horizontal.jpg")
NONE_JPG_PATH = path.join(TEST_FOLDER, "img", "lines", "none.jpg")
UP_LEFT_JPG_PATH = path.join(TEST_FOLDER, "img", "lines", "up-left.jpg")
UP_RIGHT_JPG_PATH = path.join(TEST_FOLDER, "img", "lines", "up-right.jpg")
VERTICAL_JPG_PATH = path.join(TEST_FOLDER, "img", "lines", "vertical.jpg")


def test_bottom_left():
    img = cv2.imread(BOTTOM_LEFT_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class == "BOTTOM-LEFT"


def test_bottom_right():
    img = cv2.imread(BOTTOM_RIGHT_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class == "BOTTOM-RIGHT"


def test_crossing():
    img = cv2.imread(CROSSING_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class == "CROSSING"


def test_horizontal():
    img = cv2.imread(HORIZONTAL_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class == "HORIZONTAL"


def test_none():
    img = cv2.imread(NONE_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class == "NONE"


def test_up_left():
    img = cv2.imread(UP_LEFT_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class == "UP-LEFT"


def test_up_right():
    img = cv2.imread(UP_RIGHT_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class == "UP-RIGHT"


def test_vertical():
    img = cv2.imread(VERTICAL_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class == "VERTICAL"


def test_not_vertical():
    img = cv2.imread(HORIZONTAL_JPG_PATH)
    line_class = recognize_line(img)
    assert line_class != "VERTICAL"
