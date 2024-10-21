from os import path

import cv2

from ..src.ERmodel import get_ERmodel

TEST_FOLDER = path.dirname(path.abspath(__file__))
EXAMPLE_1_JPG_PATH = path.join(TEST_FOLDER, "img", "example_1.jpg")
EXAMPLE_2_JPG_PATH = path.join(TEST_FOLDER, "img", "example_2.jpg")
EXAMPLE_3_JPG_PATH = path.join(TEST_FOLDER, "img", "example_3.jpg")


def test_example1():
    img = cv2.imread(EXAMPLE_1_JPG_PATH)
    ermodel = get_ERmodel(img=img)
    assert len(ermodel["entities"]) == 4
    assert len(ermodel["connections"]) == 3


def test_example2():
    img = cv2.imread(EXAMPLE_2_JPG_PATH)
    ermodel = get_ERmodel(img=img)
    assert len(ermodel["entities"]) == 2
    assert len(ermodel["connections"]) == 1


def test_example3():
    img = cv2.imread(EXAMPLE_3_JPG_PATH)
    ermodel = get_ERmodel(img=img)
    assert len(ermodel["entities"]) > 10
    assert len(ermodel["connections"]) > 13
